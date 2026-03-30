from __future__ import annotations

import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from train_stage_src.core.model_config import ModelConfig, model_config_from_dict
from train_stage_src.core.training_config import OptimizerConfig, TrainConfig, TrainResult
from train_stage_src.model.GPTZero import GPTZero
from train_stage_src.training.data_loader import get_batch
from train_stage_src.training.experiments import ExperimentLogger
from train_stage_src.training.prepare import TokenArray, load_token_memmap
from train_stage_src.utils.checkpointing import (
    CheckpointPayload,
    load_checkpoint_payload,
    restore_checkpoint_payload,
    save_checkpoint,
)
from train_stage_src.utils.grad_clipping import clip_gradients_l2_norm_
from train_stage_src.utils.log_loss import log_loss
from train_stage_src.utils.lr_scheduling import get_lr_cosine_schedule
from train_stage_src.utils.optimizer import apply_optimizer_config, build_optimizer, set_learning_rate


class Trainer:
    def __init__(
        self,
        train_dataset: TokenArray,
        valid_dataset: TokenArray,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        train_config: TrainConfig,
        checkpoint_path: str | os.PathLike[str] | None = None,
    ):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.train_config = train_config
        self.device = torch.device(train_config.device)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.rng = np.random.default_rng(train_config.seed)
        torch.manual_seed(train_config.seed)

        self.model = GPTZero(
            vocab_size=model_config.vocab_size,
            context_length=model_config.context_length,
            num_layers=model_config.num_layers,
            d_model=model_config.d_model,
            num_heads=model_config.num_heads,
            d_ff=model_config.d_ff,
            theta=model_config.theta,
            tie_embeddings=model_config.tie_embeddings,
            init_std=model_config.init_std,
            device=self.device,
        )
        self.optimizer = build_optimizer(
            self.model,
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps,
        )
        self.experiment = ExperimentLogger(
            experiment_dir=train_config.experiment_dir,
            experiment_name=train_config.experiment_name,
            tensorboard_dir=train_config.tensorboard_logdir,
        )
        self.start_iteration = 0
        self.last_train_loss = float("nan")
        self.last_valid_loss: float | None = None
        self.best_valid_loss: float | None = None
        self.latest_checkpoint_path: Path | None = None
        self.best_checkpoint_path: Path | None = None

    def restore(self, payload: CheckpointPayload) -> None:
        self.start_iteration = restore_checkpoint_payload(payload, model=self.model, optimizer=self.optimizer)
        apply_optimizer_config(
            self.optimizer,
            lr=self.optimizer_config.lr,
            weight_decay=self.optimizer_config.weight_decay,
            betas=self.optimizer_config.betas,
            eps=self.optimizer_config.eps,
        )
        extra_state = payload.get("extra_state") or {}
        trainer_state = extra_state.get("trainer_state") or {}
        self.best_valid_loss = trainer_state.get("best_valid_loss")
        best_checkpoint_path = trainer_state.get("best_checkpoint_path")
        latest_checkpoint_path = trainer_state.get("latest_checkpoint_path")
        if best_checkpoint_path:
            self.best_checkpoint_path = Path(best_checkpoint_path)
        if latest_checkpoint_path:
            self.latest_checkpoint_path = Path(latest_checkpoint_path)

    def train(self) -> TrainResult:
        self.experiment.write_config(
            {
                "model_config": asdict(self.model_config),
                "optimizer_config": asdict(self.optimizer_config),
                "train_config": asdict(self.train_config),
            }
        )
        self.experiment.update_summary(
            experiment_dir=str(self.experiment.run_dir),
            tensorboard_dir=str(self.experiment.tensorboard_dir),
            resumed_from_iteration=self.start_iteration,
        )

        start_time = time.perf_counter()
        try:
            for iteration in range(self.start_iteration, self.train_config.max_iters):
                completed_iters = iteration + 1
                current_lr = learning_rate_at(iteration, self.optimizer_config, self.train_config.max_iters)
                set_learning_rate(self.optimizer, current_lr)

                x, y = get_batch(
                    dataset=self.train_dataset,
                    batch_size=self.train_config.batch_size,
                    context_length=self.model_config.context_length,
                    device=self.device,
                    rng=self.rng,
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss = log_loss(self.model(x), y)
                loss.backward()
                grad_norm = clip_gradients_l2_norm_(
                    self.model.parameters(),
                    self.optimizer_config.grad_clip,
                )
                self.optimizer.step()

                self.last_train_loss = float(loss.item())
                elapsed = time.perf_counter() - start_time
                tokens_seen = completed_iters * self.train_config.batch_size * self.model_config.context_length
                tokens_per_second = tokens_seen / max(elapsed, 1e-8)
                self.experiment.log_metrics(
                    completed_iters,
                    {
                        "loss/train": self.last_train_loss,
                        "lr": current_lr,
                        "grad/global_norm": float(grad_norm.detach().cpu()),
                        "perf/tokens_per_second": tokens_per_second,
                        "perf/elapsed_seconds": elapsed,
                    },
                )

                if should_log(completed_iters, self.train_config.log_every):
                    print(
                        f"iter={completed_iters:>6} "
                        f"train_loss={self.last_train_loss:.4f} "
                        f"lr={current_lr:.6g} "
                        f"tok_s={tokens_per_second:.1f} "
                        f"elapsed={elapsed:.1f}s"
                    )

                if should_eval(completed_iters, self.train_config.max_iters, self.train_config.eval_every):
                    self.last_valid_loss = self.evaluate()
                    self.experiment.log_metrics(
                        completed_iters,
                        {
                            "loss/valid": self.last_valid_loss,
                        },
                    )
                    print(f"iter={completed_iters:>6} valid_loss={self.last_valid_loss:.4f}")
                    if self.best_valid_loss is None or self.last_valid_loss < self.best_valid_loss:
                        self.best_valid_loss = self.last_valid_loss
                        self.best_checkpoint_path = self.experiment.best_checkpoint_path
                        self._save_checkpoint(self.best_checkpoint_path, completed_iters)

                if should_save(completed_iters, self.train_config.max_iters, self.train_config.save_every):
                    self.latest_checkpoint_path = self.experiment.latest_checkpoint_path
                    self._save_checkpoint(self.latest_checkpoint_path, completed_iters)
                    if self.checkpoint_path is not None and self.checkpoint_path != self.latest_checkpoint_path:
                        self._save_checkpoint(self.checkpoint_path, completed_iters)

                self.experiment.update_summary(
                    iteration=completed_iters,
                    train_loss=self.last_train_loss,
                    valid_loss=self.last_valid_loss,
                    best_valid_loss=self.best_valid_loss,
                    latest_checkpoint_path=str(self.latest_checkpoint_path) if self.latest_checkpoint_path else None,
                    best_checkpoint_path=str(self.best_checkpoint_path) if self.best_checkpoint_path else None,
                    checkpoint_path=str(self.checkpoint_path) if self.checkpoint_path else None,
                )
        finally:
            self.experiment.close()

        checkpoint_path = self.checkpoint_path or self.latest_checkpoint_path
        return TrainResult(
            iteration=self.train_config.max_iters,
            train_loss=self.last_train_loss,
            valid_loss=self.last_valid_loss,
            best_valid_loss=self.best_valid_loss,
            checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
            best_checkpoint_path=str(self.best_checkpoint_path) if self.best_checkpoint_path is not None else None,
            experiment_dir=str(self.experiment.run_dir),
        )

    def evaluate(self) -> float:
        losses: list[float] = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.train_config.eval_steps):
                x, y = get_batch(
                    dataset=self.valid_dataset,
                    batch_size=self.train_config.batch_size,
                    context_length=self.model_config.context_length,
                    device=self.device,
                    rng=self.rng,
                )
                losses.append(float(log_loss(self.model(x), y).item()))
        self.model.train()
        return float(np.mean(losses))

    def _save_checkpoint(self, path: Path, iteration: int) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            iteration=iteration,
            out=path,
            extra_state=self._checkpoint_extra_state(),
        )
        return path

    def _checkpoint_extra_state(self) -> dict:
        return {
            "model_config": asdict(self.model_config),
            "optimizer_config": asdict(self.optimizer_config),
            "train_config": asdict(self.train_config),
            "trainer_state": {
                "best_valid_loss": self.best_valid_loss,
                "best_checkpoint_path": str(self.best_checkpoint_path) if self.best_checkpoint_path else None,
                "latest_checkpoint_path": str(self.latest_checkpoint_path) if self.latest_checkpoint_path else None,
                "experiment_dir": str(self.experiment.run_dir),
            },
        }


def evaluate_model(
    model: torch.nn.Module,
    dataset: TokenArray,
    batch_size: int,
    context_length: int,
    eval_steps: int,
    device: str | torch.device,
    rng: np.random.Generator,
) -> float:
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for _ in range(eval_steps):
            x, y = get_batch(
                dataset=dataset,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
                rng=rng,
            )
            losses.append(float(log_loss(model(x), y).item()))
    model.train()
    return float(np.mean(losses))


def train_gpt_zero(
    train_tokens: str | os.PathLike[str] | TokenArray,
    valid_tokens: str | os.PathLike[str] | TokenArray,
    model_config: ModelConfig,
    optimizer_config: OptimizerConfig,
    train_config: TrainConfig,
    checkpoint_path: str | os.PathLike[str] | None = None,
    resume: bool = False,
) -> TrainResult:
    train_array = load_token_memmap(train_tokens) if isinstance(train_tokens, (str, os.PathLike)) else train_tokens
    valid_array = load_token_memmap(valid_tokens) if isinstance(valid_tokens, (str, os.PathLike)) else valid_tokens

    checkpoint_path_obj = Path(checkpoint_path) if checkpoint_path is not None else None
    resume_payload: CheckpointPayload | None = None
    if resume and checkpoint_path_obj is not None and checkpoint_path_obj.exists():
        resume_payload = load_checkpoint_payload(checkpoint_path_obj)
        extra_state = resume_payload.get("extra_state") or {}
        saved_model_config = extra_state.get("model_config")
        if saved_model_config is not None:
            model_config = model_config_from_dict(
                saved_model_config,
                state_dict=resume_payload.get("model_state_dict"),
            )
        inferred_run_dir = ExperimentLogger.infer_run_dir_from_checkpoint(checkpoint_path_obj)
        if inferred_run_dir is not None and train_config.experiment_dir is None:
            train_config = replace(train_config, experiment_dir=str(inferred_run_dir))

    trainer = Trainer(
        train_dataset=train_array,
        valid_dataset=valid_array,
        model_config=model_config,
        optimizer_config=optimizer_config,
        train_config=train_config,
        checkpoint_path=checkpoint_path_obj,
    )
    if resume_payload is not None:
        trainer.restore(resume_payload)
        print(f"Resumed from checkpoint at iteration {trainer.start_iteration}.")
    return trainer.train()


def learning_rate_at(iteration: int, config: OptimizerConfig, max_iters: int) -> float:
    cosine_cycle_iters = config.resolved_cosine_cycle_iters(max_iters)
    if cosine_cycle_iters is None or cosine_cycle_iters <= 0:
        return config.lr
    if cosine_cycle_iters <= config.warmup_iters:
        return config.min_lr if iteration >= config.warmup_iters else config.lr
    return get_lr_cosine_schedule(
        it=iteration,
        max_learning_rate=config.lr,
        min_learning_rate=config.min_lr,
        warmup_iters=config.warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
    )


def should_eval(iteration: int, max_iters: int, eval_every: int) -> bool:
    if iteration == max_iters:
        return True
    return eval_every > 0 and iteration % eval_every == 0


def should_log(iteration: int, log_every: int) -> bool:
    if iteration == 1:
        return True
    return log_every > 0 and iteration % log_every == 0


def should_save(iteration: int, max_iters: int, save_every: int) -> bool:
    if iteration == max_iters:
        return True
    return save_every > 0 and iteration % save_every == 0
