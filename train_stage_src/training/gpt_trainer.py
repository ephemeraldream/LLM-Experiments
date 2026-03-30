from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from train_stage_src.core.model_config import ModelConfig
from train_stage_src.core.training_config import OptimizerConfig, TrainConfig, TrainResult
from train_stage_src.training.data_loader import data_loader, get_batch
from train_stage_src.training.prepare import (
    TokenArray,
    infer_vocab_size,
    load_token_memmap,
    load_tokenizer_from_meta,
    meta_path_for,
    prepare_memmap_dataset,
    resolve_path,
)
from train_stage_src.training.trainer import Trainer, evaluate_model, train_gpt_zero
from train_stage_src.utils.checkpointing import (
    CheckpointPayload,
    CheckpointTarget,
    load_checkpoint,
    load_checkpoint_payload,
    move_optimizer_state_to_parameter_devices,
    restore_checkpoint_payload,
    save_checkpoint,
)

__all__ = [
    "CheckpointPayload",
    "CheckpointTarget",
    "ModelConfig",
    "OptimizerConfig",
    "TokenArray",
    "TrainConfig",
    "TrainResult",
    "Trainer",
    "data_loader",
    "evaluate_model",
    "get_batch",
    "infer_vocab_size",
    "load_checkpoint",
    "load_checkpoint_payload",
    "load_token_memmap",
    "load_tokenizer_from_meta",
    "main",
    "meta_path_for",
    "move_optimizer_state_to_parameter_devices",
    "prepare_memmap_dataset",
    "resolve_path",
    "restore_checkpoint_payload",
    "save_checkpoint",
    "train_gpt_zero",
]


def _run_prepare(args: argparse.Namespace) -> None:
    output_path, metadata_path = prepare_memmap_dataset(
        tokenizer_meta_path=args.tokenizer_meta,
        input_text_path=args.input_text,
        output_path=args.output_npy,
        force=args.force,
        show_progress=not args.no_progress,
    )
    print(f"Saved token array to: {output_path}")
    print(f"Saved dataset metadata to: {metadata_path}")


def _run_train(args: argparse.Namespace) -> None:
    model_config = ModelConfig(
        vocab_size=infer_vocab_size(
            args.train_npy,
            explicit_vocab_size=args.vocab_size,
            tokenizer_meta_path=args.tokenizer_meta,
        ),
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        tie_embeddings=not args.no_tie_embeddings,
        init_std=args.init_std,
    )
    optimizer_config = OptimizerConfig(
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.cosine_cycle_iters,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        grad_clip=args.grad_clip,
    )
    train_config = TrainConfig(
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        eval_every=args.eval_every,
        eval_steps=args.eval_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        device=args.device,
        seed=args.seed,
        experiment_name=args.experiment_name,
        experiment_dir=args.experiment_dir,
        tensorboard_logdir=args.tensorboard_dir,
    )

    result = train_gpt_zero(
        train_tokens=args.train_npy,
        valid_tokens=args.valid_npy,
        model_config=model_config,
        optimizer_config=optimizer_config,
        train_config=train_config,
        checkpoint_path=args.checkpoint_path,
        resume=args.resume,
    )
    print(json.dumps(asdict(result), indent=2))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compact GPTZero helper: prepare memory-mapped token arrays and train with checkpointing."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Tokenize a text file into a memory-mapped .npy array.")
    prepare_parser.add_argument("--tokenizer-meta", type=Path, required=True)
    prepare_parser.add_argument("--input-text", type=Path, required=True)
    prepare_parser.add_argument("--output-npy", type=Path, required=True)
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.add_argument("--no-progress", action="store_true")

    train_parser = subparsers.add_parser("train", help="Train GPTZero on .npy token arrays.")
    train_parser.add_argument("--train-npy", type=Path, required=True)
    train_parser.add_argument("--valid-npy", type=Path, required=True)
    train_parser.add_argument("--vocab-size", type=int, default=None)
    train_parser.add_argument("--tokenizer-meta", type=Path, default=None)
    train_parser.add_argument("--context-length", type=int, default=128)
    train_parser.add_argument("--num-layers", type=int, default=4)
    train_parser.add_argument("--d-model", type=int, default=256)
    train_parser.add_argument("--num-heads", type=int, default=8)
    train_parser.add_argument("--d-ff", type=int, default=None)
    train_parser.add_argument("--theta", type=float, default=10_000.0)
    train_parser.add_argument("--init-std", type=float, default=0.02)
    train_parser.add_argument("--no-tie-embeddings", action="store_true")
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--max-iters", type=int, default=500)
    train_parser.add_argument("--eval-every", type=int, default=100)
    train_parser.add_argument("--eval-steps", type=int, default=20)
    train_parser.add_argument("--log-every", type=int, default=10)
    train_parser.add_argument("--save-every", type=int, default=100)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--min-lr", type=float, default=3e-5)
    train_parser.add_argument("--warmup-iters", type=int, default=200)
    train_parser.add_argument("--cosine-cycle-iters", type=int, default=None)
    train_parser.add_argument("--weight-decay", type=float, default=0.1)
    train_parser.add_argument("--beta1", type=float, default=0.9)
    train_parser.add_argument("--beta2", type=float, default=0.95)
    train_parser.add_argument("--eps", type=float, default=1e-8)
    train_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--checkpoint-path", type=Path, default=None)
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--experiment-name", type=str, default=None)
    train_parser.add_argument("--experiment-dir", type=Path, default=None)
    train_parser.add_argument("--tensorboard-dir", type=Path, default=None)

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "prepare":
        _run_prepare(args)
        return
    if args.command == "train":
        _run_train(args)
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
