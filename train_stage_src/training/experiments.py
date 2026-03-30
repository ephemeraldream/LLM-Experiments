from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    def __init__(
        self,
        experiment_dir: str | Path | None = None,
        experiment_name: str | None = None,
        tensorboard_dir: str | Path | None = None,
        root_dir: str | Path = "runs",
    ):
        self.run_dir = self._resolve_run_dir(
            experiment_dir=experiment_dir,
            experiment_name=experiment_name,
            root_dir=root_dir,
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.config_path = self.run_dir / "config.json"
        self.summary_path = self.run_dir / "summary.json"
        self.latest_checkpoint_path = self.run_dir / "latest.pt"
        self.best_checkpoint_path = self.run_dir / "best.pt"
        self.tensorboard_dir = Path(tensorboard_dir) if tensorboard_dir is not None else self.run_dir / "tensorboard"
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
        self.summary: dict[str, Any] = {}

    def write_config(self, config: dict[str, Any]) -> None:
        self.config_path.write_text(json_dumps(config), encoding="utf-8")

    def log_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        payload = {"step": int(step), **metrics}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json_dumps(payload, indent=None) + "\n")

        for key, value in metrics.items():
            if isinstance(value, bool):
                self.writer.add_scalar(key, int(value), step)
            elif isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)
        self.writer.flush()

    def update_summary(self, **summary: Any) -> None:
        self.summary.update(summary)
        self.summary_path.write_text(json_dumps(self.summary), encoding="utf-8")

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()

    @staticmethod
    def infer_run_dir_from_checkpoint(checkpoint_path: str | Path | None) -> Path | None:
        if checkpoint_path is None:
            return None
        checkpoint_path = Path(checkpoint_path)
        candidate = checkpoint_path.parent
        if (candidate / "config.json").exists() and (candidate / "summary.json").exists():
            return candidate
        return None

    def _resolve_run_dir(
        self,
        experiment_dir: str | Path | None,
        experiment_name: str | None,
        root_dir: str | Path,
    ) -> Path:
        if experiment_dir is not None:
            return Path(experiment_dir)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        slug = experiment_name or "gptzero"
        safe_slug = slug.replace("/", "-").replace(" ", "-")
        return Path(root_dir) / f"{safe_slug}-{timestamp}"


def json_dumps(payload: Any, indent: int | None = 2) -> str:
    return json.dumps(payload, indent=indent, default=json_default)


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")
