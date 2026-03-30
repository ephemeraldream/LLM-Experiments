from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

from tqdm import tqdm  # type: ignore[import-untyped]

from cs336_basics.tokenizer.BPETokenizerFast import FastBPE, _prepare_input_path, _save_training_artifacts


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    name: str
    input_path: Path
    vocab_size: int
    save_prefix: Path


@dataclass(slots=True)
class ExperimentResult:
    name: str
    status: str
    elapsed_seconds: float
    vocab_size: int
    num_merges: int
    vocab_path: Path
    merges_path: Path
    meta_path: Path


def _artifact_paths(save_prefix: Path) -> tuple[Path, Path, Path, Path, Path]:
    vocab_path = save_prefix.with_name(f"{save_prefix.name}_vocab.json")
    merges_path = save_prefix.with_name(f"{save_prefix.name}_merges.txt")
    meta_path = save_prefix.with_name(f"{save_prefix.name}_meta.json")
    safe_vocab_path = save_prefix.with_name(f"{save_prefix.name}_vocab_bytes.json")
    safe_merges_path = save_prefix.with_name(f"{save_prefix.name}_merges_bytes.json")
    return vocab_path, merges_path, meta_path, safe_vocab_path, safe_merges_path


def _default_experiments(output_dir: Path, include_owt: bool = False) -> list[ExperimentSpec]:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    specs = [
        ExperimentSpec(
            name="tinystories_10k_fast_bpe",
            input_path=data_dir / "TinyStoriesV2-GPT4-train.txt",
            vocab_size=10_000,
            save_prefix=output_dir / "tinystories_10k_fast_bpe",
        ),
        ExperimentSpec(
            name="tinystories_32k_fast_bpe",
            input_path=data_dir / "TinyStoriesV2-GPT4-train.txt",
            vocab_size=32_000,
            save_prefix=output_dir / "tinystories_32k_fast_bpe",
        ),
    ]
    if include_owt:
        specs.extend(
            [
                ExperimentSpec(
                    name="owt_10k_fast_bpe",
                    input_path=data_dir / "owt_train.txt",
                    vocab_size=10_000,
                    save_prefix=output_dir / "owt_10k_fast_bpe",
                ),
                ExperimentSpec(
                    name="owt_32k_fast_bpe",
                    input_path=data_dir / "owt_train.txt",
                    vocab_size=32_000,
                    save_prefix=output_dir / "owt_32k_fast_bpe",
                ),
            ]
        )
    return specs


def _build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    default_output_dir = repo_root / "data" / "tokenizer_outputs"

    parser = argparse.ArgumentParser(description="Train and save the full tokenizer experiment matrix.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory where tokenizer artifacts and metadata are written.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="Run only the named experiment. Repeat the flag to run multiple experiments.",
    )
    parser.add_argument(
        "--special-token",
        action="append",
        default=None,
        help="Special token to add to the vocabulary. Repeat the flag to add more than one.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="If positive, train on only the first N characters of each corpus for debugging.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if artifacts already exist.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List experiment names and exit.",
    )
    parser.add_argument(
        "--include-owt",
        action="store_true",
        help="Also include the heavy OpenWebText experiments.",
    )
    return parser


def _select_experiments(all_specs: list[ExperimentSpec], selected_names: list[str] | None) -> list[ExperimentSpec]:
    if not selected_names:
        return all_specs

    by_name = {spec.name: spec for spec in all_specs}
    missing = [name for name in selected_names if name not in by_name]
    if missing:
        available = ", ".join(sorted(by_name))
        raise ValueError(f"Unknown experiment(s): {', '.join(missing)}. Available experiments: {available}")
    return [by_name[name] for name in selected_names]


def _run_experiment(
    spec: ExperimentSpec,
    special_tokens: list[str],
    max_chars: int,
    force: bool,
    show_progress: bool,
) -> ExperimentResult:
    vocab_path, merges_path, meta_path, safe_vocab_path, safe_merges_path = _artifact_paths(spec.save_prefix)
    if (
        not force
        and vocab_path.exists()
        and merges_path.exists()
        and meta_path.exists()
        and safe_vocab_path.exists()
        and safe_merges_path.exists()
    ):
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        return ExperimentResult(
            name=spec.name,
            status="skipped",
            elapsed_seconds=float(metadata.get("elapsed_seconds", 0.0)),
            vocab_size=int(metadata.get("vocab_size", spec.vocab_size)),
            num_merges=int(metadata.get("num_merges", max(0, spec.vocab_size - 256 - len(special_tokens)))),
            vocab_path=vocab_path,
            merges_path=merges_path,
            meta_path=meta_path,
        )

    with _prepare_input_path(spec.input_path, max_chars) as training_input_path:
        trainer = FastBPE(
            input_path=training_input_path,
            vocab_size=spec.vocab_size,
            special_tokens=special_tokens,
        )
        start_time = time.perf_counter()
        vocab, merges = trainer.train_bpe(
            show_progress=show_progress,
            progress_label=spec.name,
        )
        elapsed_seconds = time.perf_counter() - start_time

    vocab_path, merges_path = _save_training_artifacts(vocab, merges, spec.save_prefix)
    metadata = {
        "name": spec.name,
        "status": "completed",
        "input_path": str(spec.input_path),
        "vocab_size": spec.vocab_size,
        "num_merges": len(merges),
        "special_tokens": special_tokens,
        "elapsed_seconds": elapsed_seconds,
        "max_chars": max_chars,
        "artifact_format": "latin1-json-bytes",
        "vocab_path": str(vocab_path),
        "merges_path": str(merges_path),
        "safe_vocab_path": str(safe_vocab_path),
        "safe_merges_path": str(safe_merges_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return ExperimentResult(
        name=spec.name,
        status="completed",
        elapsed_seconds=elapsed_seconds,
        vocab_size=len(vocab),
        num_merges=len(merges),
        vocab_path=vocab_path,
        merges_path=merges_path,
        meta_path=meta_path,
    )


def _print_summary(results: list[ExperimentResult]) -> None:
    header = f"{'Experiment':<28} {'Status':<10} {'Seconds':>10} {'Vocab':>8} {'Merges':>8}"
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.name:<28} {result.status:<10} "
            f"{result.elapsed_seconds:>10.2f} {result.vocab_size:>8} {result.num_merges:>8}"
        )
        print(f"  vocab:  {result.vocab_path}")
        print(f"  merges: {result.merges_path}")
        print(f"  meta:   {result.meta_path}")


def main() -> None:
    args = _build_arg_parser().parse_args()
    all_specs = _default_experiments(args.output_dir, include_owt=args.include_owt)

    if args.list:
        for spec in all_specs:
            print(spec.name)
        return

    selected_specs = _select_experiments(all_specs, args.only)
    special_tokens = args.special_token or ["<|endoftext|>"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[ExperimentResult] = []
    experiment_bar = tqdm(
        selected_specs,
        desc="tokenizer experiments",
        unit="exp",
        disable=args.no_progress,
    )
    for spec in experiment_bar:
        experiment_bar.set_postfix_str(spec.name)
        tqdm.write(f"Starting {spec.name} on {spec.input_path.name} with vocab_size={spec.vocab_size:,}")
        result = _run_experiment(
            spec=spec,
            special_tokens=special_tokens,
            max_chars=args.max_chars,
            force=args.force,
            show_progress=not args.no_progress,
        )
        tqdm.write(f"{result.status.title()}: {result.name}")
        results.append(result)

    print()
    _print_summary(results)


if __name__ == "__main__":
    main()
