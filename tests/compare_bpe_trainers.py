"""
CLI benchmark for comparing the original BPE trainer against the fast one.

Examples:
    uv run python -m cs336_basics.tokenizer.compare_bpe_trainers
    uv run python -m cs336_basics.tokenizer.compare_bpe_trainers --input-path tests/fixtures/corpus.en --max-chars 0
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import tempfile
import time

from cs336_basics.tokenizer.BPEtokenizer import BPE
from cs336_basics.tokenizer.BPETokenizerFast import FastBPE


@dataclass(slots=True)
class TrainingRunReport:
    label: str
    elapsed_seconds: float
    vocab_size: int
    num_merges: int


def compare_old_and_fast_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
    max_chars: int | None = 100_000,
    fast_use_parallel: bool = True,
    num_processes: int | None = None,
    verify_outputs: bool = True,
) -> tuple[TrainingRunReport, TrainingRunReport]:
    with _prepare_benchmark_input(input_path=input_path, max_chars=max_chars) as benchmark_input_path:
        old_vocab, old_merges, old_report = _measure_training_run(
            label="Original BPE",
            trainer=BPE(
                input_path=benchmark_input_path,
                vocab_size=vocab_size,
                special_tokens=special_tokens,
            ),
            train_kwargs={},
        )
        fast_vocab, fast_merges, fast_report = _measure_training_run(
            label="Fast BPE",
            trainer=FastBPE(
                input_path=benchmark_input_path,
                vocab_size=vocab_size,
                special_tokens=special_tokens,
            ),
            train_kwargs={
                "use_parallel": fast_use_parallel,
                "num_processes": num_processes,
            },
        )

        if verify_outputs:
            _assert_training_outputs_match(
                old_vocab=old_vocab,
                old_merges=old_merges,
                fast_vocab=fast_vocab,
                fast_merges=fast_merges,
            )

        return old_report, fast_report


def _measure_training_run(
    label: str,
    trainer,
    train_kwargs: dict,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]], TrainingRunReport]:
    start_time = time.perf_counter()
    vocab, merges = trainer.train_bpe(**train_kwargs)
    elapsed_seconds = time.perf_counter() - start_time

    report = TrainingRunReport(
        label=label,
        elapsed_seconds=elapsed_seconds,
        vocab_size=len(vocab),
        num_merges=len(merges),
    )
    return vocab, merges, report


def _assert_training_outputs_match(
    old_vocab: dict[int, bytes],
    old_merges: list[tuple[bytes, bytes]],
    fast_vocab: dict[int, bytes],
    fast_merges: list[tuple[bytes, bytes]],
) -> None:
    if old_merges != fast_merges:
        raise AssertionError("Old and fast trainers produced different merge sequences.")

    if old_vocab != fast_vocab:
        raise AssertionError("Old and fast trainers produced different vocabularies.")


def _format_reports_as_table(old_report: TrainingRunReport, fast_report: TrainingRunReport) -> str:
    speedup = old_report.elapsed_seconds / fast_report.elapsed_seconds if fast_report.elapsed_seconds else float("inf")
    header = f"{'Trainer':<14} {'Seconds':>10} {'Vocab':>8} {'Merges':>8}"
    separator = "-" * len(header)
    old_line = (
        f"{old_report.label:<14} {old_report.elapsed_seconds:>10.3f} "
        f"{old_report.vocab_size:>8} {old_report.num_merges:>8}"
    )
    fast_line = (
        f"{fast_report.label:<14} {fast_report.elapsed_seconds:>10.3f} "
        f"{fast_report.vocab_size:>8} {fast_report.num_merges:>8}"
    )
    summary_line = f"Speedup: {speedup:.2f}x"
    return "\n".join([header, separator, old_line, fast_line, "", summary_line])


@contextmanager
def _prepare_benchmark_input(input_path: str | Path, max_chars: int | None):
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input corpus not found: {input_path}")

    if max_chars is None or max_chars <= 0:
        yield input_path
        return

    with open(input_path, encoding="utf-8") as f:
        text = f.read(max_chars)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as tmp_file:
        tmp_file.write(text)
        tmp_path = Path(tmp_file.name)

    try:
        yield tmp_path
    finally:
        tmp_path.unlink(missing_ok=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    default_input_path = repo_root / "data" / "TinyStoriesV2-GPT4-train.txt"

    parser = argparse.ArgumentParser(description="Benchmark the original and fast BPE trainers.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=default_input_path,
        help="Path to the corpus file to benchmark.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=500,
        help="Target vocabulary size used for both trainers.",
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
        default=100_000,
        help="If positive, benchmark on only the first N characters of the file. Use 0 to benchmark the full file.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Number of worker processes to use for the fast trainer.",
    )
    parser.add_argument(
        "--serial-fast",
        action="store_true",
        help="Disable the fast trainer's multiprocessing pretokenization path.",
    )
    parser.add_argument(
        "--skip-output-check",
        action="store_true",
        help="Skip the exact output equality check between the old and fast trainers.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    special_tokens = args.special_token or ["<|endoftext|>"]
    old_report, fast_report = compare_old_and_fast_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        max_chars=args.max_chars,
        fast_use_parallel=not args.serial_fast,
        num_processes=args.num_processes,
        verify_outputs=not args.skip_output_check,
    )

    print(f"Benchmark input: {args.input_path}")
    if args.max_chars and args.max_chars > 0:
        print(f"Using only the first {args.max_chars} characters for a fair side-by-side benchmark.")
    else:
        print("Using the full input file.")
    print(_format_reports_as_table(old_report, fast_report))


if __name__ == "__main__":
    main()
