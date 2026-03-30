from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm  # type: ignore[import-untyped]

from cs336_basics.tokenizer.tokenizer import Tokenizer

DOC_DELIMITER = "<|endoftext|>"
PILE_SIZE_BYTES = 825_000_000_000


@dataclass(slots=True)
class LoadedTokenizer:
    name: str
    tokenizer: Tokenizer
    vocab_size: int
    max_token_id: int
    special_tokens: list[str]
    source_prefix: Path


@dataclass(slots=True)
class CompressionResult:
    tokenizer_name: str
    corpus_name: str
    num_docs: int
    total_bytes: int
    total_tokens: int
    bytes_per_token: float


@dataclass(slots=True)
class ThroughputResult:
    tokenizer_name: str
    corpus_name: str
    total_bytes: int
    total_tokens: int
    elapsed_seconds: float
    bytes_per_second: float
    pile_estimated_seconds: float


@dataclass(slots=True)
class SerializationResult:
    tokenizer_name: str
    dataset_name: str
    token_count: int
    dtype: str
    output_path: Path
    meta_path: Path


def _build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run the available tokenizer experiments for TinyStories.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=repo_root / "data" / "tokenizer_outputs",
        help="Directory that stores saved tokenizer artifacts.",
    )
    parser.add_argument(
        "--arrays-dir",
        type=Path,
        default=repo_root / "data" / "tokenized",
        help="Directory where serialized uint16 arrays are written.",
    )
    parser.add_argument(
        "--report-prefix",
        type=Path,
        default=repo_root / "data" / "tokenizer_outputs" / "tinystories_experiments",
        help="Report files are written as <prefix>.json and <prefix>.md.",
    )
    parser.add_argument(
        "--sample-docs",
        type=int,
        default=10,
        help="How many documents to sample from each corpus for compression experiments.",
    )
    parser.add_argument(
        "--skip-serialize",
        action="store_true",
        help="Skip serializing TinyStories train/valid into uint16 arrays.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute reports and reserialize arrays even if outputs already exist.",
    )
    return parser


def _tokenizer_prefixes(artifacts_dir: Path) -> list[Path]:
    return [
        artifacts_dir / "tinystories_10k_fast_bpe",
        artifacts_dir / "tinystories_32k_fast_bpe",
    ]


def _load_tokenizer(prefix: Path) -> LoadedTokenizer:
    meta_path = prefix.with_name(f"{prefix.name}_meta.json")
    safe_vocab_path = prefix.with_name(f"{prefix.name}_vocab_bytes.json")
    safe_merges_path = prefix.with_name(f"{prefix.name}_merges_bytes.json")

    if not meta_path.exists():
        raise FileNotFoundError(f"Tokenizer metadata not found: {meta_path}")
    if not safe_vocab_path.exists() or not safe_merges_path.exists():
        raise FileNotFoundError(
            f"Safe tokenizer artifacts not found for {prefix.name}. Expected {safe_vocab_path.name} and {safe_merges_path.name}."
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    raw_vocab = json.loads(safe_vocab_path.read_text(encoding="utf-8"))
    raw_merges = json.loads(safe_merges_path.read_text(encoding="utf-8"))

    vocab = {int(token_id): bytes.fromhex(token_hex) for token_id, token_hex in raw_vocab.items()}
    merges = [(bytes.fromhex(left_hex), bytes.fromhex(right_hex)) for left_hex, right_hex in raw_merges]
    special_tokens = list(meta.get("special_tokens", []))
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
    return LoadedTokenizer(
        name=prefix.name,
        tokenizer=tokenizer,
        vocab_size=len(vocab),
        max_token_id=max(vocab),
        special_tokens=special_tokens,
        source_prefix=prefix,
    )


def _sample_documents(path: Path, num_docs: int, desc: str) -> list[str]:
    sampled_docs: list[str] = []
    total_bytes = path.stat().st_size
    buffer = ""
    chunk_size = 1 << 20

    with path.open(encoding="utf-8") as f, tqdm(total=total_bytes, unit="B", unit_scale=True, desc=desc) as progress:
        while len(sampled_docs) < num_docs:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            progress.update(len(chunk.encode("utf-8")))
            buffer += chunk
            parts = buffer.split(DOC_DELIMITER)
            buffer = parts.pop()
            for part in parts:
                if part.strip():
                    sampled_docs.append(part)
                if len(sampled_docs) >= num_docs:
                    break
        if len(sampled_docs) < num_docs and buffer.strip():
            sampled_docs.append(buffer)

    return sampled_docs[:num_docs]


def _measure_docs_compression(tokenizer_name: str, tokenizer: Tokenizer, corpus_name: str, docs: list[str]) -> CompressionResult:
    total_bytes = 0
    total_tokens = 0
    for doc in tqdm(docs, desc=f"{tokenizer_name} on {corpus_name}", unit="doc"):
        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(tokenizer.encode(doc))
    bytes_per_token = total_bytes / total_tokens if total_tokens else 0.0
    return CompressionResult(
        tokenizer_name=tokenizer_name,
        corpus_name=corpus_name,
        num_docs=len(docs),
        total_bytes=total_bytes,
        total_tokens=total_tokens,
        bytes_per_token=bytes_per_token,
    )


def _measure_file_tokenization(tokenizer_name: str, tokenizer: Tokenizer, path: Path, corpus_name: str) -> ThroughputResult:
    total_bytes = path.stat().st_size
    total_tokens = 0
    start = time.perf_counter()
    with path.open(encoding="utf-8") as f, tqdm(total=total_bytes, unit="B", unit_scale=True, desc=f"{tokenizer_name} throughput") as progress:
        for line in f:
            total_tokens += len(tokenizer.encode(line))
            progress.update(len(line.encode("utf-8")))
    elapsed_seconds = time.perf_counter() - start
    bytes_per_second = total_bytes / elapsed_seconds if elapsed_seconds else 0.0
    pile_estimated_seconds = PILE_SIZE_BYTES / bytes_per_second if bytes_per_second else float("inf")
    return ThroughputResult(
        tokenizer_name=tokenizer_name,
        corpus_name=corpus_name,
        total_bytes=total_bytes,
        total_tokens=total_tokens,
        elapsed_seconds=elapsed_seconds,
        bytes_per_second=bytes_per_second,
        pile_estimated_seconds=pile_estimated_seconds,
    )


def _count_tokens_in_file(tokenizer_name: str, tokenizer: Tokenizer, path: Path, dataset_name: str) -> int:
    total_bytes = path.stat().st_size
    total_tokens = 0
    with path.open(encoding="utf-8") as f, tqdm(total=total_bytes, unit="B", unit_scale=True, desc=f"{tokenizer_name} count {dataset_name}") as progress:
        for line in f:
            total_tokens += len(tokenizer.encode(line))
            progress.update(len(line.encode("utf-8")))
    return total_tokens


def _serialize_dataset(
    loaded: LoadedTokenizer,
    path: Path,
    dataset_name: str,
    arrays_dir: Path,
    force: bool,
    known_token_count: int | None = None,
) -> SerializationResult:
    if loaded.max_token_id > np.iinfo(np.uint16).max:
        raise ValueError(f"{loaded.name} does not fit into uint16: max token id {loaded.max_token_id}")

    output_dir = arrays_dir / loaded.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_name}_uint16.npy"
    meta_path = output_dir / f"{dataset_name}_uint16_meta.json"

    if not force and output_path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        return SerializationResult(
            tokenizer_name=loaded.name,
            dataset_name=dataset_name,
            token_count=int(metadata["token_count"]),
            dtype=str(metadata["dtype"]),
            output_path=output_path,
            meta_path=meta_path,
        )

    token_count = known_token_count
    if token_count is None:
        token_count = _count_tokens_in_file(loaded.name, loaded.tokenizer, path, dataset_name)

    output_array = open_memmap(output_path, mode="w+", dtype=np.uint16, shape=(token_count,))
    total_bytes = path.stat().st_size
    write_index = 0
    with path.open(encoding="utf-8") as f, tqdm(total=total_bytes, unit="B", unit_scale=True, desc=f"{loaded.name} write {dataset_name}") as progress:
        for line in f:
            encoded = loaded.tokenizer.encode(line)
            encoded_array = np.asarray(encoded, dtype=np.uint16)
            output_array[write_index : write_index + len(encoded_array)] = encoded_array
            write_index += len(encoded_array)
            progress.update(len(line.encode("utf-8")))
    output_array.flush()

    metadata = {
        "tokenizer_name": loaded.name,
        "dataset_name": dataset_name,
        "source_path": str(path),
        "token_count": token_count,
        "dtype": "uint16",
        "max_token_id": loaded.max_token_id,
        "output_path": str(output_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return SerializationResult(
        tokenizer_name=loaded.name,
        dataset_name=dataset_name,
        token_count=token_count,
        dtype="uint16",
        output_path=output_path,
        meta_path=meta_path,
    )


def _seconds_to_human(seconds: float) -> str:
    if seconds == float("inf"):
        return "inf"
    hours = seconds / 3600
    if hours < 48:
        return f"{hours:.2f} hours"
    return f"{hours / 24:.2f} days"


def _write_reports(
    report_prefix: Path,
    compression_results: list[CompressionResult],
    throughput_results: list[ThroughputResult],
    serialization_results: list[SerializationResult],
    skipped_items: list[str],
) -> None:
    report_prefix.parent.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "compression_results": [asdict(result) for result in compression_results],
        "throughput_results": [asdict(result) for result in throughput_results],
        "serialization_results": [
            {
                **asdict(result),
                "output_path": str(result.output_path),
                "meta_path": str(result.meta_path),
            }
            for result in serialization_results
        ],
        "skipped_items": skipped_items,
    }
    report_prefix.with_suffix(".json").write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    tiny_results = {result.tokenizer_name: result for result in compression_results if result.corpus_name == "TinyStories sample"}
    owt_results = {result.tokenizer_name: result for result in compression_results if result.corpus_name == "OpenWebText sample"}

    lines = ["## Available Results", ""]
    for item in skipped_items:
        lines.append(f"- {item}")

    lines.extend(["", "## Part A", ""])
    if tiny_results:
        for compression_result in sorted(tiny_results.values(), key=lambda item: item.tokenizer_name):
            lines.append(
                f"- `{compression_result.tokenizer_name}` on `{compression_result.corpus_name}`: "
                f"{compression_result.bytes_per_token:.4f} bytes/token "
                f"({compression_result.total_bytes} bytes over {compression_result.total_tokens} tokens)."
            )
    else:
        lines.append("- No TinyStories compression results were produced.")

    lines.extend(["", "## Part B", ""])
    if tiny_results and owt_results:
        for tokenizer_name in sorted(owt_results):
            tiny_ratio = tiny_results[tokenizer_name].bytes_per_token
            owt_ratio = owt_results[tokenizer_name].bytes_per_token
            delta_pct = ((owt_ratio / tiny_ratio) - 1.0) * 100 if tiny_ratio else 0.0
            lines.append(
                f"- `{tokenizer_name}` compresses the OpenWebText sample to {owt_ratio:.4f} bytes/token versus "
                f"{tiny_ratio:.4f} on the TinyStories sample, a {delta_pct:+.2f}% change."
            )
    else:
        lines.append("- OpenWebText-vs-TinyStories comparison was not available.")

    lines.extend(["", "## Part C", ""])
    for throughput_result in sorted(throughput_results, key=lambda item: item.tokenizer_name):
        lines.append(
            f"- `{throughput_result.tokenizer_name}` throughput on `{throughput_result.corpus_name}`: "
            f"{throughput_result.bytes_per_second / 1e6:.2f} MB/s, so 825 GB would take about "
            f"{_seconds_to_human(throughput_result.pile_estimated_seconds)}."
        )

    lines.extend(["", "## Part D", ""])
    if serialization_results:
        for serialization_result in serialization_results:
            lines.append(
                f"- `{serialization_result.tokenizer_name}` `{serialization_result.dataset_name}` saved to "
                f"`{serialization_result.output_path}` as `{serialization_result.dtype}` with "
                f"{serialization_result.token_count:,} token IDs."
            )
        lines.append(
            "- `uint16` is sufficient because both TinyStories tokenizers have vocab sizes below 65,536."
        )
    else:
        lines.append("- Serialization was skipped.")

    report_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _build_arg_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    tiny_train_path = repo_root / "data" / "TinyStoriesV2-GPT4-train.txt"
    tiny_valid_path = repo_root / "data" / "TinyStoriesV2-GPT4-valid.txt"
    owt_train_path = repo_root / "data" / "owt_train.txt"

    loaded_tokenizers = [_load_tokenizer(prefix) for prefix in _tokenizer_prefixes(args.artifacts_dir)]
    skipped_items = [
        "OpenWebText tokenizer-dependent experiments were skipped because only TinyStories 10k/32k tokenizers are available."
    ]

    tiny_docs = _sample_documents(tiny_train_path, args.sample_docs, "Sample TinyStories docs")
    owt_docs = _sample_documents(owt_train_path, args.sample_docs, "Sample OpenWebText docs")

    compression_results: list[CompressionResult] = []
    throughput_results: list[ThroughputResult] = []
    serialization_results: list[SerializationResult] = []

    valid_token_counts: dict[str, int] = {}
    for loaded in loaded_tokenizers:
        compression_results.append(
            _measure_docs_compression(loaded.name, loaded.tokenizer, "TinyStories sample", tiny_docs)
        )
        compression_results.append(
            _measure_docs_compression(loaded.name, loaded.tokenizer, "OpenWebText sample", owt_docs)
        )

        throughput = _measure_file_tokenization(loaded.name, loaded.tokenizer, tiny_valid_path, "TinyStories valid")
        throughput_results.append(throughput)
        valid_token_counts[loaded.name] = throughput.total_tokens

        if not args.skip_serialize:
            serialization_results.append(
                _serialize_dataset(
                    loaded=loaded,
                    path=tiny_train_path,
                    dataset_name="train",
                    arrays_dir=args.arrays_dir,
                    force=args.force,
                )
            )
            serialization_results.append(
                _serialize_dataset(
                    loaded=loaded,
                    path=tiny_valid_path,
                    dataset_name="valid",
                    arrays_dir=args.arrays_dir,
                    force=args.force,
                    known_token_count=valid_token_counts[loaded.name],
                )
            )

    _write_reports(
        report_prefix=args.report_prefix,
        compression_results=compression_results,
        throughput_results=throughput_results,
        serialization_results=serialization_results,
        skipped_items=skipped_items,
    )

    print(f"Saved JSON report to: {args.report_prefix.with_suffix('.json')}")
    print(f"Saved Markdown report to: {args.report_prefix.with_suffix('.md')}")
    if serialization_results:
        print(f"Serialized arrays under: {args.arrays_dir}")


if __name__ == "__main__":
    main()
