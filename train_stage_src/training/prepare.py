from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm  # type: ignore[import-untyped]

from train_stage_src.tokenizer.tokenizer import Tokenizer

TokenArray = np.ndarray

REPO_ROOT = Path(__file__).resolve().parents[2]


def meta_path_for(array_path: str | os.PathLike[str] | Path) -> Path:
    return Path(array_path).with_suffix(".meta.json")


def load_tokenizer_from_meta(meta_path: str | os.PathLike[str]) -> tuple[Tokenizer, dict]:
    meta_path = Path(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    prefix = meta_path.stem.removesuffix("_meta")

    raw_vocab_path = meta.get("safe_vocab_path", f"{prefix}_vocab_bytes.json")
    raw_merges_path = meta.get("safe_merges_path", f"{prefix}_merges_bytes.json")
    vocab_path = resolve_path(raw_vocab_path, meta_path.parent)
    merges_path = resolve_path(raw_merges_path, meta_path.parent)

    raw_vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    raw_merges = json.loads(merges_path.read_text(encoding="utf-8"))
    vocab = {int(token_id): bytes.fromhex(token_hex) for token_id, token_hex in raw_vocab.items()}
    merges = [(bytes.fromhex(left_hex), bytes.fromhex(right_hex)) for left_hex, right_hex in raw_merges]
    tokenizer = Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=list(meta.get("special_tokens", [])),
    )
    return tokenizer, meta


def resolve_path(path_like: str | os.PathLike[str], base_dir: Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path

    candidates = [
        REPO_ROOT / path,
        base_dir / path,
        base_dir / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def prepare_memmap_dataset(
    tokenizer_meta_path: str | os.PathLike[str],
    input_text_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    force: bool = False,
    show_progress: bool = True,
) -> tuple[Path, Path]:
    input_text_path = Path(input_text_path)
    output_path = Path(output_path)
    metadata_path = meta_path_for(output_path)

    if not force and output_path.exists() and metadata_path.exists():
        return output_path, metadata_path

    tokenizer, tokenizer_meta = load_tokenizer_from_meta(tokenizer_meta_path)
    token_count = count_tokens_in_text(input_text_path, tokenizer, show_progress=show_progress)
    max_token_id = max(tokenizer.vocab)
    dtype = smallest_uint_dtype(max_token_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_array = open_memmap(output_path, mode="w+", dtype=dtype, shape=(token_count,))

    total_bytes = input_text_path.stat().st_size
    write_index = 0
    with input_text_path.open(encoding="utf-8") as handle, tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        desc=f"write {output_path.name}",
        disable=not show_progress,
    ) as progress:
        for line in handle:
            encoded = np.asarray(tokenizer.encode(line), dtype=dtype)
            next_index = write_index + len(encoded)
            output_array[write_index:next_index] = encoded
            write_index = next_index
            progress.update(len(line.encode("utf-8")))
    output_array.flush()

    metadata = {
        "source_text_path": str(input_text_path),
        "output_path": str(output_path),
        "tokenizer_name": tokenizer_meta.get("name"),
        "tokenizer_meta_path": str(Path(tokenizer_meta_path)),
        "token_count": token_count,
        "dtype": np.dtype(dtype).name,
        "max_token_id": max_token_id,
        "vocab_size": len(tokenizer.vocab),
        "special_tokens": list(tokenizer_meta.get("special_tokens", [])),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path, metadata_path


def load_token_memmap(path: str | os.PathLike[str]) -> TokenArray:
    path = Path(path)
    array = np.load(path, mmap_mode="r")
    if array.ndim != 1:
        raise ValueError(f"Expected a 1D token array, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"Expected an integer token array, got dtype={array.dtype}")
    return array


def infer_vocab_size(
    dataset_path: str | os.PathLike[str],
    explicit_vocab_size: int | None = None,
    tokenizer_meta_path: str | os.PathLike[str] | None = None,
) -> int:
    if explicit_vocab_size is not None:
        return explicit_vocab_size

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Token dataset not found: {dataset_path}. "
            "Create it first with the `prepare` command."
        )

    metadata_path = meta_path_for(dataset_path)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "vocab_size" in metadata:
            return int(metadata["vocab_size"])
        if "max_token_id" in metadata:
            return int(metadata["max_token_id"]) + 1

    if tokenizer_meta_path is not None:
        tokenizer_meta = json.loads(Path(tokenizer_meta_path).read_text(encoding="utf-8"))
        if "vocab_size" in tokenizer_meta:
            return int(tokenizer_meta["vocab_size"])

    raise ValueError(
        "vocab_size is required when dataset metadata is unavailable. "
        "Pass --vocab-size, or pass --tokenizer-meta, or prepare the dataset with the `prepare` command first."
    )


def count_tokens_in_text(path: Path, tokenizer: Tokenizer, show_progress: bool) -> int:
    total_bytes = path.stat().st_size
    total_tokens = 0
    with path.open(encoding="utf-8") as handle, tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        desc=f"count {path.name}",
        disable=not show_progress,
    ) as progress:
        for line in handle:
            total_tokens += len(tokenizer.encode(line))
            progress.update(len(line.encode("utf-8")))
    return total_tokens


def smallest_uint_dtype(max_token_id: int):
    if max_token_id <= np.iinfo(np.uint8).max:
        return np.uint8
    if max_token_id <= np.iinfo(np.uint16).max:
        return np.uint16
    if max_token_id <= np.iinfo(np.uint32).max:
        return np.uint32
    return np.uint64
