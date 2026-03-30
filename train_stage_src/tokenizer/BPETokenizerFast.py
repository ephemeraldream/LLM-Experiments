"""
Simpler fast BPE trainer.

This version keeps only one meaningful optimization over `BPEtokenizer.py`:
after each merge, it rewrites only the words that actually contain the chosen
pair, instead of rescanning the whole compressed corpus every round.

That keeps the code much shorter and easier to read while still being
significantly faster than the original implementation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from functools import cache
import json
from os import PathLike
from pathlib import Path
import tempfile
import time

import regex as re  # type: ignore[import-untyped]
from tqdm import tqdm

GPT2_PRETOKEN_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

Word = tuple[int, ...]
Pair = tuple[int, int]


class FastBPE:
    def __init__(self, input_path: str | PathLike[str], vocab_size: int, special_tokens: list[str]):
        self.input_path = str(input_path)
        self.vocab_size = vocab_size
        self.special_tokens = tuple(dict.fromkeys(special_tokens))
        self.special_token_pattern = _compile_special_token_pattern(self.special_tokens)
        self.merges: list[tuple[bytes, bytes]] = []

    def train_bpe(
        self,
        use_parallel: bool = False,
        num_processes: int | None = None,
        show_progress: bool = False,
        progress_label: str | None = None,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        del use_parallel, num_processes

        vocab_as_bytes = self._build_initial_vocabulary_as_bytes()
        if self.vocab_size < len(vocab_as_bytes):
            raise ValueError(
                f"vocab_size={self.vocab_size} is smaller than the initial vocabulary size={len(vocab_as_bytes)}"
            )

        if show_progress:
            tqdm.write(f"[{progress_label or 'FastBPE'}] Reading corpus and building pretoken counts...")
        word_counts = self._build_word_counts(show_progress=show_progress, progress_label=progress_label)
        if show_progress:
            tqdm.write(f"[{progress_label or 'FastBPE'}] Building pair tables for {len(word_counts):,} unique pretokens...")
        pair_counts, pair_to_words = self._build_pair_tables(
            word_counts,
            show_progress=show_progress,
            progress_label=progress_label,
        )

        self.merges = []
        total_merges = self.vocab_size - len(vocab_as_bytes)
        merge_progress = tqdm(
            total=total_merges,
            desc=_progress_desc(progress_label, "merges"),
            unit="merge",
            disable=not show_progress,
        )
        try:
            while len(vocab_as_bytes) < self.vocab_size and pair_counts:
                pair_to_merge = self._select_best_pair(pair_counts, vocab_as_bytes)
                left_bytes = vocab_as_bytes[pair_to_merge[0]]
                right_bytes = vocab_as_bytes[pair_to_merge[1]]
                new_token_id = len(vocab_as_bytes)

                vocab_as_bytes.append(left_bytes + right_bytes)
                self.merges.append((left_bytes, right_bytes))
                self._apply_merge_in_place(
                    word_counts=word_counts,
                    pair_counts=pair_counts,
                    pair_to_words=pair_to_words,
                    pair_to_merge=pair_to_merge,
                    new_token_id=new_token_id,
                )
                merge_progress.update(1)
        finally:
            merge_progress.close()

        vocab = {token_id: token_bytes for token_id, token_bytes in enumerate(vocab_as_bytes)}
        return vocab, list(self.merges)

    def _build_initial_vocabulary_as_bytes(self) -> list[bytes]:
        vocab_as_bytes = [bytes([token_id]) for token_id in range(256)]
        for special_token in self.special_tokens:
            vocab_as_bytes.append(special_token.encode("utf-8"))
        return vocab_as_bytes

    def _build_word_counts(
        self,
        show_progress: bool = False,
        progress_label: str | None = None,
    ) -> Counter[Word]:
        """Stream the corpus line-by-line so large files (e.g. OpenWebText) do not need full RAM."""
        input_path = Path(self.input_path)
        word_counts: Counter[Word] = Counter()
        total_bytes = input_path.stat().st_size
        pretoken_progress = tqdm(
            total=total_bytes,
            desc=_progress_desc(progress_label, "pretokens"),
            unit="B",
            unit_scale=True,
            disable=not show_progress,
            leave=False,
        )
        try:
            with input_path.open(encoding="utf-8") as handle:
                for line in handle:
                    pretoken_progress.update(len(line.encode("utf-8")))
                    for segment in self._iter_text_segments(line):
                        for match in GPT2_PRETOKEN_PATTERN.finditer(segment):
                            word_counts[tuple(match.group(0).encode("utf-8"))] += 1
        finally:
            pretoken_progress.close()

        return word_counts

    def _iter_text_segments(self, text: str):
        if self.special_token_pattern is None:
            yield text
            return

        for segment in self.special_token_pattern.split(text):
            if segment:
                yield segment

    def _build_pair_tables(
        self,
        word_counts: Counter[Word],
        show_progress: bool = False,
        progress_label: str | None = None,
    ) -> tuple[Counter[Pair], dict[Pair, set[Word]]]:
        pair_counts: Counter[Pair] = Counter()
        pair_to_words: dict[Pair, set[Word]] = defaultdict(set)

        items = word_counts.items()
        if show_progress:
            items = tqdm(
                items,
                total=len(word_counts),
                desc=_progress_desc(progress_label, "pair table"),
                unit="word",
                leave=False,
            )

        for word, frequency in items:
            seen_pairs: set[Pair] = set()
            for pair in self._iter_adjacent_pairs(word):
                pair_counts[pair] += frequency
                seen_pairs.add(pair)
            for pair in seen_pairs:
                pair_to_words[pair].add(word)

        return pair_counts, pair_to_words

    def _select_best_pair(
        self,
        pair_counts: Counter[Pair],
        vocab_as_bytes: list[bytes],
    ) -> Pair:
        return max(
            pair_counts,
            key=lambda pair: (pair_counts[pair], vocab_as_bytes[pair[0]], vocab_as_bytes[pair[1]]),
        )

    def _apply_merge_in_place(
        self,
        word_counts: Counter[Word],
        pair_counts: Counter[Pair],
        pair_to_words: dict[Pair, set[Word]],
        pair_to_merge: Pair,
        new_token_id: int,
    ) -> None:
        # Fast path: only touch words that actually contain the winning pair.
        affected_words = list(pair_to_words.get(pair_to_merge, ()))
        if not affected_words:
            pair_counts.pop(pair_to_merge, None)
            pair_to_words.pop(pair_to_merge, None)
            return

        merged_word_deltas: Counter[Word] = Counter()
        for word in affected_words:
            frequency = word_counts.pop(word, 0)
            if frequency == 0:
                continue

            self._remove_word_contribution(
                word=word,
                frequency=frequency,
                pair_counts=pair_counts,
                pair_to_words=pair_to_words,
            )

            merged_word = self._merge_all_occurrences_in_word(
                word=word,
                pair_to_merge=pair_to_merge,
                new_token_id=new_token_id,
            )
            merged_word_deltas[merged_word] += frequency

        for merged_word, frequency_delta in merged_word_deltas.items():
            word_counts[merged_word] += frequency_delta
            self._add_word_contribution(
                word=merged_word,
                frequency_delta=frequency_delta,
                pair_counts=pair_counts,
                pair_to_words=pair_to_words,
            )

    def _remove_word_contribution(
        self,
        word: Word,
        frequency: int,
        pair_counts: Counter[Pair],
        pair_to_words: dict[Pair, set[Word]],
    ) -> None:
        seen_pairs: set[Pair] = set()
        for pair in self._iter_adjacent_pairs(word):
            pair_counts[pair] -= frequency
            if pair_counts[pair] <= 0:
                pair_counts.pop(pair, None)
            seen_pairs.add(pair)

        for pair in seen_pairs:
            words_with_pair = pair_to_words.get(pair)
            if words_with_pair is None:
                continue
            words_with_pair.discard(word)
            if not words_with_pair:
                pair_to_words.pop(pair, None)

    def _add_word_contribution(
        self,
        word: Word,
        frequency_delta: int,
        pair_counts: Counter[Pair],
        pair_to_words: dict[Pair, set[Word]],
    ) -> None:
        seen_pairs: set[Pair] = set()
        for pair in self._iter_adjacent_pairs(word):
            pair_counts[pair] += frequency_delta
            seen_pairs.add(pair)

        for pair in seen_pairs:
            pair_to_words[pair].add(word)

    def _merge_all_occurrences_in_word(
        self,
        word: Word,
        pair_to_merge: Pair,
        new_token_id: int,
    ) -> Word:
        merged_word: list[int] = []
        index = 0

        while index < len(word):
            if index < len(word) - 1 and (word[index], word[index + 1]) == pair_to_merge:
                merged_word.append(new_token_id)
                index += 2
            else:
                merged_word.append(word[index])
                index += 1

        return tuple(merged_word)

    def _iter_adjacent_pairs(self, word: Word):
        for index in range(len(word) - 1):
            yield (word[index], word[index + 1])


BPEFast = FastBPE


@cache
def _compile_special_token_pattern(special_tokens: tuple[str, ...]) -> re.Pattern | None:
    if not special_tokens:
        return None

    escaped_special_tokens = [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
    return re.compile("|".join(escaped_special_tokens))


def _progress_desc(progress_label: str | None, stage: str) -> str:
    if progress_label:
        return f"{progress_label} {stage}"
    return f"FastBPE {stage}"


def _save_training_artifacts(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    save_prefix: Path,
) -> tuple[Path, Path]:
    save_prefix.parent.mkdir(parents=True, exist_ok=True)
    vocab_path = save_prefix.with_name(f"{save_prefix.name}_vocab.json")
    merges_path = save_prefix.with_name(f"{save_prefix.name}_merges.txt")

    serializable_vocab = {
        token_id: token_bytes.decode("latin1")
        for token_id, token_bytes in vocab.items()
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f, ensure_ascii=False)

    with open(merges_path, "w", encoding="utf-8") as f:
        for left_bytes, right_bytes in merges:
            f.write(f"{left_bytes.decode('latin1')} {right_bytes.decode('latin1')}\n")

    safe_vocab_path = save_prefix.with_name(f"{save_prefix.name}_vocab_bytes.json")
    safe_merges_path = save_prefix.with_name(f"{save_prefix.name}_merges_bytes.json")
    with open(safe_vocab_path, "w", encoding="utf-8") as f:
        json.dump({token_id: token_bytes.hex() for token_id, token_bytes in vocab.items()}, f)
    with open(safe_merges_path, "w", encoding="utf-8") as f:
        json.dump([[left_bytes.hex(), right_bytes.hex()] for left_bytes, right_bytes in merges], f)

    return vocab_path, merges_path


@contextmanager
def _prepare_input_path(input_path: Path, max_chars: int):
    if not input_path.exists():
        raise FileNotFoundError(f"Input corpus not found: {input_path}")

    if max_chars <= 0:
        yield input_path
        return

    with open(input_path, encoding="utf-8") as f:
        truncated_text = f.read(max_chars)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as tmp_file:
        tmp_file.write(truncated_text)
        tmp_path = Path(tmp_file.name)

    try:
        yield tmp_path
    finally:
        tmp_path.unlink(missing_ok=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    default_input_path = repo_root / "data" / "TinyStoriesV2-GPT4-train.txt"
    default_save_prefix = repo_root / "data" / "tokenizer_outputs" / "tinystories_fast_bpe"

    parser = argparse.ArgumentParser(description="Train the simplified fast BPE tokenizer.")
    parser.add_argument("--input-path", type=Path, default=default_input_path)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--special-token", action="append", default=None)
    parser.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="If positive, train on only the first N characters for debugging.",
    )
    parser.add_argument(
        "--save-prefix",
        type=Path,
        default=default_save_prefix,
        help="Artifacts are written as <save-prefix>_vocab.json and <save-prefix>_merges.txt.",
    )
    parser.add_argument("--skip-save", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    special_tokens = args.special_token or ["<|endoftext|>"]
    with _prepare_input_path(args.input_path, args.max_chars) as training_input_path:
        trainer = FastBPE(
            input_path=training_input_path,
            vocab_size=args.vocab_size,
            special_tokens=special_tokens,
        )
        start_time = time.perf_counter()
        vocab, merges = trainer.train_bpe(
            show_progress=not args.no_progress,
            progress_label=args.save_prefix.name,
        )
        elapsed_seconds = time.perf_counter() - start_time

    if not args.skip_save:
        vocab_path, merges_path = _save_training_artifacts(vocab, merges, args.save_prefix)
        print(f"Saved vocab to: {vocab_path}")
        print(f"Saved merges to: {merges_path}")

    longest_token = max(vocab.values(), key=len)
    if args.max_chars > 0:
        print(f"Trained on first {args.max_chars} characters from: {args.input_path}")
    else:
        print(f"Trained on full corpus: {args.input_path}")
    print(f"Elapsed time: {elapsed_seconds:.2f} seconds")
    print(f"Final vocab size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print(f"Longest token length (bytes): {len(longest_token)}")
    print(f"Longest token preview: {longest_token.decode('utf-8', errors='replace')!r}")


if __name__ == "__main__":
    main()
