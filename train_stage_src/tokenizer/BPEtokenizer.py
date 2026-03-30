from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from os import PathLike
from pathlib import Path
import tempfile

import regex as re

GPT2_PRETOKEN_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    return dict(zip(bs, characters))


@lru_cache
def gpt2_byte_decoder() -> dict[str, int]:
    return {v: k for k, v in gpt2_bytes_to_unicode().items()}


Word = tuple[int, ...]
Pair = tuple[int, int]


class BPE:
    def __init__(self, input_path: str | PathLike[str], vocab_size: int, special_tokens: list[str]):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = list(dict.fromkeys(special_tokens))
        self.merges: list[tuple[bytes, bytes]] = []

        if self.special_tokens:
            escaped_tokens = [re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)]
            self.special_token_pattern = re.compile("|".join(escaped_tokens))
        else:
            self.special_token_pattern = None

    def _iter_text_segments(self, text: str):
        if self.special_token_pattern is None:
            yield text
            return

        for segment in self.special_token_pattern.split(text):
            if segment:
                yield segment

    def _build_pretoken_counts(self) -> Counter[Word]:
        with open(self.input_path, encoding="utf-8") as f:
            text = f.read()

        pretoken_counts: Counter[Word] = Counter()
        for segment in self._iter_text_segments(text):
            for match in GPT2_PRETOKEN_PATTERN.finditer(segment):
                token_bytes = match.group(0).encode("utf-8")
                pretoken_counts[tuple(token_bytes)] += 1

        return pretoken_counts

    def _count_occurrences(self, word_counts: Counter[Word]) -> dict[Pair, int]:
        counts: dict[Pair, int] = defaultdict(int)
        for word, frequency in word_counts.items():
            for i in range(len(word) - 1):
                counts[(word[i], word[i + 1])] += frequency
        return counts

    def _select_best_pair(self, counts: dict[Pair, int], vocab: dict[int, bytes]) -> Pair:
        return max(counts, key=lambda pair: (counts[pair], vocab[pair[0]], vocab[pair[1]]))

    def _merge_pair(self, word_counts: Counter[Word], pair_to_merge: Pair, new_token_id: int) -> Counter[Word]:
        merged_counts: Counter[Word] = Counter()
        for word, frequency in word_counts.items():
            merged_word: list[int] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair_to_merge:
                    merged_word.append(new_token_id)
                    i += 2
                else:
                    merged_word.append(word[i])
                    i += 1
            merged_counts[tuple(merged_word)] += frequency
        return merged_counts

    def train_bpe(
        self,
        use_parallel: bool = False,
        num_processes: int | None = None,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        del use_parallel, num_processes

        vocab = {idx: bytes([idx]) for idx in range(256)}
        for token in self.special_tokens:
            vocab[len(vocab)] = token.encode("utf-8")

        if self.vocab_size < len(vocab):
            raise ValueError(
                f"vocab_size={self.vocab_size} is smaller than the initial vocabulary size={len(vocab)}"
            )

        self.merges = []
        word_counts = self._build_pretoken_counts()
        next_token_id = len(vocab)

        while len(vocab) < self.vocab_size:
            counts = self._count_occurrences(word_counts)
            if not counts:
                break

            chosen_pair = self._select_best_pair(counts, vocab)
            left_bytes = vocab[chosen_pair[0]]
            right_bytes = vocab[chosen_pair[1]]
            vocab[next_token_id] = left_bytes + right_bytes
            self.merges.append((left_bytes, right_bytes))

            word_counts = self._merge_pair(word_counts, chosen_pair, next_token_id)
            next_token_id += 1

        return vocab, self.merges


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    tinystories_path = repo_root / "data" / "TinyStoriesV2-GPT4-train.txt"
    debug_chars = 10_000
    debug_vocab_size = 500
    debug_special_tokens = ["<|endoftext|>"]

    if not tinystories_path.exists():
        raise FileNotFoundError(f"TinyStories file not found: {tinystories_path}")

    debug_text = tinystories_path.read_text(encoding="utf-8")[:debug_chars]

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as tmp_file:
        tmp_file.write(debug_text)
        debug_input_path = Path(tmp_file.name)

    try:
        trainer = BPE(
            input_path=debug_input_path,
            vocab_size=debug_vocab_size,
            special_tokens=debug_special_tokens,
        )
        vocab, merges = trainer.train_bpe()

        print(f"Loaded first {debug_chars} characters from {tinystories_path.name}")
        print(f"Vocab size: {len(vocab)}")
        print(f"Num merges: {len(merges)}")
        print(f"First 10 merges: {merges[:10]}")
    finally:
        debug_input_path.unlink(missing_ok=True)