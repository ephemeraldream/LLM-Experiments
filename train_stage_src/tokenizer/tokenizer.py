from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from os import PathLike
from pathlib import Path

import regex as re

from train_stage_src.tokenizer.BPEtokenizer import (
    GPT2_PRETOKEN_PATTERN,
    gpt2_byte_decoder,
    gpt2_bytes_to_unicode,
)


def _pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    if len(word) < 2:
        return set()
    out: set[tuple[str, str]] = set()
    prev = word[0]
    for cur in word[1:]:
        out.add((prev, cur))
        prev = cur
    return out


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self._byte_encoder = gpt2_bytes_to_unicode()
        self._byte_decoder = gpt2_byte_decoder()
        self._bytes_to_id = {b: i for i, b in vocab.items()}
        self._merge_ranks: dict[tuple[str, str], int] = {}
        for rank, (left_b, right_b) in enumerate(merges):
            left = "".join(self._byte_encoder[x] for x in left_b)
            right = "".join(self._byte_encoder[x] for x in right_b)
            self._merge_ranks[(left, right)] = rank
        self._special_set = frozenset(self.special_tokens)
        if self.special_tokens:
            escaped = sorted(self.special_tokens, key=len, reverse=True)
            self._special_pattern = re.compile("(" + "|".join(re.escape(t) for t in escaped) + ")")
        else:
            self._special_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | PathLike,
        merges_filepath: str | PathLike,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        byte_decoder = gpt2_byte_decoder()

        with open(vocab_filepath, encoding="utf-8") as vocab_f:
            serialized_vocab: dict[str, int] = json.load(vocab_f)

        vocab = {
            token_id: bytes(byte_decoder[character] for character in token_text)
            for token_text, token_id in serialized_vocab.items()
        }

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as merges_f:
            for line in merges_f:
                cleaned_line = line.rstrip()
                parts = cleaned_line.split(" ")
                if not cleaned_line or len(parts) != 2:
                    continue

                left_text, right_text = parts
                merges.append(
                    (
                        bytes(byte_decoder[character] for character in left_text),
                        bytes(byte_decoder[character] for character in right_text),
                    )
                )

        if special_tokens:
            existing_vocab_values = set(vocab.values())
            for special_token in special_tokens:
                special_token_bytes = special_token.encode("utf-8")
                if special_token_bytes not in existing_vocab_values:
                    vocab[len(vocab)] = special_token_bytes
                    existing_vocab_values.add(special_token_bytes)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        if self._special_pattern is None:
            return self._encode_ordinary(text)
        ids: list[int] = []
        for part in self._special_pattern.split(text):
            if not part:
                continue
            if part in self._special_set:
                ids.append(self._bytes_to_id[part.encode("utf-8")])
            else:
                ids.extend(self._encode_ordinary(part))
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in GPT2_PRETOKEN_PATTERN.findall(text):
            token = "".join(self._byte_encoder[b] for b in piece.encode("utf-8"))
            for bpe_piece in self._bpe(token).split(" "):
                if bpe_piece:
                    b = bytes(self._byte_decoder[c] for c in bpe_piece)
                    ids.append(self._bytes_to_id[b])
        return ids

    def _bpe(self, token: str) -> str:
        word = tuple(token)
        pairs = _pairs(word)
        if not pairs:
            return token
        while True:
            bigram = min(pairs, key=lambda pair: (self._merge_ranks.get(pair, float("inf")), pair))
            if bigram not in self._merge_ranks:
                break
            first, second = bigram
            new_word: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _pairs(word)
            if not pairs:
                break
        return " ".join(word)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fixtures_dir = repo_root / "tests" / "fixtures"
    vocab_path = fixtures_dir / "gpt2_vocab.json"
    merges_path = fixtures_dir / "gpt2_merges.txt"
    special_tokens = ["<|endoftext|>"]

    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=special_tokens,
    )

    print(f"Loaded vocab from: {vocab_path}")
    print(f"Loaded merges from: {merges_path}")
    print(f"Vocab size: {len(tokenizer.vocab)}")
    print(f"Number of merges: {len(tokenizer.merges)}")
    print(f"First vocab entry: {tokenizer.vocab[0]!r}")
    print(f"First merge: {tokenizer.merges[0]!r}")
    print(f"Has <|endoftext|>: {b'<|endoftext|>' in set(tokenizer.vocab.values())}")


if __name__ == "__main__":
    main()
