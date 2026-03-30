from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from train_stage_src.core.model_config import model_config_from_dict
from train_stage_src.model.GPTZero import GPTZero
from train_stage_src.training.prepare import load_tokenizer_from_meta
from train_stage_src.utils.checkpointing import load_checkpoint_payload

REPO_ROOT = Path(__file__).resolve().parents[2]

# Compact terminal styling (disable with NO_COLOR=1)
_DIM = "\033[2m"
_GREEN = "\033[92m"
_RESET = "\033[0m"


def _use_color() -> bool:
    return sys.stdout.isatty() and not bool(__import__("os").environ.get("NO_COLOR"))


@dataclass(slots=True)
class GenerationParams:
    """Hyperparams for one completion (τ = temperature in the notes)."""

    temperature: float = 0.9
    top_p: float = 0.95
    max_new_tokens: int = 256


class GPTStreamDecoder:
    def __init__(
        self,
        checkpoint_path: str | Path,
        tokenizer_meta_path: str | Path | None = None,
        device: str | torch.device | None = None,
        *,
        seed: int | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self._gen = torch.Generator(device=self.device)
        if seed is not None:
            self._gen.manual_seed(seed)

        raw = load_checkpoint_payload(self.checkpoint_path)
        extra = raw.get("extra_state") or {}
        mc = extra.get("model_config")
        if not mc:
            raise ValueError(
                "Checkpoint has no extra_state['model_config']. "
                "Train with gpt_trainer (it saves model_config in checkpoints)."
            )
        self.model_config = model_config_from_dict(mc, state_dict=raw.get("model_state_dict"))

        self.model = GPTZero(
            vocab_size=self.model_config.vocab_size,
            context_length=self.model_config.context_length,
            num_layers=self.model_config.num_layers,
            d_model=self.model_config.d_model,
            num_heads=self.model_config.num_heads,
            d_ff=self.model_config.d_ff,
            theta=self.model_config.theta,
            tie_embeddings=self.model_config.tie_embeddings,
            init_std=self.model_config.init_std,
            device=self.device,
        )
        self.model.load_state_dict(raw["model_state_dict"])
        self.model.eval()

        meta_path = Path(
            tokenizer_meta_path
            or (REPO_ROOT / "data/tokenizer_outputs/tinystories_10k_fast_bpe_meta.json"),
        )
        self.tokenizer, self._tokenizer_meta = load_tokenizer_from_meta(meta_path)
        self.tokenizer_meta_path = meta_path

        eos_enc = self.tokenizer.encode("<|endoftext|>")
        self._eos_ids: tuple[int, ...] = tuple(eos_enc) if eos_enc else ()
        self._training_iteration = int(raw.get("iteration", -1))

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        return self._eos_ids

    @staticmethod
    def _sample_from_logits(
        logits_1d: torch.Tensor,
        temperature: float,
        top_p: float,
        generator: torch.Generator,
    ) -> int:
        """Next token from last-position logits (vocab_size,)."""
        if temperature <= 0.0:
            return int(logits_1d.argmax().item())

        scaled = logits_1d / max(temperature, 1e-8)
        probs = F.softmax(scaled, dim=-1)
        if top_p >= 1.0:
            return int(torch.multinomial(probs, 1, generator=generator).item())

        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        # Smallest prefix mass ≥ top_p: keep indices 0..k inclusive
        k = int((cumsum >= top_p).nonzero(as_tuple=False)[0, 0].item())
        nucleus = sorted_probs[: k + 1].clone()
        nucleus /= nucleus.sum()
        j = int(torch.multinomial(nucleus, 1, generator=generator).item())
        return int(sorted_idx[j].item())

    @torch.inference_mode()
    def iter_token_ids(
        self,
        prompt: str,
        params: GenerationParams | None = None,
    ):
        """Yield each new token id (not including prompt tokens). Stops at EOS or max_new_tokens."""
        p = params or GenerationParams()
        ids: list[int] = list(self.tokenizer.encode(prompt))
        ctx = self.model_config.context_length
        max_len = p.max_new_tokens

        for _ in range(max_len):
            window = ids[-ctx:] if len(ids) > ctx else ids
            x = torch.tensor([window], device=self.device, dtype=torch.long)
            logits = self.model(x)
            last = logits[0, -1, :]
            next_id = self._sample_from_logits(last, p.temperature, p.top_p, self._gen)
            if self._eos_ids and next_id == self._eos_ids[0]:
                break
            ids.append(next_id)
            yield next_id

    def iter_text_chunks(
        self,
        prompt: str,
        params: GenerationParams | None = None,
    ):
        """Yield UTF-8-safe text deltas (decode full sequence, emit suffix since last decode)."""
        ids: list[int] = list(self.tokenizer.encode(prompt))
        prev = self.tokenizer.decode(ids)
        for tid in self.iter_token_ids(prompt, params):
            ids.append(tid)
            full = self.tokenizer.decode(ids)
            if len(full) > len(prev):
                yield full[len(prev) :]
                prev = full

    def stream_print(
        self,
        prompt: str,
        params: GenerationParams | None = None,
        *,
        file=sys.stdout,
        show_header: bool = True,
    ) -> None:
        """Print prompt + streamed completion in one compact block (live)."""
        p = params or GenerationParams()
        color = _use_color()
        dim, grn, rst = (_DIM, _GREEN, _RESET) if color else ("", "", "")

        if show_header:
            file.write(
                f"{dim}τ={p.temperature}  top_p={p.top_p}  max_new={p.max_new_tokens}  "
                f"ctx={self.model_config.context_length}  dev={self.device}  "
                f"ckpt={self.checkpoint_path.name}  iter={self._training_iteration}{rst}\n",
            )

        file.write(f"{grn}▶{rst} {prompt}")
        file.flush()

        for chunk in self.iter_text_chunks(prompt, p):
            file.write(chunk)
            file.flush()
        file.write("\n")
        file.flush()

    def run_repl(self, default: GenerationParams | None = None) -> None:
        """Minimal REPL: type prompt, Enter; `q` to quit; `t=0.8 p=0.95 n=200` to set params."""
        params = default or GenerationParams()
        color = _use_color()
        dim, rst = (_DIM, _RESET) if color else ("", "")

        print(f"{dim}GPTStreamDecoder | {self.model_config} | /q quit | t= τ p= top_p n= max_new{rst}")
        while True:
            try:
                line = input(f"{dim}>{rst} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line or line in {"q", "quit", "/q"}:
                break
            if line.startswith("t=") or line.startswith("p=") or line.startswith("n="):
                for part in line.split():
                    if part.startswith("t="):
                        params.temperature = float(part.split("=", 1)[1])
                    elif part.startswith("p="):
                        params.top_p = float(part.split("=", 1)[1])
                    elif part.startswith("n="):
                        params.max_new_tokens = int(part.split("=", 1)[1])
                print(f"{dim}params: τ={params.temperature} top_p={params.top_p} max_new={params.max_new_tokens}{rst}")
                continue
            self.stream_print(line, params, show_header=False)
            print()


def _default_paths() -> tuple[Path, Path]:
    return (
        REPO_ROOT / "checkpoints" / "gptzero.pt",
        REPO_ROOT / "data/tokenizer_outputs/tinystories_10k_fast_bpe_meta.json",
    )


def main() -> None:
    ckpt_default, meta_default = _default_paths()
    parser = argparse.ArgumentParser(description="Stream GPTZero generation (temperature + top-p).")
    parser.add_argument("--checkpoint", type=Path, default=ckpt_default)
    parser.add_argument("--tokenizer-meta", type=Path, default=meta_default)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prompt", type=str, default=None, help="If set, generate once and exit.")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95, dest="top_p")
    parser.add_argument("--max-new-tokens", type=int, default=256, dest="max_new_tokens")
    args = parser.parse_args()

    dec = GPTStreamDecoder(
        args.checkpoint,
        tokenizer_meta_path=args.tokenizer_meta,
        device=args.device,
        seed=args.seed,
    )
    params = GenerationParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )

    if args.prompt is not None:
        dec.stream_print(args.prompt, params)
        return

    dec.run_repl(params)


if __name__ == "__main__":
    main()
