.PHONY: tokenizer-train tokenizer-experiments tokenizer-experiments-quick

tokenizer-train:
	uv run python tests/train_tokenizer_experiments.py

tokenizer-experiments:
	uv run python tests/train_tokenizer_experiments.py
	uv run python tests/tokenizer_experiments.py

tokenizer-experiments-quick:
	uv run python tests/train_tokenizer_experiments.py
	uv run python tests/tokenizer_experiments.py --skip-serialize
