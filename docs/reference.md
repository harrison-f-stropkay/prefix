# Reference

## Model

Canonical config: `configs/model/llama3_500m.yaml`.

## Data

Canonical config: `configs/data/fineweb_edu_ascii_pack2048.yaml`.

Dataset pipeline:

- FineWeb-Edu
- ASCII-only text
- pack sequences to length 2048
- persist as MDS; load via MosaicML Streaming

Download script (writes sharded JSONL under `data/raw/`):

- `python scripts/download_fineweb_edu.py --config configs/data/fineweb_edu_ascii_pack2048.yaml --out-dir data/raw/fineweb-edu --max-examples 10000`

## Hardware

- Run:ai
- single node, 8 GPUs per run

## Logging

- human-readable logs via Python `logging`
- metrics appended to a CSV (one row per loss/eval)

## Objective Decisions

- Prefix-softmax targets assign probability mass only to prefix tokens (non-prefix = zero mass).
- Length is `len(decoded_ascii_string)`; non-alphanumeric like `\n` counts as 1 character.
- Targets are pure prefix-softmax distributions (no label-smoothing epsilon).
- Special tokens or empty decoded strings use a one-hot target on themselves.
- Tokenizer is `AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)`.
- Pre-compute the target vector for each token in the vocabulary, given an objective YAML; store those values sparsely and with quick GPU access in mind.
