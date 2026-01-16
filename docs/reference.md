# Reference

## Model + Data

Canonical run config: `configs/ce_seed_0.yaml`.

Dataset pipeline:

- FineWeb-Edu
- pack sequences to length 2048
- persist as MDS; load via MosaicML Streaming

Download + MDS creation:

- `uv run python scripts/download_fineweb_edu.py --run-config configs/ce_seed_0.yaml`
- `uv run python scripts/create_mds.py --run-config configs/ce_seed_0.yaml`

## Hardware

- Run:ai
- single node, 8 GPUs per run

## Logging

- human-readable logs via Python `logging`
- eval outputs saved as JSON under `runs/<run_name>/eval/lm_eval_final.json`

## Objective Decisions

- Prefix objectives assign probability mass only to prefix tokens (non-prefix = zero mass).
- Length is the byte-level token string length from `convert_ids_to_tokens`.
- Prefix objectives use epsilon-weighted mixing with the gold token (`1 - epsilon` on gold).
- Special tokens or empty decoded strings use a one-hot target on themselves.
- Tokenizer is `AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)`.
- Pre-compute prefix weights per token for fast lookup during training.
