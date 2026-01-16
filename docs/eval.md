# Evaluation

We evaluate checkpoints with `lm-eval-harness` using the minimal wrapper in
`scripts/eval.py`. This uses `lm_eval.models.huggingface.HFLM` with a
pre-initialized `transformers` model (supported by lm-eval's HFLM).

Example:

```bash
uv run python scripts/eval.py \
  --run-config configs/ce_seed_0.yaml \
  --checkpoint runs/ce_seed_0/checkpoints/latest.pt
```

Notes:

- To keep runtime short, use `--limit` (applies per task).
- Results are printed and optionally written with `--output`.
