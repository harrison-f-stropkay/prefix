# How To Run

Local workflow + how we submit runs on Run:ai.

## Python + Environment

We target Python 3.12 (see `.python-version` and `pyproject.toml`).

This repo uses `uv` to manage a local virtualenv in `.venv/`.

## Setup

- Run commands: `uv run <command>`

Examples:

- Run tests: `uv run pytest`
- Lint: `uv run ruff check .`

## Run Configs

Cluster submission takes a single run-config YAML from `configs/` (self-contained).

Run config format: `configs/README.md`.

## Data Sanity Checks

Inspect a few batches from an MDS shard:

- `uv run python scripts/spot_check_mds.py --run-config configs/ce_seed_0.yaml --num-batches 2 --decode`

## Dry-Run (Run:ai)

Submit a dry-run via Run:ai to validate resume behavior on real data:

- `runai/submit_train.sh --dry-run configs/ce_seed_0.yaml`

## Evaluation (lm-eval-harness)

Run eval on a checkpoint:

- `uv run python scripts/eval.py --run-config configs/ce_seed_0.yaml --checkpoint runs/ce_seed_0/checkpoints/latest.pt`

## Run:ai

Runs execute from the latest `main` and write outputs to the PVC. The exact commit is recorded in the run metadata.

Prereqs: `runai` CLI installed and authenticated.

Spin up a dev workspace (Jupyter, interactive debugging):

- `runai/spinup_workspace.sh`

Submit a training run:

- `runai/submit_train.sh configs/ce_seed_0.yaml`

### Outputs + Resume

- Outputs land under `/home/apluser/runs/<run-name>/` on the mounted PVC.
- Resume-by-default contract: `docs/checkpointing.md`.

### Evaluation

Run an eval pass against a checkpoint:

- `uv run python -m prefix.eval --run-config configs/ce_seed_0.yaml --checkpoint runs/ce_seed_0/checkpoints/latest.pt`

### Submitting the 13 Runs

Run configs are checked in under `configs/`. Submit with a loop:

- `for f in configs/*.yaml; do runai/submit_train.sh "$f"; done`
