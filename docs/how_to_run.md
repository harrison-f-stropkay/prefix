# How To Run

Local workflow + how we submit runs on Run:ai.

## Python + Environment

We target Python 3.12 (see `.python-version` and `pyproject.toml`).

This repo uses `uv` to manage a local virtualenv in `.venv/`.

## Setup

- Create/update the environment (dev tools included): `uv sync --extra dev`
- Run commands without activating the venv: `uv run <command>`

Examples:

- Run tests: `uv run pytest`
- Lint: `uv run ruff check .`

## Run Configs

Cluster submission takes a single run-config TOML from `configs/runs/` (and component YAML configs under `configs/`).

Run config format: `configs/README.md`.

## Run:ai

Runs execute from the latest `main` and write outputs to the PVC. The exact commit is recorded in the run metadata.

Prereqs: `runai` CLI installed and authenticated.

Spin up a dev workspace (Jupyter, interactive debugging):

- `cluster/runai/spinup_workspace.sh`

Submit a training run:

- `cluster/runai/submit_train.sh configs/runs/ce_seed0.toml`

### Outputs + Resume

- Outputs land under `/home/apluser/runs/<run-name>/` on the mounted PVC.
- Resume-by-default contract: `docs/checkpointing.md`.

### Submitting the 13 Runs

Run specs are checked in under `configs/runs/`. Submit with a loop:

- `for f in configs/runs/*.toml; do cluster/runai/submit_train.sh "$f"; done`
