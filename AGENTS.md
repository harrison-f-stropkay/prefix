## Quick Orientation

- Repo overview: prefix-aware label smoothing experiments for Llama-3–style LM training (see `README.md`).
- Reusable logic lives in `src/prefix/`; keep `scripts/` as thin entrypoints.
- Config policy:
  - Component configs are YAML under `configs/{data,model,train,objective,eval}/`.
  - Top-level run specs are TOML under `configs/runs/` (see `configs/README.md`).
- Artifacts: keep outputs in `runs/` and datasets/artifacts in `data/` (both are gitignored). Avoid committing large binaries.

## Dev Environment

- Python: 3.12 (see `.python-version` and `pyproject.toml`).
- Env/tooling: use `uv` (local venv in `.venv/`).

Common commands:

- Install/update env (dev extras): `uv sync --extra dev`
- Run without activating venv: `uv run <command>`
- Lint: `uv run ruff check .`
- Tests: `uv run pytest`

## Coding Style

- Prefer high-signal comments for non-obvious logic; keep them brief and purposeful.
- "Just get it done": write minimal, direct code that meets the goal; avoid overly defensive patterns unless clearly warranted.
- When proposing workflow changes, keep scope tight to the request; avoid adding extra features or knobs (e.g., overrides) unless explicitly asked. Prefer suggesting additional knobs as follow-on work.

## Cluster (Run:ai)

- Cluster helpers live under `cluster/runai/`.
- Don’t run `runai ...` commands unless explicitly asked (they require cluster auth and submit real workloads).
- `cluster/runai/submit_train.sh` re-clones the repo at `main`; run-config paths must exist in the repo checkout.

## Canonical Docs

- Runbook: `docs/how_to_run.md`
- Conventions: `docs/project_conventions.md`
- Experiments/run naming: `docs/experiments.md`
- Resume/checkpoint contract: `docs/checkpointing.md`
- Reference configs + data notes: `docs/reference.md`
