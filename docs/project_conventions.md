# Project Conventions (Reproducibility + Organization)

Small research repo; optimize for reproducible experiments and low-friction iteration.

Repo layout: see `README.md`.

## Principles

- Put reusable logic in `src/prefix/`; keep `scripts/` thin wrappers.
- Keep experiment knobs in `configs/`; keep outputs in `runs/`/`data/` (gitignored).
- Keep cluster concerns in `cluster/runai/`; training code should not know Run:ai exists.

## Config Formats

See `configs/README.md` for schema details.

Format policy:

- Component configs: YAML under `configs/{data,model,train,objective,eval}/`.
- Run specs: TOML under `configs/runs/`.

## Reproducibility Expectations for Runs

For each run under `runs/<run_name>/`, save:

- resolved configs + run spec
- git commit hash
- environment snapshot (Python + key package versions)
- metrics/evals
- checkpoints

Use stable run names so resubmits resume into the same directory (see `docs/experiments.md`).

## Scripts, Notebooks, and Figures

- Prefer scripts over notebooks for anything you want reproducible.
- Commit final plots as PDFs under `figures/` (optional).

Reference configs and small notes: `docs/reference.md`.

## Testing Workflow (Simple but Effective)

- Use `pytest` in `tests/`. Keep tests fast/deterministic (CPU-only, tiny inputs).
