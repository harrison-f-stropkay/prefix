## Quick Orientation

- Repo overview: prefix-aware label smoothing experiments for Llama-3–style LM training (see `README.md`).
- Reusable logic lives in `src/prefix/`; keep `scripts/` as thin entrypoints and sanity checks.
- Artifacts: keep outputs in `runs/` and datasets in `data/` (both are gitignored).

## Dev Environment

- If it's after 11pm, if we're connectd to the corporate VPN, we won't have access to the internet; if we're not connected to the corporate VPN, then we won't have access to the GPU nodes.
- Env/tooling: use `uv run` for scripts (local venv in `.venv/`).
- Check your work with `uvx ty check` and `uvx ruff check` if available.
- Use `runai workspace bash prefix-cpu-only` to bash into the

## Coding Style

- Prefer high-signal comments for non-obvious logic; keep them brief and purposeful.
- "Just get it done": write minimal, direct code that meets the goal; avoid overly defensive patterns unless clearly warranted.
- When proposing workflow changes, keep scope tight to the request; avoid adding extra features or knobs (e.g., overrides) unless explicitly asked. Prefer suggesting additional knobs as follow-on work. For starters, just get it done as simply as possible.
