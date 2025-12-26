# Experiments

Run matrix: 1× CE baseline + 2 prefix-aware families × 6 temperatures = 13 runs.

## How Runs Are Represented

- Objective-level knobs live in `configs/objective/`.
- Each cluster run is kicked off via a top-level TOML in `configs/runs/` (see `configs/README.md`).
  - The run config defines the `run.name` and points at the component configs (objective required; others optional).

## Run Naming

Keep run names stable so a restarted job can resume into the same directory:

- `ce`
- `prefix_unnormalized_tau_0p1`
- `prefix_normalized_tau_2p0`

Outputs are written to the PVC under:

- `/home/apluser/runs/<run_name>/`

## Notes for Implementation

- The submission script (`cluster/runai/submit_train.sh`) is intentionally thin; it will re-run the same `run_name` into the same output directory.
- To support preemptions, training should be resume-by-default (see `docs/checkpointing.md`).

Run specs live under `configs/runs/`.
