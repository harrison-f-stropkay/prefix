# Experiments

Run matrix: CE baseline + label smoothing + prefix-simple + 2 prefix-aware families × 6 temperatures = 15 runs.

## How Runs Are Represented

- Each cluster run is kicked off via a self-contained YAML in `configs/` (see `configs/README.md`).
  - The run config defines the `run.name` and inlines the data/model/train/eval/objective sections.

## Run Naming

Run configs include a type + seed. We name runs as:

- `ce_seed_0`
- `label_smoothing_seed_0`
- `prefix_simple_seed_0`
- `prefix_unnormalized_tau_0p1_seed_0`
- `prefix_normalized_tau_2p0_seed_0`

Outputs are written to the PVC under:

- `/home/apluser/runs/<run_name>/`

## Notes for Implementation

- The submission script (`runai/submit_train.sh`) is intentionally thin; it will re-run the same `run_name` into the same output directory.
- To support preemptions, training should be resume-by-default (see `docs/checkpointing.md`).

Run configs live under `configs/`.
