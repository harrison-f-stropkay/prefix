## Resuming Runs (Preemption-Friendly)

Since we're using Run:ai, our runs can get preempted, so we should checkpoint fairly often.

We save a full "training state" dict, including:

- `model.state_dict()`
- `optimizer.state_dict()`
- `scheduler.state_dict()`
- `global_step`
- `tokens_seen`
- RNG states (PyTorch, CUDA, NumPy, Python)
- MosaicML `StreamingDataset`/`StreamingDataLoader` state dict

### Multi-GPU Checkpoint Files

To keep per-rank dataloader/RNG state without duplicating large model weights:

- `checkpoints/latest.pt`: global state (model/optimizer/scheduler/step/tokens_seen + dataloader state) written by rank 0.
- `checkpoints/latest_rank{rank}.pt`: per-rank RNG state written by every rank.

On resume, every rank loads `latest.pt`, then its own `latest_rank{rank}.pt`.

## Intended Preemption/Resume Workflow

Goal: a run can be preempted and later restarted, and training continues from the latest checkpoint.

This requires two things:

1. **Persistent storage**: checkpoints and run state must be written to the mounted PVC (not ephemeral container storage).
2. **Resume-by-default training code**: when `prefix.train` starts, it should detect an existing checkpoint in the run directory and resume automatically.

### Run Directory Contract

For a given `run_name`, the run output directory is stable:

- `/home/apluser/runs/<run_name>/`

Recommended substructure:

- `/home/apluser/runs/<run_name>/checkpoints/` (latest + at most one periodic)
- `/home/apluser/runs/<run_name>/logs/` (stdout/stderr, training logs)
- `/home/apluser/runs/<run_name>/meta/` (git ref/commit, resolved configs, environment snapshot)

### Resume-by-Default Behavior (When Implementing)

On startup:

- If a checkpoint exists under `.../checkpoints/`, restore and continue.
- Otherwise, initialize from scratch and start writing checkpoints.
- Save checkpoints frequently enough to make preemptions tolerable.

### Cluster Behavior Note

Run:ai preemption can terminate a pod; whether the workload later resumes automatically depends on cluster policy. The training code should be restart-safe regardless.

## What the Submission Script Does (and Doesn’t)

`runai/submit_train.sh`:

- Submits a preemptible workload and writes outputs to the PVC under `/home/apluser/runs/<run_name>/`.
- Does not implement resume logic itself; it just starts `python -m prefix.train ...` again.

Practical implication:

- If Run:ai restarts the workload automatically, `prefix.train` must resume-by-default.
- If Run:ai does not restart automatically, re-running the same submit command should be sufficient as long as the run name (and therefore the run directory) is the same.
