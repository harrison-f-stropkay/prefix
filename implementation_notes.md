## Dataset Pipeline (FineWeb-Edu, ASCII-Only)

- Remove all non-ASCII
- Pack sequences of length 2048
- Persist using MosaicML's `MDSWriter`
- Load with MosaicML `Streaming` during training

## Hardware

- Run:ai
- 1 node, 8 GPUs (single-node DDP) per run

## Resuming runs

Since we're using Run:ai, our runs can get preempted, so we should checkpoint fairly often. We save a full "training state" dict, including:

- `model.state_dict()`
- `optimizer.state_dict()`
- `scheduler.state_dict()`
- `global_step`
- `tokens_seen`
- RNG states (PyTorch, CUDA, NumPy, Python)
- MosaicML `StreamingDataset`/`StreamingDataLoader` state dict

## Logging

We use Python’s logging module for human-readable messages. We keep all metrics (loss and evals) in a separate csv file. Each eval or loss number gets i ts own line in the run's csv.
