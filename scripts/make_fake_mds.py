"""Create a tiny MDS dataset for smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from streaming import MDSWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", required=True, type=Path)
    parser.add_argument("--num-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.run_config.read_text(encoding="utf-8"))
    data_dir = Path(cfg["data"]["dir"])
    seq_len = int(cfg["data"]["packing"]["sequence_length"])
    per_device_batch = int(cfg.get("train", {}).get("per_gpu_batch_size", 1))
    world_size = int(cfg.get("train", {}).get("distributed", {}).get("world_size", 1))
    # Ensure enough sequences for multiple global batches across all ranks.
    min_samples = max(256, per_device_batch * world_size * 16)
    num_samples = max(args.num_samples or min_samples, min_samples)

    if data_dir.exists() and any(data_dir.iterdir()):
        print(f"[run] data dir already populated: {data_dir}")
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    with MDSWriter(out=str(data_dir), columns={"input_ids": f"ndarray:int32:{seq_len}"}) as writer:
        for i in range(num_samples):
            data = (np.arange(seq_len, dtype=np.int32) + i) % 32000
            writer.write({"input_ids": data})
    print(f"[run] wrote {num_samples} fake sequences to {data_dir}")


if __name__ == "__main__":
    main()
