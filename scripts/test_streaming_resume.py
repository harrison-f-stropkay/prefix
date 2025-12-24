import argparse
import logging
from pathlib import Path

import torch
import yaml
from streaming import StreamingDataLoader, StreamingDataset

logger = logging.getLogger(__name__)

class Args(argparse.Namespace):
    data_config: Path
    batch_size: int
    num_workers: int
    state_path: Path


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--state-path", type=Path, default=Path("/tmp/streaming_dl_state.pt"))
    return parser.parse_args(namespace=Args())


def load_data_config(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def batch_fingerprint(batch: dict) -> list[int]:
    return batch["input_ids"][0][:8].tolist()


def build_dataloader(output_dir: Path, batch_size: int, num_workers: int) -> StreamingDataLoader:
    dataset = StreamingDataset(
        local=str(output_dir),
        shuffle=True,
        batch_size=batch_size,
    )
    return StreamingDataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
    )


def shutdown_dataloader(dataloader: StreamingDataLoader) -> None:
    shutdown = getattr(dataloader, "shutdown", None)
    if callable(shutdown):
        shutdown()
    dataset = getattr(dataloader, "dataset", None)
    if dataset is not None:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()


def main() -> None:
    args = parse_args()
    config = load_data_config(args.data_config)
    output_dir = Path(config["dir"])
    if not output_dir.exists():
        raise FileNotFoundError(f"MDS directory not found: {output_dir}")

    dataloader = build_dataloader(output_dir, args.batch_size, args.num_workers)
    try:
        data_iter = iter(dataloader)

        next(data_iter)
        next(data_iter)
        torch.save(dataloader.state_dict(), args.state_path)

        b3 = next(data_iter)
        b4 = next(data_iter)

        original_3 = batch_fingerprint(b3)
        original_4 = batch_fingerprint(b4)
    finally:
        shutdown_dataloader(dataloader)

    dataloader = build_dataloader(output_dir, args.batch_size, args.num_workers)
    try:
        logger.info("Loading dataset state...")
        dataloader.load_state_dict(torch.load(args.state_path))
        data_iter = iter(dataloader)

        resumed_3 = batch_fingerprint(next(data_iter))
        resumed_4 = batch_fingerprint(next(data_iter))

        assert resumed_3 == original_3, f"batch3 mismatch: orig={original_3} resumed={resumed_3}"
        assert resumed_4 == original_4, f"batch4 mismatch: orig={original_4} resumed={resumed_4}"
        logger.info("RESUME_OK: True")
    finally:
        shutdown_dataloader(dataloader)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    main()
