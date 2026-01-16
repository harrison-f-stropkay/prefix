import argparse
import logging
from pathlib import Path

from prefix.config import load_run_config
from prefix.data import build_streaming_dataloader
from prefix.objectives import load_tokenizer


class Args(argparse.Namespace):
    run_config: Path
    num_batches: int
    decode: bool


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Spot check a local MDS dataset.")
    parser.add_argument("--run-config", required=True, type=Path)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--decode", action="store_true")
    return parser.parse_args(namespace=Args())


def main() -> None:
    args = parse_args()
    config = load_run_config(args.run_config)
    data_cfg = config["data"]
    train_cfg = config.get("train") or {}
    run_cfg = config.get("run") or {}
    seed = int(run_cfg.get("seed", 0))

    output_dir = Path(data_cfg["dir"])
    if not output_dir.exists():
        raise FileNotFoundError(f"MDS directory not found: {output_dir}")

    batch_size = int(train_cfg.get("per_gpu_batch_size", 1))
    loader = build_streaming_dataloader(
        output_dir,
        batch_size,
        shuffle=True,
        streaming_cfg=data_cfg.get("streaming") or {},
        shuffle_seed=seed,
    )

    dataset = loader.dataset
    logging.info("dataset size: %s", len(dataset))
    logging.info("state_dict: %s", loader.state_dict())

    if not args.decode:
        return

    tokenizer = load_tokenizer(data_cfg["tokenizer"]["hf_id"])
    it = iter(loader)
    for batch_idx in range(args.num_batches):
        batch = next(it)
        ids = batch["input_ids"][-1]
        text = tokenizer.decode(ids, skip_special_tokens=False)
        logging.info("batch %d last sample: %s", batch_idx, text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
