import argparse
import logging
from pathlib import Path
from tqdm import tqdm

import yaml
from streaming import StreamingDataLoader, StreamingDataset
from transformers import AutoTokenizer


class Args(argparse.Namespace):
    data_config: Path


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True, type=Path)
    return parser.parse_args(namespace=Args())


def load_data_config(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def spot_check_mds(
    config: dict,
    num_batches: int = 10,
    batch_size: int = 8,
    n_workers: int = 4,
) -> None:
    output_dir = Path(config["dir"])
    if not output_dir.exists():
        raise FileNotFoundError(f"MDS directory not found: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["hf_id"], use_fast=True)

    dataset = StreamingDataset(
        local=str(output_dir),
        shuffle=True,
        batch_size=1,
    )

    print(len(dataset))

    # dataloader = StreamingDataLoader(
    #     dataset=dataset,
    #     batch_size=batch_size,
    #     num_workers=n_workers,
    # )

    # print("state_dict: ", dataloader.state_dict())

    # it = iter(dataloader)
    # for batch_idx in range(num_batches):
    #     try:
    #         sample = next(it)
    #     except StopIteration:
    #         logging.warning("MDS ended after %d batches", batch_idx)
    #         break

    #     ids = sample["input_ids"][sample["input_ids"].shape[0] - 1]
    #     text = tokenizer.decode(ids, skip_special_tokens=False)
    #     print(f"--- last example in the batch {batch_idx} ---")
    #     print(text)
    #     print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    spot_check_mds(load_data_config(args.data_config))
