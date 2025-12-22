import argparse
import logging
from pathlib import Path

import yaml
from streaming import Reader
from transformers import AutoTokenizer


class Args(argparse.Namespace):
    data_config: Path


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True, type=Path)
    return parser.parse_args(namespace=Args())


def load_data_config(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def spot_check_mds(config: dict, num_batches: int = 10) -> None:
    output_dir = Path(config["dir"])
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["hf_id"], use_fast=True)
    reader = Reader(str(output_dir))
    it = iter(reader)

    for batch_idx in range(num_batches):
        try:
            sample = next(it)
        except StopIteration:
            logging.warning("MDS ended after %d batches", batch_idx)
            break
        input_ids = sample["input_ids"]
        ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        text = tokenizer.decode(ids, skip_special_tokens=False)
        print(f"--- batch {batch_idx} ---")
        print(text)
        print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    spot_check_mds(load_data_config(args.data_config))
