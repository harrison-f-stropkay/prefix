import argparse
from pathlib import Path

import yaml
from datasets import load_dataset


class Args(argparse.Namespace):
    data_config: Path


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True, type=Path)
    return parser.parse_args(namespace=Args())


def load_data_config(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def download_dataset(config: dict) -> None:
    dataset_config = config["dataset"]
    load_dataset(
        dataset_config["hf_id"],
        name=dataset_config["name"],
        split=dataset_config["split"],
        cache_dir="./data/",
    )


if __name__ == "__main__":
    args = parse_args()
    download_dataset(load_data_config(args.data_config))
