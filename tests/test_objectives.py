import argparse
from pathlib import Path

import yaml

from prefix.objectives import load_tokenizer


class Args(argparse.Namespace):
    data_config: Path


def load_tokenizer_hf_id(data_config_path: Path) -> str:
    config = yaml.safe_load(data_config_path.read_text(encoding="utf-8"))
    return config["tokenizer"]["hf_id"]


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True, type=Path)
    return parser.parse_args(namespace=Args())


if __name__ == "__main__":
    args = parse_args()
    tokenizer = load_tokenizer(load_tokenizer_hf_id(args.data_config))
