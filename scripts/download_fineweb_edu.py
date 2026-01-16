import argparse
from pathlib import Path

from prefix.config import load_run_config
from prefix.data import download_dataset


class Args(argparse.Namespace):
    run_config: Path


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", required=True, type=Path)
    return parser.parse_args(namespace=Args())


if __name__ == "__main__":
    args = parse_args()
    config = load_run_config(args.run_config)
    download_dataset(config["data"])
