import argparse
import logging
from pathlib import Path

from prefix.config import load_run_config
from prefix.data import create_mds


class Args(argparse.Namespace):
    run_config: Path


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", required=True, type=Path)
    return parser.parse_args(namespace=Args())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = load_run_config(args.run_config)
    create_mds(config["data"])
