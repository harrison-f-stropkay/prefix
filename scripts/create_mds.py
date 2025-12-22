import argparse
import logging
from pathlib import Path
from shutil import rmtree

import numpy as np
import yaml
from datasets import load_dataset
from streaming import MDSWriter
from transformers import AutoTokenizer


class Args(argparse.Namespace):
    data_config: Path


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True, type=Path)
    return parser.parse_args(namespace=Args())


def load_data_config(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def ascii_only_batch(texts: list[str]) -> list[str]:
    return [text.encode("ascii", "ignore").decode("ascii") for text in texts]


def pack_token_ids(examples, sequence_length: int):
    buffer: list[int] = []
    start = 0
    for example in examples:
        buffer.extend(example["input_ids"])
        while len(buffer) - start >= sequence_length:
            end = start + sequence_length
            yield buffer[start:end]
            start = end
        if start >= 10_000:
            buffer = buffer[start:]
            start = 0
    # Note: any remaining tail shorter than sequence_length is dropped.


def create_mds(config: dict) -> None:
    dataset_config = config["dataset"]
    tokenizer_config = config["tokenizer"]
    text_config = config["text"]
    packing_config = config["packing"]

    seq_len = int(packing_config["sequence_length"])

    dataset = load_dataset(
        dataset_config["hf_id"],
        name=dataset_config["name"],
        split=dataset_config["split"],
        cache_dir="./data/",
    )
    if "text" not in dataset.column_names:
        raise ValueError("Dataset must include a 'text' column.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config["hf_id"], use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must have an eos_token_id.")
    eos_id = tokenizer.eos_token_id

    ascii_only = bool(text_config["ascii_only"])
    num_proc = int(text_config["num_proc"])
    batch_size = int(text_config["batch_size"])

    def tokenize_batch(batch: dict) -> dict:
        texts = batch["text"]
        if ascii_only:
            texts = ascii_only_batch(texts)
        tokenized = tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        tokenized["input_ids"] = [ids + [eos_id] for ids in tokenized["input_ids"]]
        return {"input_ids": tokenized["input_ids"]}

    remove_columns = [col for col in dataset.column_names if col != "text"]
    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=remove_columns,
    )

    output_dir = Path(config["dir"])
    if output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Writing MDS to %s (sequence_length=%d, num_proc=%d, batch_size=%d)",
        output_dir,
        seq_len,
        num_proc,
        batch_size,
    )

    written = 0
    with MDSWriter(
        out=str(output_dir), columns={"input_ids": f"ndarray:int32:{seq_len}"}
    ) as writer:
        for packed in pack_token_ids(tokenized, seq_len):
            assert len(packed) == seq_len, f"Packed len {len(packed)} != seq_len {seq_len}"
            writer.write({"input_ids": np.asarray(packed).astype(np.int32)})
            written += 1
            if written % 1000 == 0:
                logging.info("Wrote %d packed sequences", written)

    logging.info("Finished writing %d packed sequences", written)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    create_mds(load_data_config(args.data_config))
