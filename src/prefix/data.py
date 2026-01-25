"""Data utilities for MDS creation and loading."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from streaming import MDSWriter, StreamingDataLoader, StreamingDataset
from transformers import AutoTokenizer

LOGGER = logging.getLogger(__name__)


def download_dataset(data_cfg: dict[str, Any]) -> None:
    dataset_cfg = data_cfg["dataset"]
    load_dataset(
        dataset_cfg["hf_id"],
        name=dataset_cfg["name"],
        split=dataset_cfg["split"],
        cache_dir="./data/",
    )


def pack_token_ids(examples: Iterable[dict[str, Any]], sequence_length: int) -> Iterable[list[int]]:
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


def create_mds(data_cfg: dict[str, Any]) -> None:
    dataset_cfg = data_cfg["dataset"]
    tokenizer_cfg = data_cfg["tokenizer"]
    text_cfg = data_cfg.get("text_processing")
    if not text_cfg:
        raise ValueError("Missing data.text_processing in run config.")
    packing_cfg = data_cfg["packing"]

    seq_len = int(packing_cfg["sequence_length"])

    dataset = load_dataset(
        dataset_cfg["hf_id"],
        name=dataset_cfg["name"],
        split=dataset_cfg["split"],
        cache_dir="./data/",
    )
    if "text" not in dataset.column_names:
        raise ValueError("Dataset must include a 'text' column.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg["hf_id"], use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must have an eos_token_id.")
    eos_id = tokenizer.eos_token_id

    num_proc = int(text_cfg["num_proc"])
    batch_size = int(text_cfg["batch_size"])

    def tokenize_batch(batch: dict[str, Any]) -> dict[str, Any]:
        tokenized = tokenizer(
            batch["text"],
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

    output_dir = Path(data_cfg["dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output dir {output_dir} exists and is not empty.")
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
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
            writer.write({"input_ids": np.asarray(packed, dtype=np.int32)})
            written += 1
            if written % 10000 == 0:
                LOGGER.info("Wrote %d packed sequences", written)

    LOGGER.info("Finished writing %d packed sequences", written)


def collate_input_ids(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    input_ids = np.asarray([sample["input_ids"] for sample in batch], dtype=np.int64)
    return {"input_ids": torch.from_numpy(input_ids)}


def build_streaming_dataloader(
    data_dir: Path,
    batch_size: int,
    shuffle: bool,
    *,
    streaming_cfg: dict[str, Any],
    shuffle_seed: int,
) -> StreamingDataLoader:
    validate_mds_shards(data_dir)
    dataset = StreamingDataset(
        local=str(data_dir),
        split=None,
        batch_size=batch_size,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        shuffle_algo=streaming_cfg.get("shuffle_algo", "py1e"),
        batching_method=streaming_cfg.get("batching_method", "random"),
        sampling_method=streaming_cfg.get("sampling_method", "balanced"),
        sampling_granularity=streaming_cfg.get("sampling_granularity", 1),
        partition_algo=streaming_cfg.get("partition_algo", "relaxed"),
        num_canonical_nodes=streaming_cfg.get("num_canonical_nodes"),
        predownload=streaming_cfg.get("predownload"),
        cache_limit=streaming_cfg.get("cache_limit"),
        allow_unsafe_types=streaming_cfg.get("allow_unsafe_types", False),
        replication=streaming_cfg.get("replication"),
    )
    return StreamingDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_input_ids,
    )


def validate_mds_shards(data_dir: Path) -> None:
    index_path = data_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"MDS index not found: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = index.get("shards") or []
    if not shards:
        raise ValueError(f"MDS index has no shards: {index_path}")
    for shard in shards:
        raw = shard.get("raw_data") or shard.get("zip_data") or {}
        basename = raw.get("basename")
        expected = raw.get("bytes")
        if not basename or expected is None:
            raise ValueError(f"MDS index missing shard metadata: {index_path}")
        shard_path = data_dir / basename
        if not shard_path.exists():
            raise FileNotFoundError(f"MDS shard not found: {shard_path}")
        actual = shard_path.stat().st_size
        if actual != expected:
            raise RuntimeError(
                f"MDS shard size mismatch: {shard_path} expected {expected} bytes, got {actual} bytes"
            )
