#!/usr/bin/env python3
"""Print total, embedding, and non-embedding parameter counts for a config."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from prefix.modeling import build_llama_model
from prefix.objectives import load_tokenizer


def count_params(model) -> tuple[int, int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    embed_params = 0
    seen = set()
    for embed in (model.get_input_embeddings(), model.get_output_embeddings()):
        if embed is None:
            continue
        weight = getattr(embed, "weight", None)
        if weight is None or id(weight) in seen:
            continue
        seen.add(id(weight))
        embed_params += weight.numel()
    return total_params, embed_params, total_params - embed_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Count model parameters from a config.")
    parser.add_argument("config", help="Path to a model config YAML file.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    tokenizer = load_tokenizer(cfg["data"]["tokenizer"]["hf_id"])
    model = build_llama_model(cfg["model"], vocab_size=len(tokenizer))
    total_params, embed_params, non_embed_params = count_params(model)

    print(f"total_params={total_params / 10**6}M")
    print(f"embedding_params={embed_params / 10**6}M")
    print(f"non_embedding_params={non_embed_params / 10**6}M")


if __name__ == "__main__":
    main()
