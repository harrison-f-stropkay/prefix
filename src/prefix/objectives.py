"""Objective utilities for prefix-aware training."""

from pathlib import Path

import yaml
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def load_tokenizer(tokenizer_hf_id: str) -> PreTrainedTokenizerFast:
    return AutoTokenizer.from_pretrained(tokenizer_hf_id, use_fast=True)


def load_objective_config(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def decode_token(tokenizer: PreTrainedTokenizerFast, token_id: int) -> str:
    return tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def is_special_token(tokenizer: PreTrainedTokenizerFast, token_id: int) -> bool:
    return token_id in tokenizer.all_special_ids


def decoded_token_length(decoded_token: str) -> int:
    return len(decoded_token)


def build_decoded_vocab(tokenizer: PreTrainedTokenizerFast) -> list[str]:
    return [decode_token(tokenizer, token_id) for token_id in range(tokenizer.vocab_size)]


def prefix_token_ids(
    tokenizer: PreTrainedTokenizerFast,
    token_id: int,
    decoded_vocab: list[str] | None = None,
) -> list[int]:
    if decoded_vocab is None:
        decoded_vocab = build_decoded_vocab(tokenizer)
    decoded_target = decoded_vocab[token_id]
    if is_special_token(tokenizer, token_id) or decoded_target == "":
        return [token_id]
    special_ids = set(tokenizer.all_special_ids)
    prefix_ids = []
    for candidate_id, candidate_decoded in enumerate(decoded_vocab):
        if candidate_id in special_ids or candidate_decoded == "":
            continue
        if decoded_target.startswith(candidate_decoded):
            prefix_ids.append(candidate_id)
    return prefix_ids


if __name__ == "__main__":
    tokenizer = load_tokenizer("meta-llama/Meta-Llama-3-8B")
    print(tokenizer.all_special_tokens)
    print(tokenizer.all_special_tokens_extended)
    for t in tokenizer.get_vocab():
        print(t)
