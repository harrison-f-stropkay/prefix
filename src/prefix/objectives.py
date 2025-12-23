"""Objective utilities for prefix-aware training."""

import math
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def load_tokenizer(tokenizer_hf_id: str) -> PreTrainedTokenizerFast:
    return AutoTokenizer.from_pretrained(tokenizer_hf_id, use_fast=True)


def build_lookup(tokenizer: PreTrainedTokenizerFast) -> list[tuple[list[int], list[int]]]:
    decoded_vocab = [
        tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id in range(tokenizer.vocab_size)
    ]
    decoded_lengths = [len(decoded) for decoded in decoded_vocab]
    special_ids = set(tokenizer.all_special_ids)
    vocab_size = tokenizer.vocab_size

    class _TrieNode:
        __slots__ = ("children", "terminal_ids")

        def __init__(self) -> None:
            self.children: dict[str, _TrieNode] = {}
            self.terminal_ids: list[int] = []

    root = _TrieNode()
    for token_id, decoded in enumerate(decoded_vocab):
        if token_id in special_ids or decoded == "":
            continue
        node = root
        for ch in decoded:
            node = node.children.setdefault(ch, _TrieNode())
        node.terminal_ids.append(token_id)

    prefix_lookup: list[tuple[list[int], list[int]]] = [([], []) for _ in range(vocab_size)]
    for token_id, decoded in enumerate(decoded_vocab):
        if token_id in special_ids or decoded == "":
            prefix_lookup[token_id] = ([token_id], [decoded_lengths[token_id]])
            continue
        node = root
        prefix_ids: list[int] = []
        prefix_lengths: list[int] = []
        for ch in decoded:
            node = node.children[ch]
            if node.terminal_ids:
                for prefix_id in node.terminal_ids:
                    prefix_ids.append(prefix_id)
                    prefix_lengths.append(decoded_lengths[prefix_id])
        prefix_lookup[token_id] = (prefix_ids, prefix_lengths)
    return prefix_lookup


def get_logprobs(lookup: list[tuple[list[int], list[int]]], token_id: int) -> dict[int, float]:
    prefix_ids, prefix_lengths = lookup[token_id]
    if not prefix_ids:
        return {}
    max_len = max(prefix_lengths)
    exp_sum = sum(math.exp(length - max_len) for length in prefix_lengths)
    log_denom = max_len + math.log(exp_sum)
    return {
        prefix_id: length - log_denom
        for prefix_id, length in zip(prefix_ids, prefix_lengths, strict=True)
    }
