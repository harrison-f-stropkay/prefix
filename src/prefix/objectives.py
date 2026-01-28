"""Objective utilities for prefix-aware training."""

import math
from typing import Any

import marisa_trie
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def load_tokenizer(tokenizer_hf_id: str) -> PreTrainedTokenizerFast:
    return AutoTokenizer.from_pretrained(tokenizer_hf_id, use_fast=True)


def build_lookup(tokenizer: Any) -> list[tuple[list[int], list[int]]]:
    vocab_size = len(tokenizer)
    token_strings = [tokenizer.convert_ids_to_tokens(token_id) for token_id in range(vocab_size)]
    # Lengths are in byte-level token space (token string length), matching model units.
    token_lengths = [len(token_str) for token_str in token_strings]
    special_ids = set(tokenizer.all_special_ids)
    token_ids_by_string: dict[str, list[int]] = {}
    for token_id, token_str in enumerate(token_strings):
        if token_id in special_ids or token_str == "":
            continue
        token_ids = token_ids_by_string.get(token_str)
        if token_ids is None:
            token_ids = []
            token_ids_by_string[token_str] = token_ids
        token_ids.append(token_id)
    trie = marisa_trie.Trie(token_ids_by_string.keys())

    prefix_lookup: list[tuple[list[int], list[int]]] = [([], []) for _ in range(vocab_size)]
    for token_id, token_str in enumerate(token_strings):
        if token_id in special_ids or token_str == "":
            prefix_lookup[token_id] = ([token_id], [token_lengths[token_id]])
            continue
        prefix_ids: list[int] = []
        prefix_lengths: list[int] = []
        for prefix_str in trie.prefixes(token_str):
            for prefix_id in token_ids_by_string[prefix_str]:
                prefix_ids.append(prefix_id)
                prefix_lengths.append(token_lengths[prefix_id])
        prefix_lookup[token_id] = (prefix_ids, prefix_lengths)
    return prefix_lookup


def build_target_distribution(
    lookup: list[tuple[list[int], list[int]]],
    token_id: int,
    objective_type: str,
    *,
    proper_prefixes_only: bool,
    epsilon: float = 0.1,
    tau: float = 1.0,
    normalized: bool = False,
) -> list[float]:
    vocab_size = len(lookup)
    dist = [0.0] * vocab_size

    if objective_type == "cross_entropy":
        dist[token_id] = 1.0
        return dist
    if objective_type == "label_smoothing":
        if vocab_size <= 1:
            dist[token_id] = 1.0
            return dist
        smooth = epsilon / (vocab_size - 1)
        for i in range(vocab_size):
            dist[i] = smooth
        dist[token_id] = 1.0 - epsilon
        return dist

    if objective_type not in {
        "prefix_simple",
        "prefix_softmax",
        "prefix_softmax_normalized",
    }:
        raise ValueError(f"Unknown objective type: {objective_type!r}")

    prefix_ids, prefix_lengths = lookup[token_id]
    if proper_prefixes_only:
        prefix_pairs = [
            (pid, plen)
            for pid, plen in zip(prefix_ids, prefix_lengths, strict=True)
            if pid != token_id
        ]
    else:
        prefix_pairs = list(zip(prefix_ids, prefix_lengths, strict=True))
    if not prefix_pairs:
        dist[token_id] = 1.0
        return dist

    dist[token_id] = 1.0 - epsilon
    if objective_type == "prefix_simple":
        share = epsilon / len(prefix_pairs)
        for pid, _ in prefix_pairs:
            dist[pid] += share
        return dist

    if normalized or objective_type == "prefix_softmax_normalized":
        token_len = max(prefix_lengths)
        denom = token_len if token_len > 0 else 1
        weights = [plen / denom for _, plen in prefix_pairs]
    else:
        weights = [plen for _, plen in prefix_pairs]

    scaled = [w / tau for w in weights]
    max_scaled = max(scaled)
    exp_sum = sum(math.exp(s - max_scaled) for s in scaled)
    for (pid, _), s in zip(prefix_pairs, scaled, strict=True):
        dist[pid] += epsilon * (math.exp(s - max_scaled) / exp_sum)
    return dist
