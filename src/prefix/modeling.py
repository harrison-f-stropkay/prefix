"""Model construction utilities."""

from __future__ import annotations

from typing import Any

from transformers import LlamaConfig, LlamaForCausalLM


def build_llama_model(model_cfg: dict[str, Any], *, vocab_size: int) -> LlamaForCausalLM:
    arch = model_cfg["architecture"]
    config = LlamaConfig(
        vocab_size=int(vocab_size),
        hidden_size=int(arch["hidden_size"]),
        num_hidden_layers=int(arch["num_hidden_layers"]),
        num_attention_heads=int(arch["num_attention_heads"]),
        num_key_value_heads=int(arch["num_key_value_heads"]),
        intermediate_size=int(arch["intermediate_size"]),
        max_position_embeddings=int(arch["max_position_embeddings"]),
        rope_theta=float(arch["rope_theta"]),
        rms_norm_eps=float(arch["rms_norm_eps"]),
        tie_word_embeddings=bool(arch["tie_word_embeddings"]),
    )
    return LlamaForCausalLM(config)
