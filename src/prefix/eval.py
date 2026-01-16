"""Evaluation helpers using lm-eval-harness."""

from __future__ import annotations

from typing import Any

import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM


def run_lm_eval(
    model: torch.nn.Module,
    tokenizer: Any,
    tasks: list[str],
    *,
    batch_size: int = 1,
    device: torch.device | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    device_str = str(device) if device is not None else None
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device=device_str)
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        batch_size=batch_size,
        device=device_str,
        limit=limit,
        bootstrap_iters=0,
        log_samples=False,
    )
    return results or {}
