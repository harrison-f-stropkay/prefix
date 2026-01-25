"""Evaluation helpers using lm-eval-harness."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from transformers import PreTrainedModel


def extract_eval_sample_counts(results: dict[str, Any]) -> dict[str, Any]:
    per_task: dict[str, int] = {}
    total = 0
    for task, metrics in (results.get("results") or {}).items():
        if not isinstance(metrics, dict):
            continue
        raw = (
            metrics.get("n_samples")
            or metrics.get("num_samples")
            or metrics.get("samples")
        )
        if raw is None:
            continue
        try:
            count = int(raw)
        except (TypeError, ValueError):
            continue
        per_task[task] = count
        total += count
    payload: dict[str, Any] = {}
    if total:
        payload["total"] = total
    if per_task:
        payload["per_task"] = per_task
    return payload


def evaluate_lm_harness(
    model: torch.nn.Module,
    tokenizer: Any,
    tasks: list[str],
    *,
    batch_size: int = 1,
    device: torch.device | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    device_str = str(device) if device is not None else None
    lm = HFLM(
        pretrained=cast(PreTrainedModel, model),
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device_str,
    )
    include_paths: list[str] = []
    env_tasks_dir = os.environ.get("LM_EVAL_TASKS_DIR")
    if env_tasks_dir:
        include_paths.append(env_tasks_dir)
    default_tasks_dir = Path(__file__).resolve().parents[2] / "lm_eval_tasks"
    if default_tasks_dir.exists():
        include_paths.append(str(default_tasks_dir))
    task_manager = TaskManager(include_path=include_paths) if include_paths else None
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        batch_size=batch_size,
        device=device_str,
        limit=limit,
        bootstrap_iters=0,
        log_samples=False,
        task_manager=task_manager,
    )
    return results or {}
