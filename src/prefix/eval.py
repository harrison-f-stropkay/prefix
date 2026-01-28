"""Evaluation helpers using lm-eval-harness."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, cast

import torch
from lm_eval import evaluator
from lm_eval import tasks as lm_tasks
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from transformers import PreTrainedModel

from prefix.logging_utils import log_eval_metrics, log_eval_summary


def _resolve_task_manager() -> TaskManager:
    tasks_dir = Path(__file__).resolve().parents[2] / "lm_eval_tasks"
    if not tasks_dir.exists():
        raise FileNotFoundError(f"lm_eval_tasks not found: {tasks_dir}")
    return TaskManager(include_path=[str(tasks_dir)])


def _safe_get_task_dict(
    task_names: list[str] | str,
    task_manager: lm_tasks.TaskManager,
) -> dict[str, Any]:
    if isinstance(task_names, str):
        task_names = [task_names]
    if not all(isinstance(name, str) for name in task_names):
        raise TypeError("Expected task names to be a list of strings.")
    task_dict = task_manager.load_task_or_group(task_names)
    lm_tasks.eval_logger.info("Selected tasks:")
    for name, task in task_dict.items():
        if isinstance(name, str) and isinstance(task, lm_tasks.ConfigurableTask):
            yaml_path = Path(task_manager.task_index[name]["yaml_path"])
            try:
                rel = yaml_path.relative_to(Path(lm_tasks.__file__).parent)
            except ValueError:
                rel = yaml_path
            lm_tasks.eval_logger.info("Task: %s (%s)", name, rel)
        else:
            lm_tasks.eval_logger.info("%s: %s", name, task)
    return task_dict


def evaluate_lm_harness(
    model: torch.nn.Module,
    tokenizer: Any,
    tasks: list[str],
    *,
    batch_size: int = 1,
    device: torch.device | None = None,
    limit: int | None = None,
    num_fewshot: int | None = None,
    log_samples: bool = False,
) -> dict[str, Any]:
    if not tasks:
        raise ValueError("No lm-eval tasks provided.")
    device_str = str(device) if device is not None else None
    lm = HFLM(
        pretrained=cast(PreTrainedModel, model),
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device_str,
    )
    task_manager = _resolve_task_manager()
    tasks_dir = Path(__file__).resolve().parents[2] / "lm_eval_tasks"
    if str(tasks_dir) not in sys.path:
        sys.path.append(str(tasks_dir))
    original_get_task_dict = evaluator.get_task_dict
    evaluator.get_task_dict = lambda task_names, _tm=None: _safe_get_task_dict(  # type: ignore[assignment]
        task_names,
        task_manager,
    )
    try:
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
            batch_size=batch_size,
            device=device_str,
            limit=limit,
            num_fewshot=num_fewshot,
            bootstrap_iters=0,
            log_samples=log_samples,
            task_manager=task_manager,
        )
    finally:
        evaluator.get_task_dict = original_get_task_dict
    if results is None:
        raise RuntimeError("lm-eval returned no results.")
    return results


def run_eval_and_log(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    tasks: list[str],
    metrics_path: Path,
    step: int,
    tokens_seen: int,
    eval_name: str,
    label: str,
    batch_size: int = 1,
    device: torch.device | None = None,
    limit: int | None = None,
    num_fewshot: int | None = None,
    log_samples: bool = False,
) -> dict[str, Any]:
    results = evaluate_lm_harness(
        model,
        tokenizer,
        tasks,
        batch_size=batch_size,
        device=device,
        limit=limit,
        num_fewshot=num_fewshot,
        log_samples=log_samples,
    )
    log_eval_summary(results=results, label=label)
    log_eval_metrics(
        metrics_path,
        step=step,
        tokens_seen=tokens_seen,
        eval_name=eval_name,
        results=results,
    )
    return results
