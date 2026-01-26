"""Evaluation helpers using lm-eval-harness."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import torch
from lm_eval import evaluator
from lm_eval import tasks as lm_tasks
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from transformers import PreTrainedModel

from prefix.logging_utils import log_eval_metrics, log_eval_summary


def _safe_get_task_dict(
    task_name_list: str | list[str | dict[str, Any] | lm_tasks.Task],
    task_manager: lm_tasks.TaskManager | None = None,
) -> dict[str, Any]:
    task_name_from_string_dict: dict[Any, Any] = {}
    task_name_from_config_dict: dict[Any, Any] = {}
    task_name_from_object_dict: dict[Any, Any] = {}

    if isinstance(task_name_list, str):
        task_name_list = [task_name_list]
    elif isinstance(task_name_list, list):
        if not all(
            isinstance(task, (str, dict, lm_tasks.Task)) for task in task_name_list
        ):
            raise TypeError(
                "Expected list items of types 'str', 'dict', or 'Task', but at least one entry did not match."
            )
    else:
        raise TypeError(
            f"Expected a 'str' or 'list' but received {type(task_name_list)}."
        )

    string_task_name_list = [task for task in task_name_list if isinstance(task, str)]
    others_task_name_list = [
        task for task in task_name_list if not isinstance(task, str)
    ]
    if task_manager is None and (string_task_name_list or others_task_name_list):
        task_manager = lm_tasks.TaskManager()
    if string_task_name_list:
        task_name_from_string_dict = task_manager.load_task_or_group(
            string_task_name_list
        )

    for task_element in others_task_name_list:
        if isinstance(task_element, dict):
            task_name_from_config_dict = {
                **task_name_from_config_dict,
                **task_manager.load_config(config=task_element),
            }
        elif isinstance(task_element, lm_tasks.Task):
            task_name_from_object_dict = {
                **task_name_from_object_dict,
                lm_tasks.get_task_name_from_object(task_element): task_element,
            }

    if not set(task_name_from_string_dict.keys()).isdisjoint(
        set(task_name_from_object_dict.keys())
    ):
        raise ValueError

    final_task_dict: dict[Any, Any] = {
        **task_name_from_string_dict,
        **task_name_from_config_dict,
        **task_name_from_object_dict,
    }

    lm_tasks._check_duplicates(lm_tasks.get_subtask_list(final_task_dict))

    def pretty_print_task(task_name: Any, manager: lm_tasks.TaskManager, indent: int):
        yaml_path = Path(manager.task_index[task_name]["yaml_path"])
        lm_eval_tasks_path = Path(lm_tasks.__file__).parent
        try:
            relative_yaml_path = yaml_path.relative_to(lm_eval_tasks_path)
        except ValueError:
            relative_yaml_path = yaml_path

        pad = "  " * indent
        lm_tasks.eval_logger.info(
            f"{pad}Task: {task_name} ({relative_yaml_path})"
        )

    lm_tasks.eval_logger.info("Selected tasks:")
    assert task_manager is not None
    for key, value in final_task_dict.items():
        if isinstance(key, lm_tasks.ConfigurableGroup):
            lm_tasks.eval_logger.info(f"Group: {key.group}")

            if isinstance(value, dict):
                first_key = next(iter(value.keys()))

                if isinstance(first_key, lm_tasks.ConfigurableGroup):
                    for subgroup, task_dict in value.items():
                        lm_tasks.eval_logger.info(f"  Subgroup: {subgroup.group}")
                        for task_name, configurable_task in task_dict.items():
                            if isinstance(configurable_task, lm_tasks.ConfigurableTask):
                                pretty_print_task(task_name, task_manager, indent=2)
                            else:
                                lm_tasks.eval_logger.info(
                                    f"{task_name}: {configurable_task}"
                                )
                else:
                    lm_tasks.eval_logger.info(f"{key}: {value}")
            else:
                lm_tasks.eval_logger.info(f"{key}: {value}")
        elif isinstance(key, str) and isinstance(value, lm_tasks.ConfigurableTask):
            pretty_print_task(key, task_manager, indent=0)
        else:
            lm_tasks.eval_logger.info(f"{key}: {value}")

    return final_task_dict


def evaluate_lm_harness(
    model: torch.nn.Module,
    tokenizer: Any,
    tasks: list[str],
    *,
    batch_size: int = 1,
    device: torch.device | None = None,
    limit: int | None = None,
    log_samples: bool = False,
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
    # Work around lm-eval pretty_print_task assuming task YAMLs live under its package.
    original_get_task_dict: Any = evaluator.get_task_dict
    evaluator.get_task_dict = cast(Any, _safe_get_task_dict)
    try:
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
            batch_size=batch_size,
            device=device_str,
            limit=limit,
            bootstrap_iters=0,
            log_samples=log_samples,
            task_manager=task_manager,
        )
    finally:
        evaluator.get_task_dict = original_get_task_dict
    return results or {}


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
    log_samples: bool = False,
) -> dict[str, Any]:
    results = evaluate_lm_harness(
        model,
        tokenizer,
        tasks,
        batch_size=batch_size,
        device=device,
        limit=limit,
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
