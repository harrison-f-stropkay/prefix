from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
LOG_RANK: str | int = "?"


def configure_logging(log_path: Path | None = None) -> None:
    class RankFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.rank = LOG_RANK
            return True

    # Reset handlers to avoid duplicate logs when configure_logging is called twice.
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = old_factory(*args, **kwargs)
        if not hasattr(record, "rank"):
            record.rank = LOG_RANK
        return record

    logging.setLogRecordFactory(record_factory)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [rank=%(rank)s] %(name)s: %(message)s",
    )
    if not any(isinstance(filt, RankFilter) for filt in root.filters):
        root.addFilter(RankFilter())
    if log_path is not None:
        handler = logging.FileHandler(log_path)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [rank=%(rank)s] %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(handler)


def set_log_rank(rank: int) -> None:
    global LOG_RANK
    LOG_RANK = rank


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return list(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(value)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")


def extract_eval_sample_counts(results: dict[str, Any]) -> dict[str, Any]:
    per_task: dict[str, int] = {}
    total = 0
    for task, metrics in (results.get("results") or {}).items():
        if not isinstance(metrics, dict):
            continue
        raw = metrics.get("n_samples") or metrics.get("num_samples") or metrics.get("samples")
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


def log_train_metrics(
    metrics_path: Path,
    *,
    step: int,
    loss: float,
    tokens_seen: int,
    lr: float,
) -> None:
    append_jsonl(
        metrics_path,
        {
            "type": "train",
            "timestamp": datetime.utcnow().isoformat(),
            "step": step,
            "loss": loss,
            "tokens_seen": tokens_seen,
            "lr": lr,
        },
    )


def log_eval_metrics(
    metrics_path: Path,
    *,
    step: int,
    tokens_seen: int,
    eval_name: str,
    results: dict[str, Any],
) -> None:
    sample_counts = extract_eval_sample_counts(results)
    per_task_counts = sample_counts.get("per_task") or {}
    for task_name, task_metrics in (results.get("results") or {}).items():
        if not isinstance(task_metrics, dict):
            continue
        for metric_name, value in task_metrics.items():
            if metric_name == "alias":
                continue
            if "stderr" in metric_name:
                continue
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            clean_metric = metric_name.split(",", 1)[0]
            append_jsonl(
                metrics_path,
                {
                    "type": "eval",
                    "timestamp": datetime.utcnow().isoformat(),
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "eval_name": eval_name,
                    "task": task_name,
                    "metric": clean_metric,
                    "value": numeric,
                    "num_samples": per_task_counts.get(task_name),
                },
            )
            if task_name.startswith("charbench") and clean_metric == "loglikelihood":
                append_jsonl(
                    metrics_path,
                    {
                        "type": "eval",
                        "timestamp": datetime.utcnow().isoformat(),
                        "step": step,
                        "tokens_seen": tokens_seen,
                        "eval_name": eval_name,
                        "task": task_name,
                        "metric": "ppl",
                        "value": math.exp(-numeric),
                        "num_samples": per_task_counts.get(task_name),
                    },
                )


def log_eval_summary(*, results: dict[str, Any], label: str) -> None:
    LOGGER.info("%s results: %s", label, results.get("results"))
    sample_counts = extract_eval_sample_counts(results)
    LOGGER.info("%s samples: %s", label, sample_counts.get("total"))
