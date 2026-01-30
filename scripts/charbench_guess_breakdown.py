from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import numpy as np
import seaborn as sns
import torch
from lm_eval.api.instance import Instance
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from matplotlib import pyplot as plt
from transformers import PreTrainedModel

from prefix.config import load_run_config
from prefix.train import build_model_and_tokenizer, load_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_NAME = "charbench"
LIMIT = 1000

RUNS_DIR = REPO_ROOT / "runs"


def _load_charbench_task():
    task_dir = REPO_ROOT / "lm_eval_tasks"
    task_manager = TaskManager(include_path=[str(task_dir)])
    tasks = task_manager.load_task_or_group([TASK_NAME])
    task = tasks[TASK_NAME]
    task.set_fewshot_seed(0)
    return task


def _iter_docs(docs, limit: int):
    if hasattr(docs, "__len__"):
        return [docs[i] for i in range(min(limit, len(docs)))]
    out = []
    for doc in docs:
        out.append(doc)
        if len(out) >= limit:
            break
    return out


def _score_choices(
    lm: HFLM, contexts: list[str], choices: list[list[str]], delimiter: str
) -> np.ndarray:
    requests: list[Instance] = []
    for idx, (ctx, choice_list) in enumerate(zip(contexts, choices, strict=True)):
        for choice in choice_list:
            requests.append(
                Instance(
                    "loglikelihood",
                    doc={},
                    arguments=(ctx, f"{delimiter}{choice}"),
                    idx=idx,
                )
            )
    results = lm.loglikelihood(requests, disable_tqdm=False)
    scores = [score for score, _ in results]
    num_choices = len(choices[0]) if choices else 0
    matrix = []
    for i in range(0, len(scores), num_choices):
        chunk = scores[i : i + num_choices]
        matrix.append(chunk)
    return np.array(matrix, dtype=float)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = _load_charbench_task()
    docs = _iter_docs(task.eval_docs, LIMIT)
    num_fewshot = int(task.config.num_fewshot or 0)
    delimiter = task.config.target_delimiter or " "

    contexts = [task.fewshot_context(doc, num_fewshot) for doc in docs]
    choices = [task.doc_to_choice(doc) for doc in docs]
    gold = [int(task.doc_to_target(doc)) for doc in docs]

    num_classes = len(choices[0])
    labels = [str(i) for i in range(num_classes)]

    entries = []
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        checkpoint = run_dir / "checkpoints" / "latest.pt"
        run_config = REPO_ROOT / "configs" / f"{run_dir.name}.yaml"
        if not checkpoint.exists() or not run_config.exists():
            continue
        entries.append(
            {
                "name": run_dir.name,
                "run_config": run_config,
                "checkpoint": checkpoint,
            }
        )
    if not entries:
        raise SystemExit(f"No runs with checkpoints found under {RUNS_DIR}")

    fig_rows = math.ceil(math.sqrt(len(entries)))
    fig_cols = math.ceil(len(entries) / fig_rows)
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(4 * fig_cols, 4 * fig_rows))
    axes = np.array(axes).reshape(-1)

    for ax, entry in zip(axes, entries, strict=False):
        config = load_run_config(entry["run_config"])
        model, tokenizer = build_model_and_tokenizer(config["model"], config["data"], device=device)
        state = load_checkpoint(entry["checkpoint"])
        model.load_state_dict(state["model"])
        model.eval()
        lm = HFLM(
            pretrained=cast(PreTrainedModel, model),
            tokenizer=tokenizer,
            batch_size=8,
            device=str(device),
        )
        score_matrix = _score_choices(lm, contexts, choices, delimiter)
        preds = np.argmax(score_matrix, axis=1)
        confusion = np.zeros((num_classes, num_classes), dtype=int)
        for g, p in zip(gold, preds, strict=True):
            confusion[g, p] += 1
        sns.heatmap(
            confusion,
            ax=ax,
            cmap="viridis",
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
        )
        ax.set_title(entry["name"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("gold")

    for ax in axes[len(entries) :]:
        ax.axis("off")

    fig.tight_layout()
    out_path = REPO_ROOT / "figures" / "charbench_confusion.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)

    fig2, axes2 = plt.subplots(fig_rows, fig_cols, figsize=(4 * fig_cols, 4 * fig_rows))
    axes2 = np.array(axes2).reshape(-1)
    for ax, entry in zip(axes2, entries, strict=False):
        config = load_run_config(entry["run_config"])
        model, tokenizer = build_model_and_tokenizer(config["model"], config["data"], device=device)
        state = load_checkpoint(entry["checkpoint"])
        model.load_state_dict(state["model"])
        model.eval()
        lm = HFLM(
            pretrained=cast(PreTrainedModel, model),
            tokenizer=tokenizer,
            batch_size=8,
            device=str(device),
        )
        score_matrix = _score_choices(lm, contexts, choices, delimiter)
        probs = np.exp(score_matrix - score_matrix.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        avg_probs = np.zeros((num_classes, num_classes), dtype=float)
        counts = np.zeros(num_classes, dtype=int)
        for g, row in zip(gold, probs, strict=True):
            avg_probs[g] += row
            counts[g] += 1
        for g in range(num_classes):
            if counts[g] > 0:
                avg_probs[g] /= counts[g]
        sns.heatmap(
            avg_probs,
            ax=ax,
            cmap="viridis",
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
        )
        ax.set_title(entry["name"])
        ax.set_xlabel("predicted distribution")
        ax.set_ylabel("gold")

    for ax in axes2[len(entries) :]:
        ax.axis("off")

    fig2.tight_layout()
    out_path2 = REPO_ROOT / "figures" / "charbench_avg_probs.pdf"
    out_path2.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(out_path2, dpi=200)


if __name__ == "__main__":
    main()
