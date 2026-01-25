from __future__ import annotations

import random

import datasets


def _round_robin(lists: list[list[int]]) -> list[int]:
    indices: list[int] = []
    max_len = max((len(items) for items in lists), default=0)
    for i in range(max_len):
        for items in lists:
            if i < len(items):
                indices.append(items[i])
    return indices


def process_docs(
    docs: datasets.Dataset,
    *,
    per_task: int = 100,
    seed: int = 0,
) -> datasets.Dataset:
    task_to_indices: dict[str, list[int]] = {}
    for idx, task_name in enumerate(docs["task"]):
        task_to_indices.setdefault(task_name, []).append(idx)

    rng = random.Random(seed)
    selected: list[list[int]] = []
    for task_name, indices in task_to_indices.items():
        rng.shuffle(indices)
        selected.append(indices[:per_task])

    round_robin = _round_robin(selected)
    return docs.select(round_robin)
