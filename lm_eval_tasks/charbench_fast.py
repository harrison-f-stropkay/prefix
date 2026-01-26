from __future__ import annotations

import datasets

from .charbench_subset import process_docs as _process_docs


def process_docs(
    docs: datasets.Dataset,
    *,
    per_task: int = 100,
    seed: int = 0,
) -> datasets.Dataset:
    return _process_docs(docs, per_task=per_task, seed=seed)
