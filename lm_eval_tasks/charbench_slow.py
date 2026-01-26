from __future__ import annotations

import os
import sys

import datasets

sys.path.append(os.path.dirname(__file__))
from charbench_subset import process_docs as _process_docs  # noqa: E402


def process_docs(
    docs: datasets.Dataset,
    *,
    per_task: int = 1000,
    seed: int = 0,
) -> datasets.Dataset:
    return _process_docs(docs, per_task=per_task, seed=seed)
