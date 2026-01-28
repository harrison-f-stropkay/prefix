from __future__ import annotations

import datasets


def shuffle_docs(docs: datasets.Dataset) -> datasets.Dataset:
    return docs.shuffle(seed=0)
