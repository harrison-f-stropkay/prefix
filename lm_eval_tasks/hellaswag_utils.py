from __future__ import annotations

import os
import sys

import datasets

sys.path.append(os.path.dirname(__file__))
from sample_utils import shuffle_docs  # noqa: E402


def process_docs(docs: datasets.Dataset) -> datasets.Dataset:
    docs = shuffle_docs(docs)

    def _process(doc: dict) -> dict:
        return {
            "ctx": doc["ctx"],
            "endings": doc["endings"],
            "gold": int(doc["label"]),
        }

    return docs.map(_process)
