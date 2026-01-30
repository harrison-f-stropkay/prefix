from __future__ import annotations

import os
import sys

import datasets

sys.path.append(os.path.dirname(__file__))
from sample_utils import shuffle_docs  # type: ignore[import-not-found]  # noqa: E402


def process_docs(docs: datasets.Dataset) -> datasets.Dataset:
    docs = shuffle_docs(docs)

    def _process(doc: dict) -> dict:
        labels = doc["choices"]["label"]
        texts = doc["choices"]["text"]
        answer_key = doc["answerKey"]
        try:
            gold = labels.index(answer_key)
        except ValueError:
            gold = 0
        return {
            "question": doc["question"],
            "choices": texts,
            "gold": int(gold),
        }

    return docs.map(_process)
