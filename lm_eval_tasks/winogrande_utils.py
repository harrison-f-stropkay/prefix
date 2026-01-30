from __future__ import annotations

import os
import sys

import datasets

sys.path.append(os.path.dirname(__file__))
from sample_utils import shuffle_docs  # type: ignore[import-not-found]  # noqa: E402


def process_docs(docs: datasets.Dataset) -> datasets.Dataset:
    docs = shuffle_docs(docs)

    def _process(doc: dict) -> dict:
        sentence = doc["sentence"]
        option1 = doc["option1"]
        option2 = doc["option2"]
        choices = [
            sentence.replace("_", option1),
            sentence.replace("_", option2),
        ]
        return {
            "query": "",
            "choices": choices,
            "gold": int(doc["answer"]) - 1,
        }

    return docs.map(_process)
