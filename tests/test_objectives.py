import math

import pytest

from prefix.objectives import build_lookup, build_target_distribution


class DummyTokenizer:
    def __init__(self, vocab: list[str], special_ids: list[int] | None = None) -> None:
        self._vocab = vocab
        self.vocab_size = len(vocab)
        self.all_special_ids = special_ids or []

    def __len__(self) -> int:
        return len(self._vocab)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self._vocab[token_id]


def test_build_lookup_prefixes() -> None:
    tokenizer = DummyTokenizer(["a", "ab", "b"])
    lookup = build_lookup(tokenizer)

    assert lookup[0][0] == [0]
    assert lookup[1][0] == [0, 1]
    assert lookup[2][0] == [2]


def test_target_distribution_cross_entropy() -> None:
    lookup = [([0], [1]), ([0, 1], [1, 2])]
    dist = build_target_distribution(lookup, 1, "cross_entropy")
    assert dist == [0.0, 1.0]


def test_target_distribution_label_smoothing() -> None:
    lookup = [([0], [1]), ([0, 1], [1, 2]), ([0, 1, 2], [1, 2, 3])]
    dist = build_target_distribution(lookup, 2, "label_smoothing", epsilon=0.1)
    assert dist[2] == pytest.approx(0.9)
    assert dist[0] == pytest.approx(0.05)
    assert dist[1] == pytest.approx(0.05)


def test_target_distribution_prefix_simple() -> None:
    lookup = [([0], [1]), ([0, 1], [1, 2])]
    dist = build_target_distribution(lookup, 1, "prefix_simple", epsilon=0.2)
    assert dist[1] == pytest.approx(0.8)
    assert dist[0] == pytest.approx(0.2)


def test_target_distribution_prefix_softmax() -> None:
    lookup = [([0], [1]), ([0, 1], [1, 2]), ([0, 1, 2], [1, 2, 3])]
    dist = build_target_distribution(lookup, 2, "prefix_softmax", epsilon=0.5)
    weight_0 = math.exp(1) / (math.exp(1) + math.exp(2))
    weight_1 = math.exp(2) / (math.exp(1) + math.exp(2))
    assert dist[2] == pytest.approx(0.5)
    assert dist[0] == pytest.approx(0.5 * weight_0)
    assert dist[1] == pytest.approx(0.5 * weight_1)


def test_target_distribution_prefix_softmax_normalized() -> None:
    lookup = [([0], [1]), ([0, 1], [1, 2]), ([0, 1, 2], [1, 2, 3])]
    dist = build_target_distribution(
        lookup, 2, "prefix_softmax_normalized", epsilon=0.5
    )
    weight_0 = math.exp(1 / 3) / (math.exp(1 / 3) + math.exp(2 / 3))
    weight_1 = math.exp(2 / 3) / (math.exp(1 / 3) + math.exp(2 / 3))
    assert dist[2] == pytest.approx(0.5)
    assert dist[0] == pytest.approx(0.5 * weight_0)
    assert dist[1] == pytest.approx(0.5 * weight_1)
