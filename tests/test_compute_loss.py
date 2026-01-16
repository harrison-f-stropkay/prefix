import torch
import torch.nn.functional as F

from prefix.objectives import build_lookup
from prefix.train import build_prefix_tables, build_prefix_weights, compute_loss


class DummyTokenizer:
    def __init__(self, vocab: list[str], special_ids: list[int] | None = None) -> None:
        self._vocab = vocab
        self.vocab_size = len(vocab)
        self.all_special_ids = special_ids or []

    def __len__(self) -> int:
        return len(self._vocab)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self._vocab[token_id]


def test_compute_loss_cross_entropy_matches_torch() -> None:
    logits = torch.tensor([[[2.0, 0.0, -1.0]]])
    labels = torch.tensor([[0]])
    expected = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss = compute_loss(
        logits,
        labels,
        objective_type="cross_entropy",
        epsilon=0.1,
        prefix_tables=None,
    )
    assert torch.allclose(loss, expected)


def test_compute_loss_label_smoothing_matches_torch() -> None:
    logits = torch.tensor([[[2.0, 0.0, -1.0]]])
    labels = torch.tensor([[0]])
    expected = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        label_smoothing=0.2,
    )
    loss = compute_loss(
        logits,
        labels,
        objective_type="label_smoothing",
        epsilon=0.2,
        prefix_tables=None,
    )
    assert torch.allclose(loss, expected)


def test_compute_loss_prefix_simple_with_unicode_and_special_tokens() -> None:
    vocab = ["e", "e\u0301", "e\u0301b", "<eos>", ""]
    tokenizer = DummyTokenizer(vocab, special_ids=[3])
    lookup = build_lookup(tokenizer)
    prefix_weights = build_prefix_weights(
        lookup,
        {
            "type": "prefix_simple",
            "epsilon": 0.2,
        },
    )
    prefix_tables = build_prefix_tables(prefix_weights)

    logits = torch.tensor([[[1.0, 0.5, -0.5, 0.2, -1.0]]])
    labels = torch.tensor([[2]])
    loss = compute_loss(
        logits,
        labels,
        objective_type="prefix_simple",
        epsilon=0.2,
        prefix_tables=prefix_tables,
    )

    log_probs = F.log_softmax(logits.view(-1, logits.size(-1)), dim=-1)
    logp_true = log_probs[0, 2]
    logp_prefix = (log_probs[0, 0] + log_probs[0, 1]) / 2.0
    expected = -0.8 * logp_true - 0.2 * logp_prefix
    assert torch.allclose(loss, expected)

    labels_eos = torch.tensor([[3]])
    loss_eos = compute_loss(
        logits,
        labels_eos,
        objective_type="prefix_simple",
        epsilon=0.2,
        prefix_tables=prefix_tables,
    )
    expected_eos = F.cross_entropy(logits.view(-1, logits.size(-1)), labels_eos.view(-1))
    assert torch.allclose(loss_eos, expected_eos)
