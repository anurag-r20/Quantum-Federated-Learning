"""Unit tests for src/utils/engine.py: the train / test loops.

The model in these tests returns its input as the logits (nn.Identity), so predictions, accuracy and
cross-entropy can be worked out by hand.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils import engine


def one_hot_logits(classes: list[int], scale: float = 2.0) -> torch.Tensor:
    """Logits that put `scale` on the given class and 0 on the other three."""
    return torch.nn.functional.one_hot(torch.tensor(classes), 4).float() * scale


def test__test__given_logits_that_predict_every_label__returns_100_percent_and_the_hand_derived_loss() -> None:
    """Known oracle: logits (2, 0, 0, 0) for the true class give CE = log(1 + 3 e^-2) = 0.340748 per sample."""
    # Arrange
    loader = DataLoader(TensorDataset(one_hot_logits([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3])), batch_size=4)

    # Act
    loss, accuracy, y_pred, y_true, y_proba = engine.test(torch.nn.Identity(), loader, torch.nn.CrossEntropyLoss(), "cpu")

    # Assert
    assert accuracy == pytest.approx(100.0)
    assert loss == pytest.approx(math.log(1 + 3 * math.exp(-2)), rel=1e-6)
    assert list(y_pred) == [0, 1, 2, 3]
    assert y_proba.sum(axis=1) == pytest.approx([1.0] * 4)


@pytest.mark.xfail(
    reason="BUG-01: engine.test averages per-batch accuracies, so a smaller last batch is over-weighted",
    raises=AssertionError,
)
def test__test__given_a_smaller_last_batch__returns_the_accuracy_over_all_samples() -> None:
    """Known oracle: 2 of 3 samples are right, so accuracy is 66.67%. With batch_size=2 the loader yields
    [right, right] and [wrong]; averaging the two batch accuracies (100% and 0%) gives 50% instead."""
    # Arrange
    logits = one_hot_logits([0, 1, 2])
    loader = DataLoader(TensorDataset(logits, torch.tensor([0, 1, 3])), batch_size=2)

    # Act
    _, accuracy, *_ = engine.test(torch.nn.Identity(), loader, torch.nn.CrossEntropyLoss(), "cpu")

    # Assert
    assert accuracy == pytest.approx(200 / 3)


def test__train_step__given_a_separable_problem__lowers_the_loss() -> None:
    """Provable relation: gradient steps on a linearly separable problem reduce the training loss."""
    # Arrange
    torch.manual_seed(0)
    x = torch.cat([torch.randn(32, 2) + 3, torch.randn(32, 2) - 3])
    y = torch.cat([torch.zeros(32, dtype=torch.long), torch.ones(32, dtype=torch.long)])
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)
    model = torch.nn.Linear(2, 2)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_before = loss_fn(model(x), y).item()

    # Act
    for _ in range(5):
        engine.train_step(model, loader, loss_fn, torch.optim.SGD(model.parameters(), lr=0.1), "cpu")

    # Assert
    assert loss_fn(model(x), y).item() < loss_before
