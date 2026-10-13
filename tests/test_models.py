"""Unit tests for src/models.py: the attacked MRINet and the MIA evaluation classifier."""

from __future__ import annotations

import pytest
import torch

from models import EvaluationNet, MRINet


@pytest.mark.parametrize("img_size", [16, 64], ids=["16px", "64px"])
def test__mrinet__given_a_batch_of_grayscale_images__returns_one_logit_per_class(img_size) -> None:
    # Arrange
    model = MRINet(num_classes=4, img_size=img_size)

    # Act
    logits = model(torch.zeros(3, 1, img_size, img_size))

    # Assert
    assert logits.shape == (3, 4)


def test__mrinet__given_64px_images__has_529412_parameters() -> None:
    """Known oracle: conv 1->16 (16*9 + 16 = 160), conv 16->32 (32*16*9 + 32 = 4640),
    fc 32*16*16 -> 64 (8192*64 + 64 = 524352), fc 64 -> 4 (256 + 4 = 260); total 529412."""
    # Act
    count = sum(p.numel() for p in MRINet(num_classes=4, img_size=64).parameters())

    # Assert
    assert count == 529_412


def test__evaluation_net__given_64px_images__returns_one_logit_per_class() -> None:
    # Arrange
    model = EvaluationNet(num_classes=4).eval()

    # Act
    logits = model(torch.zeros(2, 1, 64, 64))

    # Assert
    assert logits.shape == (2, 4)
