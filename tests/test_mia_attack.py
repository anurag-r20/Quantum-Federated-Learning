"""Unit tests for src/mia_attack.py: the model inversion attack on the final model.

Oracles: PCA identities (orthonormal components, decode(0) = mean), a "brightness" model whose class
probabilities are provably monotone in mean intensity, identity look-ups, and hand-counted attack accuracy.
"""

from __future__ import annotations

import json

import pytest
import torch

import mia_attack as mia

CLASSES = ("dark", "bright")


class BrightnessModel(torch.nn.Module):
    """Logits (-10 m, +10 m) for mean intensity m: P(bright) = sigmoid(20 m) rises with brightness."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = x.mean(dim=(1, 2, 3))
        return torch.stack([-10 * m, 10 * m], dim=1)


class ConstantModel(torch.nn.Module):
    """Always predicts class 0 (a deliberately useless evaluator)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[5.0, 0.0]]).expand(len(x), -1)


def public_images(n: int = 40, size: int = 8, seed: int = 0) -> torch.Tensor:
    """Images whose main variation is overall brightness, plus a little noise, in [-1, 1]."""
    generator = torch.Generator().manual_seed(seed)
    levels = torch.linspace(-0.8, 0.8, n).view(n, 1, 1, 1)
    return (levels + 0.05 * torch.randn(n, 1, size, size, generator=generator)).clamp(-1, 1)


def test__fit_pca_prior__given_public_images__decodes_the_zero_latent_to_their_mean() -> None:
    """Known oracle: x(0) = mean, the pixel-wise average of the public images."""
    # Arrange
    images = public_images()

    # Act
    prior = mia.fit_pca_prior(images, n_components=3)

    # Assert
    assert torch.allclose(mia.decode(prior, torch.zeros(1, 3)), images.mean(0, keepdim=True), atol=1e-5)


def test__fit_pca_prior__given_public_images__returns_orthonormal_components_in_decreasing_variance() -> None:
    """PCA properties: C C^T = I and the per-component standard deviations are sorted, largest first."""
    # Act
    prior = mia.fit_pca_prior(public_images(), n_components=5)

    # Assert
    components, std = prior["components"], prior["std"]
    assert torch.allclose(components @ components.T, torch.eye(5), atol=1e-5)
    assert torch.all(std[:-1] >= std[1:]) and torch.all(std >= 0)


def test__decode__given_standardized_projections_of_the_public_images__reproduces_those_images() -> None:
    """Round-trip relation for x(z) = mean + (z * std) @ components with z in standard-deviation units:
    for images lying exactly in a 2-D subspace, z = ((x - mean) @ C^T) / std has unit sample variance per
    component and decodes back to x."""
    # Arrange
    generator = torch.Generator().manual_seed(7)
    basis = torch.linalg.qr(torch.randn(64, 2, generator=generator))[0].T          # 2 orthonormal directions
    coefficients = torch.randn(30, 2, generator=generator) * torch.tensor([3.0, 1.0])
    images = (0.1 + 0.05 * coefficients @ basis).view(30, 1, 8, 8)                 # stays inside [-1, 1]
    prior = mia.fit_pca_prior(images, n_components=2)
    flat = images.view(30, -1)
    z = ((flat - prior["mean"]) @ prior["components"].T) / prior["std"]

    # Act
    decoded = mia.decode(prior, z)

    # Assert
    assert z.var(dim=0, unbiased=True).tolist() == pytest.approx([1.0, 1.0], abs=1e-4)
    assert torch.allclose(decoded, images, atol=1e-5)


def test__decode__given_a_huge_latent__stays_within_the_model_input_range() -> None:
    # Arrange
    prior = mia.fit_pca_prior(public_images(), n_components=3)

    # Act
    images = mia.decode(prior, torch.full((2, 3), 1e3))

    # Assert
    assert images.min() >= -1 and images.max() <= 1


def test__invert_class__given_a_brightness_model__returns_a_brighter_image_for_bright_than_for_dark() -> None:
    """Provable relation: P(bright) increases with mean intensity, so inverting 'bright' must give a brighter
    image than inverting 'dark', each with high confidence."""
    # Arrange
    prior = mia.fit_pca_prior(public_images(), n_components=3)
    cfg = mia.MIAConfig(n_components=3, iterations=150, restarts=2)
    model = BrightnessModel()

    # Act
    dark, dark_confidence = mia.invert_class(model, 0, prior, cfg)
    bright, bright_confidence = mia.invert_class(model, 1, prior, cfg)

    # Assert
    assert float(bright.mean()) > 0 > float(dark.mean())
    assert min(dark_confidence, bright_confidence) > 0.9


def test__nearest_private_images__given_one_of_the_private_images__ranks_it_first_with_ssim_one() -> None:
    """Identity oracle: an image is its own nearest neighbour."""
    # Arrange
    private = torch.rand(12, 1, 16, 16, generator=torch.Generator().manual_seed(3)) * 2 - 1
    labels = torch.arange(12) % 4

    # Act
    indices, scores, neighbour_labels = mia.nearest_private_images(private[5:6], private, labels, k=3)

    # Assert
    assert int(indices[0]) == 5 and float(scores[0]) == pytest.approx(1.0, abs=1e-5)
    assert int(neighbour_labels[0]) == int(labels[5])


@pytest.mark.xfail(
    reason="BUG-02: nearest_private_images asks torch.topk for k=10 neighbours even when fewer images exist",
    raises=RuntimeError,
)
def test__nearest_private_images__given_fewer_private_images_than_k__returns_all_of_them() -> None:
    """Contract: with 3 private images and the default k, all 3 are returned, best first."""
    # Arrange
    private = torch.rand(3, 1, 16, 16, generator=torch.Generator().manual_seed(6)) * 2 - 1

    # Act
    indices, _, _ = mia.nearest_private_images(private[1:2], private, torch.arange(3))

    # Assert
    assert sorted(indices.tolist()) == [0, 1, 2] and int(indices[0]) == 1


def test__run_mia__given_an_evaluator_that_always_says_class_0__scores_one_class_out_of_two() -> None:
    """Known oracle: the evaluator agrees with the target class only for class 0, so accuracy = 1/2."""
    # Arrange
    # 16px: SSIM in the nearest-neighbour search uses an 11x11 window
    prior = mia.fit_pca_prior(public_images(size=16), n_components=3)
    private = public_images(n=12, size=16, seed=4)
    labels = (private.mean(dim=(1, 2, 3)) > 0).long()

    # Act
    report = mia.run_mia(BrightnessModel(), prior, ConstantModel(), private, labels, CLASSES,
                         mia.MIAConfig(n_components=3, iterations=50, restarts=1))

    # Assert
    assert report["attack_accuracy"] == pytest.approx(0.5)
    assert [r["evaluation_classifier_prediction"] for r in report["per_class"]] == ["dark", "dark"]


def test__save_report__given_a_report_with_images__writes_json_without_the_images(tmp_path) -> None:
    # Arrange
    report = {"attack_accuracy": 0.75, "seconds": 1.0,
              "per_class": [{"class": "dark", "image": torch.zeros(1, 1, 2, 2), "target_model_confidence": 0.9}]}
    path = tmp_path / "mia.json"

    # Act
    mia.save_report(report, path)

    # Assert
    saved = json.loads(path.read_text())
    assert saved["attack_accuracy"] == 0.75
    assert saved["per_class"] == [{"class": "dark", "target_model_confidence": 0.9}]


def test__train_evaluation_classifier__given_classes_separated_by_brightness__classifies_held_out_images() -> None:
    """Oracle: four brightness levels are linearly separable, so a trained CNN should get nearly all
    held-out images right (>= 90%). 64 images per class for 10 epochs gives 40 optimizer steps, enough for
    BatchNorm's running statistics (used in eval mode) to settle."""
    # Arrange
    generator = torch.Generator().manual_seed(5)

    def make(n_per_class: int):
        labels = torch.arange(4).repeat_interleave(n_per_class)
        images = (-0.75 + 0.5 * labels.view(-1, 1, 1, 1) + 0.05 * torch.randn(len(labels), 1, 16, 16,
                                                                                generator=generator))
        return images.clamp(-1, 1), labels

    (x_train, y_train), (x_test, y_test) = make(64), make(6)

    # Act
    _, test_accuracy = mia.train_evaluation_classifier(x_train, y_train, x_test, y_test, epochs=10)

    # Assert
    assert test_accuracy >= 90
