"""Unit tests for src/gia_attack.py: the server-side gradient inversion attack.

Oracles: the label and image we feed in (the attack must recover them), gradients computed independently
with torch.autograd, a forward hook for the first FC layer's input, hand-computed metric values, and the
security property that the server's public context cannot decrypt.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import gia_attack as gia
from models import MRINet
from utils import data_setup, security

IMG = 16  # small images keep the attack fast; MRINet's FC input is 32 * (16 / 4)^2 = 512


def model_and_image(seed: int = 0):
    torch.manual_seed(seed)
    model = MRINet(num_classes=4, img_size=IMG)
    # a smooth "scan": a bright disc on a dark background, in model space [-1, 1]
    yy, xx = torch.meshgrid(torch.arange(IMG), torch.arange(IMG), indexing="ij")
    disc = (((yy - 7.5) ** 2 + (xx - 6.5) ** 2) < 25).float()
    image = (disc * 1.4 - 0.7).view(1, 1, IMG, IMG)
    return model, image


def true_gradients(model, images, labels) -> dict:
    """Independent oracle: the client's gradient, straight from autograd."""
    loss = F.cross_entropy(model(images), labels)
    grads = torch.autograd.grad(loss, list(model.parameters()))
    return {name: g for (name, _), g in zip(model.named_parameters(), grads)}


def fedsgd_observation(model, images, labels, lr=0.01, tensor_type=None, update_tensors=None) -> dict:
    """What federated.py records for a FedSGD round: global weights, the client's weights after one SGD step."""
    grads = true_gradients(model, images, labels)
    global_w = [v.detach().numpy() for v in model.state_dict().values()]
    client_w = [w - lr * grads[name].numpy() for name, w in zip(model.state_dict().keys(), global_w)]
    global_params = security.weights_to_parameters(global_w)
    client_params = security.weights_to_parameters(client_w)
    return {
        "round": 1,
        "fit_config": {"fedsgd": True, "lr": lr, "batch_size": len(labels)},
        "tensor_type": global_params.tensor_type,
        "global_tensors": global_params.tensors,
        "updates": {"0": {"tensors": update_tensors or client_params.tensors,
                          "tensor_type": tensor_type or client_params.tensor_type,
                          "num_examples": len(labels)}},
    }


@pytest.mark.parametrize("label", [0, 1, 2, 3], ids=["glioma", "meningioma", "notumor", "pituitary"])
def test__recover_labels_idlg__given_a_single_image_gradient__returns_the_true_label(label) -> None:
    """Known oracle (iDLG): d CE / d bias = softmax - one_hot(y); only the true class entry is negative."""
    # Arrange
    model, image = model_and_image()
    grads = true_gradients(model, image, torch.tensor([label]))

    # Act
    recovered = gia.recover_labels_idlg(grads, model)

    # Assert
    assert recovered.tolist() == [label]


def test__recover_fc_input__given_a_single_image_gradient__returns_the_first_fc_layers_input() -> None:
    """Oracle: grad_W = grad_b * h^T for batch 1, so h is exact; compared with the activation computed directly."""
    # Arrange
    model, image = model_and_image()
    grads = true_gradients(model, image, torch.tensor([2]))
    expected = torch.flatten(model.features(image), 1)[0].detach()

    # Act
    recovered = gia.recover_fc_input(grads, model)

    # Assert
    assert torch.allclose(recovered, expected, rtol=1e-4, atol=1e-6)


def test__observed_gradient__given_a_plaintext_fedsgd_update__returns_the_clients_gradient() -> None:
    """Oracle: SGD defines w_client = w_global - lr * g, so (w_global - w_client) / lr = g (autograd)."""
    # Arrange
    model, image = model_and_image()
    expected = true_gradients(model, image, torch.tensor([1]))
    observation = fedsgd_observation(model, image, torch.tensor([1]))

    # Act
    grads, _ = gia.observed_gradient(observation, "0", MRINet(num_classes=4, img_size=IMG))

    # Assert
    for name, g in expected.items():
        assert (grads[name] - g).norm() <= 1e-3 * g.norm() + 1e-7, name


def test__observed_gradient__given_an_encrypted_update__raises_permission_error() -> None:
    # Arrange
    model, image = model_and_image()
    observation = fedsgd_observation(model, image, torch.tensor([1]), tensor_type=security.TENSOR_TYPE_CKKS)

    # Act
    with pytest.raises(Exception) as error:
        gia.observed_gradient(observation, "0", model)

    # Assert
    assert isinstance(error.value, PermissionError)


@pytest.mark.parametrize(
    "transform, expected",
    [(lambda g: g, 0.0), (lambda g: 3.0 * g, 0.0), (lambda g: -g, 2.0)],
    ids=["identical", "scaled-copy", "opposite"],
)
def test__cosine_gradient_loss__given_related_gradients__returns_one_minus_their_cosine(transform, expected) -> None:
    """Known oracle: 1 - cos is 0 for parallel vectors (any positive scale) and 2 for opposite ones."""
    # Arrange
    grads = [torch.tensor([1.0, 2.0]), torch.tensor([[3.0, -1.0]])]

    # Act
    loss = gia.cosine_gradient_loss([transform(g) for g in grads], grads)

    # Assert
    assert float(loss) == pytest.approx(expected, abs=1e-6)


def test__cosine_gradient_loss__given_orthogonal_gradients__returns_one() -> None:
    # Act
    loss = gia.cosine_gradient_loss([torch.tensor([1.0, 0.0])], [torch.tensor([0.0, 5.0])])

    # Assert
    assert float(loss) == pytest.approx(1.0, abs=1e-6)


def test__total_variation__given_a_two_by_two_vertical_edge__returns_one() -> None:
    """Known oracle: [[0, 1], [0, 1]] has vertical differences (0, 0) -> mean 0 and horizontal
    differences (1, 1) -> mean 1, so TV = 0 + 1 = 1."""
    # Act
    tv = gia.total_variation(torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]]))

    # Assert
    assert float(tv) == pytest.approx(1.0)


def test__psnr__given_a_uniform_error_of_0_1__returns_20_db() -> None:
    """Known oracle: MSE = 0.01, PSNR = 10 log10(1 / 0.01) = 20 dB."""
    # Act
    value = gia.psnr(torch.zeros(1, 1, 4, 4), torch.full((1, 1, 4, 4), 0.1))

    # Assert
    assert float(value) == pytest.approx(20.0, abs=1e-4)


def test__ssim__given_identical_images__returns_one() -> None:
    # Arrange
    images = torch.rand(2, 1, 16, 16, generator=torch.Generator().manual_seed(0))

    # Act
    value = gia.ssim(images, images)

    # Assert
    assert value.tolist() == pytest.approx([1.0, 1.0], abs=1e-5)


def test__ssim__given_swapped_arguments__returns_the_same_value() -> None:
    """Metamorphic relation: SSIM is symmetric."""
    # Arrange
    generator = torch.Generator().manual_seed(1)
    a, b = torch.rand(1, 1, 16, 16, generator=generator), torch.rand(1, 1, 16, 16, generator=generator)

    # Act
    forward, backward = gia.ssim(a, b), gia.ssim(b, a)

    # Assert
    assert float(forward) == pytest.approx(float(backward))


def test__evaluate_reconstruction__given_a_permuted_batch__matches_each_reconstruction_to_its_original() -> None:
    """Oracle: reconstructions are the ground truth in order (2, 0, 1), so the matching must undo that
    and every pair is identical (PSNR capped at 100 dB by the 1e-10 MSE floor)."""
    # Arrange
    truth = torch.rand(3, 1, 16, 16, generator=torch.Generator().manual_seed(2)) * 2 - 1

    # Act
    metrics = gia.evaluate_reconstruction(truth[[2, 0, 1]], truth)

    # Assert
    assert metrics["order"] == [2, 0, 1]
    assert metrics["psnr"] == pytest.approx([100.0] * 3)


def test__reconstruct__given_one_image_fedsgd_gradient__recovers_the_private_image() -> None:
    """The attack's promise. Oracle: the image we fed in; PSNR above 25 dB means a near-exact copy."""
    # Arrange
    model, image = model_and_image()
    grads = true_gradients(model, image, torch.tensor([3]))
    cfg = gia.GIAConfig(iterations=600, restarts=1, tv_weight=5e-3)

    # Act
    best = gia.reconstruct(model, grads, gia.recover_labels_idlg(grads, model), (1, IMG, IMG), cfg,
                           fc_input=gia.recover_fc_input(grads, model), log_every=0)

    # Assert
    assert float(gia.psnr(data_setup.to_pixels(best["images"]), data_setup.to_pixels(image))) > 25


def test__attack_plaintext__given_a_batch_of_two_without_known_labels__raises_value_error() -> None:
    # Arrange
    model, image = model_and_image()
    observation = fedsgd_observation(model, torch.cat([image, -image]), torch.tensor([0, 1]))

    # Act
    with pytest.raises(Exception) as error:
        gia.attack_plaintext(observation, MRINet(num_classes=4, img_size=IMG), "0", data_shape=(1, IMG, IMG))

    # Assert
    assert isinstance(error.value, ValueError)


def test__attempt_on_encrypted__given_the_servers_public_context__reports_blocked(
    ckks_secret_context, ckks_public_context
) -> None:
    """Security property: without the secret key, TenSEAL refuses to decrypt the client's update."""
    # Arrange
    model, image = model_and_image()
    ciphertext = security.encrypt_array(ckks_secret_context, np.ones(10)).serialize()
    observation = fedsgd_observation(model, image, torch.tensor([0]), tensor_type=security.TENSOR_TYPE_CKKS,
                                     update_tensors=[ciphertext])

    # Act
    report = gia.attempt_on_encrypted(observation, ckks_public_context, "0")

    # Assert
    assert report["status"] == "blocked"
    assert "secret_key" in report["error"]


def test__attempt_on_encrypted__given_a_context_holding_the_secret_key__reports_not_blocked(
    ckks_secret_context,
) -> None:
    """Differential check of the report: with the client's key the same ciphertext decrypts."""
    # Arrange
    model, image = model_and_image()
    ciphertext = security.encrypt_array(ckks_secret_context, np.ones(10)).serialize()
    observation = fedsgd_observation(model, image, torch.tensor([0]), tensor_type=security.TENSOR_TYPE_CKKS,
                                     update_tensors=[ciphertext])

    # Act
    report = gia.attempt_on_encrypted(observation, ckks_secret_context, "0")

    # Assert
    assert report["status"] == "NOT BLOCKED"


def test__ciphertext_preview__given_known_bytes__returns_them_as_a_square_image() -> None:
    """Known oracle: bytes 0..15 as a 4x4 image, row by row."""
    # Arrange
    observation = {"updates": {"0": {"tensors": [bytes(range(32))]}}}

    # Act
    preview = gia.ciphertext_preview(observation, "0", side=4)

    # Assert
    assert preview.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
