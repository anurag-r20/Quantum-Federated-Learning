"""Unit tests for src/utils/common.py: getting/setting model parameters (plaintext and CKKS) and helpers.

Expected values come from identities (a model's own state_dict), hand-written constants, or files on disk.
"""

from __future__ import annotations

import numpy as np
import pytest
import tenseal as ts
import torch

from utils import common


def tiny_net(seed: int) -> torch.nn.Module:
    """A conv + linear model on 1x4x4 inputs: conv(1->2, 3x3) gives 2x2x2 = 8 features."""
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3), torch.nn.Flatten(), torch.nn.Linear(8, 3))


def test__get_parameters2__given_no_context__returns_the_state_dict_as_numpy_arrays() -> None:
    # Arrange
    net = tiny_net(0)

    # Act
    parameters = common.get_parameters2(net)

    # Assert
    expected = [v.numpy() for v in net.state_dict().values()]
    assert all(isinstance(p, np.ndarray) and np.array_equal(p, e) for p, e in zip(parameters, expected))


def test__set_parameters__given_another_models_plaintext_parameters__makes_the_models_identical() -> None:
    """Identity oracle: loading model A's parameters into model B makes B's state equal A's."""
    # Arrange
    source, target = tiny_net(0), tiny_net(1)

    # Act
    common.set_parameters(target, common.get_parameters2(source))

    # Assert
    assert all(torch.equal(a, b) for a, b in zip(source.state_dict().values(), target.state_dict().values()))


def test__set_parameters__given_parameters_encrypted_with_the_public_key__decrypts_them_into_the_model(
    ckks_secret_context, ckks_public_context, tmp_path
) -> None:
    """Identity oracle: encrypt with the server's public context, decrypt with the client's secret context,
    and every tensor comes back with its shape and values (up to CKKS noise)."""
    # Arrange
    source, target = tiny_net(0), tiny_net(1)
    encrypted = common.get_parameters2(source, ckks_public_context, save_path=str(tmp_path))

    # Act
    common.set_parameters(target, encrypted, ckks_secret_context, save_path=str(tmp_path))

    # Assert
    assert all(isinstance(p, ts.tensors.CKKSTensor) for p in encrypted)
    for a, b in zip(source.state_dict().values(), target.state_dict().values()):
        assert a.shape == b.shape
        np.testing.assert_allclose(b.numpy(), a.numpy(), atol=1e-4)


def test__choice_device__given_cpu__returns_cpu() -> None:
    # Act
    device = common.choice_device("cpu")

    # Assert
    assert device == "cpu"


@pytest.mark.parametrize(
    "dataset, expected",
    [
        ("MRI", ("glioma", "meningioma", "notumor", "pituitary")),
        ("cifar", ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")),
    ],
    ids=["mri", "cifar"],
)
def test__classes_string__given_a_known_dataset__returns_its_class_names_in_label_order(dataset, expected) -> None:
    """Known oracle: the MRI folders sort alphabetically (ImageFolder labels); CIFAR-10's documented order."""
    # Act
    classes = common.classes_string(dataset)

    # Assert
    assert classes == expected


def test__classes_string__given_an_unknown_dataset__returns_an_empty_tuple() -> None:
    """Characterization test: the current behaviour is to warn and return (); nothing raises."""
    # Act
    classes = common.classes_string("dna")

    # Assert
    assert classes == ()


def test__save_graphs__given_three_epochs_of_results__writes_the_accuracy_and_loss_figures(tmp_path) -> None:
    # Arrange
    results = {"train_acc": [50, 60, 70], "val_acc": [45, 55, 65], "train_loss": [1.0, 0.8, 0.6],
               "val_loss": [1.1, 0.9, 0.7]}

    # Act
    common.save_graphs(str(tmp_path) + "/", 3, results, "_client0")

    # Assert
    assert sorted(p.name for p in tmp_path.iterdir()) == ["Accuracy_curves_client0.png", "Loss_curves_client0.png"]
