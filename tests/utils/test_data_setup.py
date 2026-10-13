"""Unit tests for src/utils/data_setup.py: client partitioning, the cached MRI loader and pixel scaling.

Expected values come from hand-worked label layouts, conservation properties (every index assigned once)
and the definition of the [-1, 1] scaling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import datasets

from utils import data_setup

# glioma=0, meningioma=1, notumor=2, pituitary=3 (alphabetical ImageFolder order)
HAND_TARGETS = np.array([0, 0, 1, 3, 3, 3, 2, 2, 2, 2, 2, 2])


def write_image_folder(root: Path, classes=data_setup.MRI_CLASSES, per_class: int = 2, size: int = 20) -> Path:
    """Training/ and Testing/ folders with constant-intensity PNGs: class c has intensity 30 + 60 * c."""
    for split in ("Training", "Testing"):
        for c, name in enumerate(classes):
            folder = root / split / name
            folder.mkdir(parents=True)
            for i in range(per_class):
                Image.new("RGB", (size, size), color=(30 + 60 * c,) * 3).save(folder / f"{i}.png")
    return root


def test__partition_indices__given_iid__assigns_every_index_to_exactly_one_client_in_near_equal_parts() -> None:
    """Conservation property: the union is range(n), the parts are disjoint, and sizes differ by at most 1."""
    # Arrange
    targets = np.zeros(103, dtype=int)

    # Act
    parts = data_setup.partition_indices(targets, 3, "iid", seed=0)

    # Assert
    merged = np.concatenate(parts)
    assert sorted(merged.tolist()) == list(range(103))
    assert sorted(len(p) for p in parts) == [34, 34, 35]


def test__partition_indices__given_iid_and_the_same_seed__returns_the_same_split() -> None:
    # Act
    first = data_setup.partition_indices(np.zeros(50), 3, "iid", seed=7)
    second = data_setup.partition_indices(np.zeros(50), 3, "iid", seed=7)

    # Assert
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test__partition_indices__given_non_iid__gives_each_client_one_tumour_class_and_a_share_of_no_tumour() -> None:
    """Known oracle: glioma {0, 1}, meningioma {2}, pituitary {3, 4, 5}, no tumour {6..11} split in thirds:
    client 0 = {0, 1, 6, 7}, client 1 = {2, 8, 9}, client 2 = {3, 4, 5, 10, 11}."""
    # Act
    parts = data_setup.partition_indices(HAND_TARGETS, 3, "non_iid")

    # Assert
    assert [sorted(p.tolist()) for p in parts] == [[0, 1, 6, 7], [2, 8, 9], [3, 4, 5, 10, 11]]


@pytest.mark.parametrize(
    "num_clients, strategy",
    [(2, "non_iid"), (3, "dirichlet")],
    ids=["non-iid-needs-three-clients", "unknown-strategy"],
)
def test__partition_indices__given_an_unsupported_setting__raises_value_error(num_clients, strategy) -> None:
    # Act
    with pytest.raises(Exception) as error:
        data_setup.partition_indices(HAND_TARGETS, num_clients, strategy)

    # Assert
    assert isinstance(error.value, ValueError)


def test__split_data_client__given_an_mri_image_folder_and_non_iid__returns_class_specific_subsets(tmp_path) -> None:
    """Known oracle: 2 images per class at indices glioma {0, 1}, meningioma {2, 3}, no tumour {4, 5},
    pituitary {6, 7}; the no-tumour pair splits in thirds as [4], [5], [], so labels are
    [0, 0, 2], [1, 1, 2] and [3, 3]."""
    # Arrange
    folder = datasets.ImageFolder(str(write_image_folder(tmp_path) / "Training"))

    # Act
    subsets = data_setup.split_data_client(folder, 3, seed=0, partition_strategy="non_iid", dataset_type="MRI")

    # Assert
    labels = [sorted(folder.targets[i] for i in subset.indices) for subset in subsets]
    assert labels == [[0, 0, 2], [1, 1, 2], [3, 3]]


def test__split_data_client__given_non_iid_for_a_non_mri_dataset__raises_not_implemented_error(tmp_path) -> None:
    # Arrange
    folder = datasets.ImageFolder(str(write_image_folder(tmp_path) / "Training"))

    # Act
    with pytest.raises(Exception) as error:
        data_setup.split_data_client(folder, 3, seed=0, partition_strategy="non_iid", dataset_type="cifar")

    # Assert
    assert isinstance(error.value, NotImplementedError)


def test__load_mri_arrays__given_an_image_folder__returns_grayscale_arrays_labelled_by_class_folder(tmp_path) -> None:
    """Known oracle: resizing a constant image keeps its value, and RGB (v, v, v) converts to gray v,
    so class c images are 30 + 60c everywhere."""
    # Arrange
    root = write_image_folder(tmp_path)

    # Act
    arrays = data_setup.load_mri_arrays(str(root), img_size=8)

    # Assert
    assert arrays["x_train"].shape == (8, 1, 8, 8) and arrays["x_train"].dtype == np.uint8
    assert arrays["y_train"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [np.unique(x).tolist() for x in arrays["x_train"][::2]] == [[30], [90], [150], [210]]
    assert arrays["classes"] == data_setup.MRI_CLASSES


def test__load_mri_arrays__given_a_cache_from_an_earlier_call__returns_the_same_arrays_without_the_images(
    tmp_path,
) -> None:
    """Relation: the second call reads cache_gray_<size>.npz, so it still works after the images are gone."""
    # Arrange
    root = write_image_folder(tmp_path)
    first = data_setup.load_mri_arrays(str(root), img_size=8)
    for png in root.rglob("*.png"):
        png.unlink()

    # Act
    second = data_setup.load_mri_arrays(str(root), img_size=8)

    # Assert
    assert (root / "cache_gray_8.npz").exists()
    assert np.array_equal(first["x_train"], second["x_train"]) and second["classes"] == first["classes"]


def test__load_mri_arrays__given_unexpected_class_folders__raises_value_error(tmp_path) -> None:
    # Arrange
    root = write_image_folder(tmp_path, classes=("cat", "dog", "fish", "owl"))

    # Act
    with pytest.raises(Exception) as error:
        data_setup.load_mri_arrays(str(root), img_size=8)

    # Assert
    assert isinstance(error.value, ValueError)


def test__to_model_input__given_black_white_and_one_fifth_grey__maps_to_minus_one_plus_one_and_minus_0_6() -> None:
    """Known oracle: x = (p / 255 - 0.5) / 0.5, so 0 -> -1, 255 -> 1, 51 -> (0.2 - 0.5) / 0.5 = -0.6."""
    # Act
    scaled = data_setup.to_model_input(np.array([0, 255, 51], dtype=np.uint8))

    # Assert
    assert scaled.tolist() == pytest.approx([-1.0, 1.0, -0.6])


def test__to_pixels__given_model_inputs__inverts_to_model_input() -> None:
    """Identity oracle: to_pixels(to_model_input(p)) == p / 255 for every uint8 value."""
    # Arrange
    pixels = np.arange(256, dtype=np.uint8)

    # Act
    restored = data_setup.to_pixels(data_setup.to_model_input(pixels))

    # Assert
    assert torch.allclose(restored, torch.arange(256) / 255.0, atol=1e-6)


def test__to_pixels__given_values_outside_the_model_range__clamps_to_zero_and_one() -> None:
    # Act
    clamped = data_setup.to_pixels(torch.tensor([-3.0, 3.0]))

    # Assert
    assert clamped.tolist() == [0.0, 1.0]
