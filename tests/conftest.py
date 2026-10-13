"""Shared fixtures for the test suite.

matplotlib must use its non-interactive "Agg" backend before pyplot is imported anywhere, so that modules which
plot can be imported and tested on a machine with no display. Everything a test writes goes under tmp_path.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import tenseal as ts  # noqa: E402

from utils import data_setup, security  # noqa: E402

os.environ.setdefault("FLWR_TELEMETRY_ENABLED", "0")


@pytest.fixture(autouse=True)
def close_all_figures():
    """Closes every matplotlib figure after each test, so figures never leak between tests."""
    yield
    plt.close("all")


@pytest.fixture
def ckks_secret_context():
    """A client's CKKS context: holds the secret key."""
    return security.context()


@pytest.fixture
def ckks_public_context(ckks_secret_context):
    """The server's view of the same key pair: public key only."""
    return ts.context_from(ckks_secret_context.serialize(save_secret_key=False))


@pytest.fixture
def make_mri_cache():
    """Factory writing a small synthetic MRI cache (the format load_mri_arrays reads) and returning its folder.

    Class c images are filled with intensity 40 + 50 * c plus noise, so classes are separable by brightness.
    """

    def make(root: Path, img_size: int = 16, per_class_train: int = 12, per_class_test: int = 3, seed: int = 0) -> Path:
        rng = np.random.default_rng(seed)
        arrays = {}
        for split, per_class in (("train", per_class_train), ("test", per_class_test)):
            labels = np.repeat(np.arange(4), per_class)
            images = 40 + 50 * labels[:, None, None, None] + rng.integers(0, 20, size=(len(labels), 1, img_size, img_size))
            arrays[f"x_{split}"] = images.astype(np.uint8)
            arrays[f"y_{split}"] = labels.astype(np.int64)
        folder = root / "MRI"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / f"cache_gray_{img_size}.npz", classes=np.array(data_setup.MRI_CLASSES), **arrays)
        return folder

    return make
