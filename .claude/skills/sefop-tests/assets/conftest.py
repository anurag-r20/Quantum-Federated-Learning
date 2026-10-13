"""Shared fixtures for the test suite.

matplotlib must use its non-interactive "Agg" backend before pyplot is imported anywhere, so that modules which
plot can be imported and tested on a machine with no display. Delete the matplotlib parts if the repo has no
plotting.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)
import pytest  # noqa: E402

# Small copies of real data used by the tests (keep them small; large data stays outside the repo).
RESOURCES_DIR = Path(__file__).parent / "resources"


@pytest.fixture(autouse=True)
def close_all_figures():
    """Closes every matplotlib figure after each test, so figures never leak between tests."""
    yield
    plt.close("all")


@pytest.fixture
def resources_dir() -> Path:
    """The folder holding the test data copies."""
    return RESOURCES_DIR
