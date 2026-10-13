"""Unit tests for <package>/<module>.py.

One test per promised behaviour, named test__<unit>__<scenario>__<expected outcome>, laid out as
Arrange / Act / Assert. Expected values come from an oracle (a hand calculation, an independent
implementation, or a provable relation), never from running the code under test.
Tests marked xfail document a confirmed bug in the repo's bug list; delete the marker when the bug is fixed.
"""

from __future__ import annotations

import pytest

from package import module  # replace with the module under test


def test__function__given_a_small_known_case__returns_the_hand_derived_value() -> None:
    """Known oracle: write the derivation here so a reviewer can check it."""
    # Arrange
    data = ...

    # Act
    result = module.function(data)

    # Assert
    assert result == pytest.approx(...)


@pytest.mark.parametrize("value", [0.0, 1.5, -3.7, 1e15], ids=["zero", "positive", "negative", "large"])
def test__function__given_any_value__keeps_the_property(value: float) -> None:
    # Arrange
    ...

    # Act
    result = module.function(value)

    # Assert
    assert result == pytest.approx(value)  # the property that must hold for every input


def test__function__given_invalid_input__raises_value_error() -> None:
    # Act
    with pytest.raises(Exception) as error:  # Act records what happened...
        module.function(None)

    # Assert
    assert isinstance(error.value, ValueError)  # ...Assert checks which error it was


@pytest.mark.xfail(reason="BUG-01: <what is wrong, in one line>", raises=AssertionError)
def test__function__given_the_case_that_exposes_the_bug__returns_the_correct_value() -> None:
    # Act
    result = module.function(...)

    # Assert
    assert result == pytest.approx(...)  # the correct behaviour, which the code does not deliver yet
