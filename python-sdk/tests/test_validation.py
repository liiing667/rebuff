"""Tests for detection request validation.

These tests are intentionally dependency-free: the validation module only uses
the Python standard library, so they can run without the full SDK dependency
tree (pydantic/openai/langchain).
"""

import math

import pytest

from rebuff.validation import (
    RebuffValidationError,
    validate_boolean,
    validate_threshold,
)


@pytest.mark.parametrize(
    "bad",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        -0.0001,
        1.0001,
        -1,
        2,
    ],
    ids=["nan", "infinity", "negative_infinity", "negative", "above_one",
         "negative_int", "int_above_one"],
)
def test_validate_threshold_rejects_invalid_values(bad):
    with pytest.raises(RebuffValidationError):
        validate_threshold(bad, "max_score")


@pytest.mark.parametrize(
    "good",
    [0, 1, 0.0, 1.0, 0.75, 0.9, 0.0001],
    ids=["zero_int", "one_int", "zero_float", "one_float",
         "default_heuristic", "default_vector", "small_positive"],
)
def test_validate_threshold_accepts_boundaries_and_defaults(good):
    assert validate_threshold(good, "max_score") == float(good)


def test_validate_threshold_rejects_non_numeric_types():
    for bad in ["0.5", True, False, None, [0.5]]:
        with pytest.raises(RebuffValidationError):
            validate_threshold(bad, "max_score")  # type: ignore[arg-type]


def test_validate_boolean_accepts_real_booleans():
    assert validate_boolean(True, "check") is True
    assert validate_boolean(False, "check") is False


@pytest.mark.parametrize("bad", ["true", "false", 0, 1, None, 0.0])
def test_validate_boolean_rejects_truthy_values(bad):
    with pytest.raises(RebuffValidationError):
        validate_boolean(bad, "check")  # type: ignore[arg-type]


def test_nan_is_not_finite_and_caught():
    assert math.isnan(float("nan"))
    with pytest.raises(RebuffValidationError, match="NaN"):
        validate_threshold(float("nan"), "max_model_score")
