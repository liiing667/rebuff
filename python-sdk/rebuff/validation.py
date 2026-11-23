"""Validation helpers for detection requests.

Plain ``isinstance(x, float)`` checks are not sufficient: ``float("nan")`` and
``float("inf")`` are floats, but every comparison against them silently
evaluates to False, which lets malformed thresholds bypass detection. These
helpers reject NaN, Infinity and out-of-range values before any detection work
runs, while keeping the 0 and 1 boundaries (and the default thresholds) valid.
"""

import math
from typing import TypeVar


class RebuffValidationError(ValueError):
    """Raised when a detection request contains an invalid value."""


T = TypeVar("T")


def validate_threshold(value: float, parameter_name: str) -> float:
    """Validate that ``value`` is a finite score within the inclusive 0..1 range.

    Args:
        value: The threshold supplied by the caller.
        parameter_name: Parameter name used in the error message.

    Returns:
        The unchanged, validated threshold.

    Raises:
        RebuffValidationError: If the value is NaN, infinite, not a real
            number or outside [0, 1].
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RebuffValidationError(
            f"{parameter_name} must be a finite number between 0 and 1"
        )
    if math.isnan(value):
        raise RebuffValidationError(
            f"{parameter_name} must not be NaN"
        )
    if math.isinf(value):
        raise RebuffValidationError(
            f"{parameter_name} must be finite (Infinity is not allowed)"
        )
    if value < 0 or value > 1:
        raise RebuffValidationError(
            f"{parameter_name} must be between 0 and 1 (inclusive)"
        )
    return float(value)


def validate_boolean(value: object, parameter_name: str) -> bool:
    """Strict boolean validation.

    Truthy values such as 0/1 or the strings "true"/"false" are rejected
    instead of being silently coerced.
    """
    if not isinstance(value, bool):
        raise RebuffValidationError(f"{parameter_name} must be a boolean")
    return value
