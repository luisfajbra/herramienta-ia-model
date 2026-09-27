"""main._fmt_metric: metrics the evaluator reports as None or NaN.

A scenario with too few flooded nodes leaves f1/nse/error_pct_total as None,
and dict.get(key, default) does not substitute the default for a present-but-
None value, so formatting it directly raised TypeError mid-run.
"""

import math

import main


def test_formats_a_normal_float():
    assert main._fmt_metric(0.8123) == "0.812"


def test_honours_an_explicit_format_spec():
    assert main._fmt_metric(-3.456, ".1f") == "-3.5"


def test_none_becomes_the_fallback():
    assert main._fmt_metric(None) == "N/A"


def test_nan_becomes_the_fallback():
    assert main._fmt_metric(float("nan")) == "N/A"
    assert main._fmt_metric(math.nan, ".1f") == "N/A"


def test_infinity_is_still_formatted():
    """Infinity is a real (if extreme) result, not a missing measurement."""
    assert main._fmt_metric(float("inf")) == "inf"


def test_custom_fallback():
    assert main._fmt_metric(None, ".3f", "sin datos") == "sin datos"


def test_a_non_numeric_value_does_not_raise():
    assert main._fmt_metric("no soy un número") == "N/A"


def test_integers_are_accepted():
    assert main._fmt_metric(1) == "1.000"
