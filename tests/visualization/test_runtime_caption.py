from swmm_resilience.visualization.runtime_caption import format_runtime_text


def test_format_seconds_ge_1_uses_two_decimals():
    assert format_runtime_text(1.85) == "Compute time: 1.85 s"


def test_format_seconds_lt_1_uses_four_decimals():
    assert format_runtime_text(0.024) == "Compute time: 0.0240 s"


def test_format_none_returns_none():
    assert format_runtime_text(None) is None
