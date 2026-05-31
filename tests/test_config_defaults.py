from swmm_resilience.config import DEFAULT_INP_FILE


def test_default_inp_file_exists():
    assert DEFAULT_INP_FILE.exists(), f"DEFAULT_INP_FILE does not exist: {DEFAULT_INP_FILE}"
