import numpy as np
import pandas as pd
import pytest

from scripts.evaluacion_conjunta_hidrogramas_nuevos import validate_alignment


def test_validate_alignment_rejects_missing_or_duplicate_nodes():
    inputs = pd.DataFrame({'run_id': [1, 1], 'node_id': ['a', 'b']})
    truth = pd.DataFrame({'run_id': [1, 1], 'node_id': ['a', 'b']})
    validate_alignment(inputs, truth)
    with pytest.raises(ValueError):
        validate_alignment(inputs, truth.iloc[:1])
    with pytest.raises(ValueError):
        validate_alignment(inputs, pd.concat([truth, truth.iloc[:1]]))
    with pytest.raises(ValueError):
        validate_alignment(inputs, truth.assign(node_id=['a', 'c']))
