import pandas as pd
import pytest
from scripts.puntuar_conjuntos_escenarios_nuevos import verify_event_novelty


def test_novelty_rejects_any_training_factor_or_shape():
    data = pd.DataFrame({'shape_id': ['new', 'new'], 'factor_mult': [1.9, 2.1]})
    verify_event_novelty(data, {'old'}, {2.0, 2.2})
    with pytest.raises(ValueError):
        verify_event_novelty(data, {'new'}, {2.0})
    with pytest.raises(ValueError):
        verify_event_novelty(data, {'old'}, {2.1})
