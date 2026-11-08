import pandas as pd
import pytest

from swmm_resilience.simulation.swmm_api_io import get_node_series, get_link_series


class FakeOut:
    def __init__(self, index):
        self.index = index

    def get_part(self, kind, identifier):
        return pd.DataFrame({'depth': [1., 2.]}, index=self.index)


@pytest.mark.parametrize('reader', [get_node_series, get_link_series])
def test_datetime_out_times_are_seconds_since_explicit_simulation_start(reader):
    out = FakeOut(pd.to_datetime(['2025-01-04 00:03:00', '2025-01-04 00:06:00']))
    result = reader(out, '100C', simulation_start='2025-01-04 00:00:00')
    assert result.time_sec.tolist() == [180., 360.]


@pytest.mark.parametrize('reader', [get_node_series, get_link_series])
def test_datetime_out_rejects_missing_or_late_simulation_start(reader):
    out = FakeOut(pd.to_datetime(['2025-01-04 00:03:00', '2025-01-04 00:06:00']))
    with pytest.raises(ValueError, match='simulation_start'):
        reader(out, '100C')
    with pytest.raises(ValueError, match='before'):
        reader(out, '100C', simulation_start='2025-01-04 00:04:00')


@pytest.mark.parametrize('reader', [get_node_series, get_link_series])
def test_timedelta_out_times_remain_elapsed_seconds(reader):
    out = FakeOut(pd.to_timedelta([180, 360], unit='s'))
    assert reader(out, '100C').time_sec.tolist() == [180., 360.]


@pytest.mark.parametrize('reader', [get_node_series, get_link_series])
def test_datetime_out_rejects_missing_timestamp_or_timezone_mismatch(reader):
    with pytest.raises(ValueError, match='timestamps'):
        reader(FakeOut(pd.to_datetime(['2025-01-04 00:03:00', None])), '100C',
               simulation_start='2025-01-04 00:00:00')
    aware = FakeOut(pd.to_datetime(['2025-01-04 00:03:00Z', '2025-01-04 00:06:00Z']))
    with pytest.raises(ValueError, match='incompatible'):
        reader(aware, '100C', simulation_start='2025-01-04 00:00:00')
