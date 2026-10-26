from datetime import date

import pandas as pd

from transaction_forecasting.data import heroes_days, simulate_series
from transaction_forecasting.models import MODELS


def test_heroes_day_is_second_monday():
    assert heroes_days(2025) == [date(2025, 8, 11), date(2025, 8, 12)]
    assert heroes_days(2024) == [date(2024, 8, 12), date(2024, 8, 13)]


def test_series_has_no_gaps():
    s = simulate_series("2024-01-01", "2024-12-31")
    assert len(s) == 366
    assert (s > 0).all()


def test_models_return_horizon_days():
    s = simulate_series("2023-01-01", "2024-06-30")
    for name, fn in MODELS.items():
        pred = fn(s, 14)
        assert len(pred) == 14, name
        assert pred.index[0] == s.index[-1] + pd.Timedelta(days=1), name
