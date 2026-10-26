"""Three forecasters, all with the same signature: fit on history, return
`horizon` days of predictions."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .data import holiday_calendar, payday_window


def seasonal_naive(history: pd.Series, horizon: int) -> pd.Series:
    """Same weekday last week. Hard to beat more often than you'd think."""
    idx = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon)
    last_week = history.iloc[-7:].to_numpy()
    return pd.Series(np.resize(last_week, horizon), index=idx)


def holt_winters(history: pd.Series, horizon: int) -> pd.Series:
    model = ExponentialSmoothing(
        np.log(history), trend="add", damped_trend=True,
        seasonal="add", seasonal_periods=7,
    ).fit()
    return np.exp(model.forecast(horizon))


def calendar_features(idx: pd.DatetimeIndex, holidays: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t": (idx - pd.Timestamp("2023-01-01")).days,
            "dow": idx.day_of_week,
            "dom": idx.day,
            "month": idx.month,
            "payday": payday_window(idx).astype(int),
            "holiday": idx.isin(holidays).astype(int),
        },
        index=idx,
    )


def gradient_boosting(history: pd.Series, horizon: int) -> pd.Series:
    """Model log volume on calendar features, then detrend.

    Trees can't extrapolate a trend, so I fit a straight line to the last
    six months of log volume first and let the model learn what's left.
    """
    idx = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon)
    holidays = holiday_calendar(str(history.index[0].date()), str(idx[-1].date()))

    y = np.log(history)
    recent = y.iloc[-182:]
    slope, intercept = np.polyfit(np.arange(len(recent)), recent.to_numpy(), 1)
    t_hist = np.arange(len(y)) - (len(y) - len(recent))
    resid = y.to_numpy() - (intercept + slope * t_hist)

    X = calendar_features(history.index, holidays)
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                          max_leaf_nodes=15, random_state=0)
    # older residuals carry the pre-tariff level, weight them down
    weights = np.where(np.arange(len(y)) >= len(y) - 365, 1.0, 0.3)
    model.fit(X, resid, sample_weight=weights)

    t_future = np.arange(len(recent), len(recent) + horizon)
    pred = model.predict(calendar_features(idx, holidays)) + intercept + slope * t_future
    return pd.Series(np.exp(pred), index=idx)


MODELS = {
    "seasonal_naive": seasonal_naive,
    "holt_winters": holt_winters,
    "gradient_boosting": gradient_boosting,
}
