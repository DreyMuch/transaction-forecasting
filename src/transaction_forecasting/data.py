"""Simulated daily transaction counts for a mobile wallet.

The series has the things that make this kind of data annoying to forecast:
  - steady growth in the user base
  - a weekly cycle (Fridays busy, Sundays quiet)
  - a payday surge from roughly the 25th to the 2nd
  - public holidays that behave like Sundays
  - one permanent level drop when a tariff change kicks in
"""

from datetime import date

import numpy as np
import pandas as pd

# Fixed-date Zimbabwean public holidays. Easter moves around and I've left it
# out for now - something to add if this gets used on real data.
FIXED_HOLIDAYS = [(1, 1), (2, 21), (4, 18), (5, 1), (5, 25), (12, 22), (12, 25), (12, 26)]

WEEKDAY_EFFECT = np.array([1.00, 0.97, 0.98, 1.03, 1.15, 1.08, 0.78])  # Mon..Sun


def heroes_days(year: int) -> list[date]:
    """Heroes Day is the 2nd Monday of August, Defence Forces Day the day after."""
    first = date(year, 8, 1)
    offset = (0 - first.weekday()) % 7
    monday = date(year, 8, 1 + offset + 7)
    return [monday, date(year, 8, monday.day + 1)]


def holiday_calendar(start: str, end: str) -> pd.DatetimeIndex:
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    days = [date(y, m, d) for y in years for m, d in FIXED_HOLIDAYS]
    days += [d for y in years for d in heroes_days(y)]
    return pd.DatetimeIndex(sorted(days))


def payday_window(idx: pd.DatetimeIndex) -> np.ndarray:
    return (idx.day >= 25) | (idx.day <= 2)


def simulate_series(start="2023-01-01", end="2025-12-31", seed=11) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="D")
    t = np.arange(len(idx))

    level = 180_000 * np.exp(0.00045 * t)
    tariff_change = idx >= "2024-09-01"
    level = level * np.where(tariff_change, 0.88, 1.0)

    weekly = WEEKDAY_EFFECT[idx.day_of_week]
    payday = np.where(payday_window(idx), 1.22, 1.0)
    holidays = np.where(idx.isin(holiday_calendar(start, end)), 0.80, 1.0)
    noise = np.exp(rng.normal(0, 0.035, len(idx)))

    y = level * weekly * payday * holidays * noise
    return pd.Series(y.round().astype(int), index=idx, name="transactions")
