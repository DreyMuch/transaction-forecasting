"""Rolling-origin backtest: cut the series at several points, forecast the
next 28 days from each cut, and score against what actually happened.

    uv run python -m transaction_forecasting.backtest
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import simulate_series
from .models import MODELS

HORIZON = 28
N_FOLDS = 8
REPORTS = Path("reports")


def mape(actual, pred):
    return float(np.mean(np.abs(actual - pred) / actual) * 100)


def backtest(series: pd.Series, horizon=HORIZON, n_folds=N_FOLDS):
    rows, last_fold = [], {}
    cutoffs = [len(series) - horizon * (k + 1) for k in reversed(range(n_folds))]
    for fold, cut in enumerate(cutoffs):
        history = series.iloc[:cut]
        actual = series.iloc[cut:cut + horizon]
        for name, fn in MODELS.items():
            pred = fn(history, horizon)
            rows.append({
                "fold": fold,
                "cutoff": str(history.index[-1].date()),
                "model": name,
                "mape": mape(actual.to_numpy(), pred.to_numpy()),
            })
            if fold == len(cutoffs) - 1:
                last_fold[name] = pred
    return pd.DataFrame(rows), last_fold


def main():
    series = simulate_series()
    results, last_fold = backtest(series)

    summary = results.groupby("model")["mape"].agg(["mean", "std", "max"]).round(2)
    summary = summary.sort_values("mean")
    print(summary)

    REPORTS.joinpath("figures").mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS / "backtest_results.csv", index=False)
    REPORTS.joinpath("metrics.json").write_text(
        json.dumps(summary.to_dict(orient="index"), indent=2) + "\n"
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    window = series.iloc[-HORIZON - 56:]
    ax.plot(window.index, window / 1000, color="black", lw=1.2, label="actual")
    for name, pred in last_fold.items():
        ax.plot(pred.index, pred / 1000, lw=1.2, label=name.replace("_", " "))
    ax.axvline(series.index[-HORIZON - 1], color="grey", ls=":", lw=1)
    ax.set_ylabel("transactions (thousands)")
    ax.set_title("Last backtest fold: 28 days ahead")
    ax.legend(ncol=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(REPORTS / "figures/last_fold_forecast.png", dpi=120)

    fig, ax = plt.subplots(figsize=(7, 4))
    pivot = results.pivot(index="cutoff", columns="model", values="mape")
    pivot.plot(ax=ax, marker="o")
    ax.set_ylabel("MAPE (%)")
    ax.set_xlabel("forecast origin")
    ax.set_title("Error by backtest fold")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(REPORTS / "figures/mape_by_fold.png", dpi=120)
    print("saved reports/")


if __name__ == "__main__":
    main()
