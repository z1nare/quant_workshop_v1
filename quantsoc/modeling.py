"""Targets, chronological splits and the linear model used in the workshop.

Target (what the strategy can actually earn):
    target[t] = open[t+2] / open[t+1] - 1
A decision made after bar t completes is executed at the first tradable price,
the open of bar t+1, and held until the next decision executes at open[t+2].
Examples whose t+1 or t+2 falls in another session or another split are dropped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, build_features

SPLIT_NAMES = ["train", "validation", "test"]
DEFAULT_FRACTIONS = (0.6, 0.2, 0.2)
TARGET_NAME = "target"
EXEC_LAG = 1          # bars between decision and fill  (fill at open[t+1])
HOLD_BARS = 1         # bars a fill is held             (until open[t+2])


def tradable_target(opens: pd.DataFrame, sessions: pd.Series) -> pd.DataFrame:
    """open[t+2] / open[t+1] - 1 per symbol, NaN when the window leaves the session."""
    nxt = opens.groupby(sessions).shift(-EXEC_LAG)
    after = opens.groupby(sessions).shift(-(EXEC_LAG + HOLD_BARS))
    return after / nxt - 1


def split_boundaries(timestamps: pd.Index, fractions=DEFAULT_FRACTIONS) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    """Chronological [start, end] timestamp ranges for train / validation / test.

    Boundaries are the same for every asset because they are defined on the shared timestamp index.
    """
    ts = pd.DatetimeIndex(sorted(pd.unique(timestamps)))
    n = len(ts)
    n_train = int(n * fractions[0])
    n_val = int(n * fractions[1])
    edges = {
        "train": (ts[0], ts[n_train - 1]),
        "validation": (ts[n_train], ts[n_train + n_val - 1]),
        "test": (ts[n_train + n_val], ts[-1]),
    }
    return edges


def assign_split(timestamps: pd.Series, boundaries: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]) -> pd.Series:
    out = pd.Series(pd.NA, index=timestamps.index, dtype="object")
    for name, (start, end) in boundaries.items():
        mask = (timestamps >= start) & (timestamps <= end)
        out[mask] = name
    return out


def build_dataset(bars: pd.DataFrame, fractions=DEFAULT_FRACTIONS) -> pd.DataFrame:
    """Features + target + split label for every (timestamp, symbol) row.

    Rows keep NaN features/targets so that plots can show warm-up; use `clean_rows` for modelling.
    A row is *usable* for split S only if the decision bar t AND the bars t+1, t+2 that define
    the target all belong to S ("targets must not cross split boundaries").
    """
    table = build_features(bars)
    opens = bars.pivot(index="timestamp", columns="symbol", values="open").sort_index()
    sessions = bars.drop_duplicates("timestamp").set_index("timestamp")["session"].sort_index()
    target = tradable_target(opens, sessions).stack(future_stack=True)
    target.index.names = ["timestamp", "symbol"]
    table = table.merge(target.rename(TARGET_NAME).reset_index(), on=["timestamp", "symbol"], how="left")

    bounds = split_boundaries(table["timestamp"], fractions)
    table["split"] = assign_split(table["timestamp"], bounds)
    # split of the bars that define the target (t+1 and t+2)
    ts = pd.DatetimeIndex(sorted(table["timestamp"].unique()))
    pos = pd.Series(np.arange(len(ts)), index=ts)
    idx = table["timestamp"].map(pos).to_numpy()
    split_at = assign_split(pd.Series(ts), bounds).to_numpy()
    def split_of(offset):
        j = idx + offset
        ok = j < len(ts)
        res = np.full(len(idx), None, dtype=object)
        res[ok] = split_at[j[ok]]
        return res
    same = (split_of(EXEC_LAG) == table["split"].to_numpy()) & (split_of(EXEC_LAG + HOLD_BARS) == table["split"].to_numpy())
    table["crosses_split"] = ~same
    return table


def clean_rows(dataset: pd.DataFrame, split: str) -> pd.DataFrame:
    """Rows of one split with complete features and a target that stays inside the split."""
    d = dataset[dataset["split"] == split]
    d = d[~d["crosses_split"]]
    return d.dropna(subset=FEATURE_NAMES + [TARGET_NAME]).reset_index(drop=True)


def xy(rows: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    return rows[FEATURE_NAMES].to_numpy(dtype=float), rows[TARGET_NAME].to_numpy(dtype=float)


def make_model() -> Pipeline:
    """StandardScaler (fit on training data only) followed by ordinary least squares."""
    return make_pipeline(StandardScaler(), LinearRegression())


def fit_model(train_rows: pd.DataFrame) -> Pipeline:
    X, y = xy(train_rows)
    return make_model().fit(X, y)


@dataclass
class ForecastScore:
    mse: float
    baseline_mse: float
    correlation: float
    hit_rate: float
    n: int

    @property
    def mse_improvement_pct(self) -> float:
        return 100.0 * (1 - self.mse / self.baseline_mse) if self.baseline_mse > 0 else float("nan")

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "metric": ["mean squared error (model)", "mean squared error (zero baseline)",
                       "MSE improvement vs baseline", "correlation(forecast, realised)", "hit rate (sign correct)", "examples"],
            "value": [f"{self.mse:.3e}", f"{self.baseline_mse:.3e}", f"{self.mse_improvement_pct:.2f} %",
                      f"{self.correlation:.3f}", f"{100*self.hit_rate:.1f} %", f"{self.n}"],
        })


def score_forecasts(y_true: np.ndarray, y_pred: np.ndarray) -> ForecastScore:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mse = float(np.mean((y_true - y_pred) ** 2))
    base = float(np.mean(y_true ** 2))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 0 else 0.0
    hit = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    return ForecastScore(mse=mse, baseline_mse=base, correlation=corr, hit_rate=hit, n=len(y_true))


def predictions_wide(model, dataset: pd.DataFrame, split: str) -> pd.DataFrame:
    """Forecast for every (timestamp, symbol) of a split as a timestamp x symbol table.

    NaN where the features are not available (warm-up).  Rows whose target crosses the split
    boundary still get a forecast: the *backtest* uses actual prices, only model *training*
    and *scoring* exclude them.
    """
    d = dataset[dataset["split"] == split]
    ok = d[FEATURE_NAMES].notna().all(axis=1)
    pred = pd.Series(np.nan, index=d.index)
    if ok.any():
        pred[ok] = model.predict(d.loc[ok, FEATURE_NAMES].to_numpy(dtype=float))
    out = d.assign(pred=pred).pivot(index="timestamp", columns="symbol", values="pred").sort_index()
    out.columns.name = None
    return out


def coefficient_table(model: Pipeline) -> pd.DataFrame:
    lin = model.named_steps["linearregression"]
    return pd.DataFrame({"feature": FEATURE_NAMES, "coefficient (per 1 std of feature)": lin.coef_}).set_index("feature")
