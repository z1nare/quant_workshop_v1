"""Reference solutions for the six exercises.

They are shown to students on request (`ex.show_solution(n)`) and used by `ex.use_reference(n)`.
Nothing here is ever substituted silently: using a reference solution is recorded and labelled
in the exercise tracker and in the export manifest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES
from .portfolio import PortfolioConfig


# --- Exercise 1 -----------------------------------------------------------------
def compute_bar_returns(closes, sessions):
    """Return of each completed bar, within its session: close[t] / close[t-1] - 1."""
    return closes.groupby(sessions).pct_change(fill_method=None)


# --- Exercise 2 -----------------------------------------------------------------
def compute_momentum(closes, sessions, bars=3):
    """Return over the last `bars` completed bars, within a session: close[t] / close[t-bars] - 1."""
    return closes.groupby(sessions).pct_change(periods=bars, fill_method=None)


# --- Exercise 3 -----------------------------------------------------------------
def fit_linear_model(X_train, y_train):
    """Scale features using TRAINING statistics only, then fit ordinary least squares."""
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(X_train, y_train)
    return model


# --- Exercise 4 -----------------------------------------------------------------
def my_allocate(predictions, config):
    """Top-N forecasts above the threshold, each min(budget / N, cap); the rest stays in cash."""
    preds = pd.Series(predictions, dtype=float)
    finite = preds[np.isfinite(preds.to_numpy())]
    candidates = finite[finite > config.threshold]
    ranked = sorted(candidates.items(), key=lambda kv: (-kv[1], kv[0]))   # best first, ties by name
    chosen = [symbol for symbol, _ in ranked[: config.max_positions]]
    slot = min(config.exposure_budget / config.max_positions, config.per_asset_cap)
    weights = pd.Series(0.0, index=preds.index)
    weights[chosen] = slot
    return weights


# --- Exercise 5 -----------------------------------------------------------------
REFERENCE_CONFIG = PortfolioConfig(threshold=0.0003, max_positions=3, exposure_budget=0.9, per_asset_cap=0.3, cost_bps=2.0)
REFERENCE_JUSTIFICATION = ("Raising the threshold from 2 to 3 bps cut validation turnover noticeably while keeping "
                           "most of the gross return, so net-of-cost equity and drawdown both improved on validation.")


# --- Exercise 6 (strategy.py) ---------------------------------------------------
STRATEGY_PY = '''"""My strategy - the two functions the trading engine calls.

predict_returns(history, model) -> pd.Series of forecasts per symbol (NaN when features are not ready)
allocate(predictions, config)   -> pd.Series of target weights per symbol (cash = 1 - sum)

This file must not depend on notebook variables: everything it needs arrives as an argument
or is imported from the quantsoc package.
"""
import numpy as np
import pandas as pd

from quantsoc.features import FEATURE_NAMES, latest_features
from quantsoc.portfolio import PortfolioConfig


def predict_returns(history: pd.DataFrame, model) -> pd.Series:
    latest = latest_features(history)                        # one row per symbol, FEATURE_NAMES order
    preds = pd.Series(np.nan, index=sorted(latest.index), dtype=float)
    ready = latest[FEATURE_NAMES].notna().all(axis=1)
    if ready.any():
        X = latest.loc[ready, FEATURE_NAMES].to_numpy(dtype=float)
        preds[latest.index[ready]] = model.predict(X)
    return preds


def allocate(predictions: pd.Series, config: PortfolioConfig) -> pd.Series:
    preds = pd.Series(predictions, dtype=float)
    finite = preds[np.isfinite(preds.to_numpy())]
    candidates = finite[finite > config.threshold]
    ranked = sorted(candidates.items(), key=lambda kv: (-kv[1], kv[0]))
    chosen = [symbol for symbol, _ in ranked[: config.max_positions]]
    slot = min(config.exposure_budget / config.max_positions, config.per_asset_cap)
    weights = pd.Series(0.0, index=preds.index)
    weights[chosen] = slot
    return weights
'''
