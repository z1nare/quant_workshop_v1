"""Reference implementation of the learner-facing strategy interface.

This file has the SAME contract as the `strategy.py` students write in Exercise 6.
It depends only on the `quantsoc` package (never on notebook globals).

    predict_returns(history, model) -> pd.Series   forecast per symbol (NaN if features unavailable)
    allocate(predictions, config)   -> pd.Series   target weight per symbol (cash = 1 - sum)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantsoc.features import FEATURE_NAMES, latest_features
from quantsoc.portfolio import PortfolioConfig, allocate_weights


def predict_returns(history: pd.DataFrame, model) -> pd.Series:
    """Forecast the next tradable return for every symbol in `history`.

    history : long table of COMPLETED bars (timestamp, session, symbol, open, high, low, close, volume)
    model   : the fitted scikit-learn pipeline loaded from model.joblib
    """
    latest = latest_features(history)                       # one row per symbol, FEATURE_NAMES order
    preds = pd.Series(np.nan, index=sorted(latest.index), dtype=float)
    ready = latest[FEATURE_NAMES].notna().all(axis=1)
    if ready.any():
        X = latest.loc[ready, FEATURE_NAMES].to_numpy(dtype=float)
        preds[latest.index[ready]] = model.predict(X)
    return preds


def allocate(predictions: pd.Series, config: PortfolioConfig) -> pd.Series:
    """Long-only, unleveraged: top `max_positions` forecasts above `threshold`, each min(budget/max, cap)."""
    return allocate_weights(predictions, config)
