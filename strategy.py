"""strategy.py - the learner-facing interface between research and the trading engine.

The engine calls exactly two functions:

    predict_returns(history, model) -> pd.Series   one forecast per symbol (NaN = not ready)
    allocate(predictions, config)   -> pd.Series   target weight per symbol; cash = 1 - sum

In the workshop you write your own version of this file (Exercise 6) and export it in your
bundle.  This copy is the complete reference implementation so that the local application
works straight after cloning.  It depends only on the quantsoc package, never on notebook state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantsoc.features import FEATURE_NAMES, latest_features
from quantsoc.portfolio import PortfolioConfig


def predict_returns(history: pd.DataFrame, model) -> pd.Series:
    """history: long table of COMPLETED 5-minute bars; model: fitted pipeline from model.joblib."""
    latest = latest_features(history)                        # one row per symbol, FEATURE_NAMES order
    preds = pd.Series(np.nan, index=sorted(latest.index), dtype=float)
    ready = latest[FEATURE_NAMES].notna().all(axis=1)
    if ready.any():
        X = latest.loc[ready, FEATURE_NAMES].to_numpy(dtype=float)
        preds[latest.index[ready]] = model.predict(X)
    return preds


def allocate(predictions: pd.Series, config: PortfolioConfig) -> pd.Series:
    """Long-only, unleveraged: top `max_positions` forecasts above `threshold`, each min(budget/N, cap)."""
    preds = pd.Series(predictions, dtype=float)
    finite = preds[np.isfinite(preds.to_numpy())]
    candidates = finite[finite > config.threshold]
    ranked = sorted(candidates.items(), key=lambda kv: (-kv[1], kv[0]))   # best first, ties by name
    chosen = [symbol for symbol, _ in ranked[: config.max_positions]]
    slot = min(config.exposure_budget / config.max_positions, config.per_asset_cap)
    weights = pd.Series(0.0, index=preds.index)
    weights[chosen] = slot
    return weights
