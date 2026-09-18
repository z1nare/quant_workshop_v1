"""The ONE feature implementation shared by research, the exported strategy and the runner.

Timing contract (see docs/timing_contract.md):

* A bar labelled T covers [T, T+5min).  Its close is known at T+5min.
* Features for bar t use only closes of bars <= t inside the SAME session.
* The first bars of a session are warm-up: features are NaN until enough
  completed bars exist (12 for vol_12).  Nothing is invented for the overnight gap.

FEATURE_NAMES fixes the order used everywhere; the exported bundle records it.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

FEATURE_NAMES: List[str] = ["ret_1", "mom_3", "vol_12"]
WARMUP_BARS = 12            # completed bars needed before every feature is defined
MIN_HISTORY_BARS = WARMUP_BARS + 1
TIMEFRAME = "5Min"


def _position_in_session(sessions: pd.Series) -> np.ndarray:
    """0-based bar number inside its session, for a time-ordered session label Series."""
    s = np.asarray(sessions)
    new = np.r_[True, s[1:] != s[:-1]]
    idx = np.arange(len(s))
    starts = np.maximum.accumulate(np.where(new, idx, 0))
    return idx - starts


def _mask_warmup(frame: pd.DataFrame, sessions: pd.Series, needed: int) -> pd.DataFrame:
    """Set rows whose window would reach back across the session boundary to NaN."""
    pos = _position_in_session(sessions.reindex(frame.index))
    out = frame.copy()
    out.iloc[pos < needed] = np.nan
    return out


def bar_returns(closes: pd.DataFrame, sessions: pd.Series) -> pd.DataFrame:
    """Simple return of each completed bar: close[t] / close[t-1] - 1, within a session.

    `closes` is timestamp x symbol, `sessions` is a Series (index = timestamp) of session labels.
    The first bar of every session has no previous bar in that session and is NaN.
    (Equivalent to closes.groupby(sessions).pct_change(), computed faster by masking.)
    """
    return _mask_warmup(closes.pct_change(fill_method=None), sessions, 1)


def momentum(closes: pd.DataFrame, sessions: pd.Series, bars: int = 3) -> pd.DataFrame:
    """Return over the last `bars` completed bars: close[t] / close[t-bars] - 1, within a session."""
    return _mask_warmup(closes.pct_change(periods=bars, fill_method=None), sessions, bars)


def rolling_volatility(returns: pd.DataFrame, sessions: pd.Series, window: int = 12) -> pd.DataFrame:
    """Standard deviation of the last `window` bar returns, within a session (NaN during warm-up).

    The window must contain `window` returns of the same session, i.e. the bar must be at least
    position `window` in its session (position 0 has no return).
    """
    return _mask_warmup(returns.rolling(window, min_periods=window).std(), sessions, window)


def _wide_features(bars: pd.DataFrame):
    needed = {"timestamp", "session", "symbol", "close"}
    if not needed.issubset(bars.columns):
        raise ValueError(f"bars must contain {sorted(needed)}")
    closes = bars.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    sessions = bars.drop_duplicates("timestamp").set_index("timestamp")["session"].sort_index().reindex(closes.index)
    ret_1 = bar_returns(closes, sessions)
    return closes, sessions, {"ret_1": ret_1, "mom_3": momentum(closes, sessions, 3),
                              "vol_12": rolling_volatility(ret_1, sessions, 12)}


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Long feature table with columns [timestamp, session, symbol] + FEATURE_NAMES.

    Only completed bars in `bars` are used, and only their closes.  The value in row
    (t, symbol) depends on closes at times <= t of the same session (tests/test_features.py
    checks that changing future bars cannot change earlier features).
    """
    closes, sessions, wide = _wide_features(bars)
    n, k = closes.shape
    table = pd.DataFrame({
        "timestamp": closes.index.repeat(k),
        "session": np.repeat(sessions.to_numpy(), k),
        "symbol": np.tile(np.asarray(closes.columns, dtype=object), n),
        **{name: wide[name].to_numpy().ravel() for name in FEATURE_NAMES},
    })
    return table          # already sorted by (timestamp, symbol) because the pivot sorts both axes


def latest_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Feature row for the most recent completed bar of each symbol (index = symbol).

    A symbol that is missing the latest bar gets NaN features (nothing is invented for it).
    """
    closes, sessions, wide = _wide_features(bars)
    last = pd.DataFrame({name: wide[name].iloc[-1] for name in FEATURE_NAMES})
    last.insert(0, "timestamp", closes.index[-1])
    last.index.name = "symbol"
    return last.sort_index()


def feature_matrix(table: pd.DataFrame) -> np.ndarray:
    """Numeric matrix in FEATURE_NAMES order (the only order the model ever sees)."""
    return table[FEATURE_NAMES].to_numpy(dtype=float)
