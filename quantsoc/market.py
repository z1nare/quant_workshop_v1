"""The controlled synthetic teaching market.

Five fictional assets, regular 5-minute sessions (09:30-16:00 New York time,
78 bars per session), no overnight bars.  The generating mechanism is revealed
to students in the final workshop section; see docs/timing_contract.md and
data/DATA_CARD.md for the exact design.

Mechanism (per asset i, bar t, within a session):

    close-to-close log return
        r[i,t] = drift[i]
                 + beta * clip(m[i,t-1], -cap_i, cap_i)        # planted momentum
                 + sigma[i] * (rho * F[t] + sqrt(1-rho^2) * e[i,t])   # common + own shock

    m[i,t-1] = sum of the previous three close-to-close log returns in the SAME session
    F[t]     = shared market factor (one draw per bar, common to all assets)
    e[i,t]   = asset-specific noise

    open[t]  = close[t-1] * exp(gap[i,t])      # small gap: the next tradable price is NOT the close
    close[t] = open[t] * exp(r[i,t] - gap[i,t]) # so that close[t]/close[t-1] = exp(r[i,t])

    The first bar of a session opens at the previous session's close times an
    overnight gap (larger, no momentum carry).  High/low wrap open/close.

Momentum is a property of *closes*; the tradable target is open[t+2]/open[t+1]-1,
which contains r[i,t+1] and therefore stays learnable (see tests/test_targets_splits.py).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SYMBOLS: List[str] = ["AURA", "BOLT", "CRUX", "DUNE", "ECHO"]
ASSET_NAMES: Dict[str, str] = {
    "AURA": "Aura Robotics (fictional)",
    "BOLT": "Bolt Energy (fictional)",
    "CRUX": "Crux Semiconductors (fictional)",
    "DUNE": "Dune Minerals (fictional)",
    "ECHO": "Echo Media (fictional)",
}
MARKET_TZ = "America/New_York"
BAR_MINUTES = 5
BARS_PER_SESSION = 78          # 09:30 .. 15:55 bar starts (16:00 close)
SESSION_OPEN = "09:30"

BAR_COLUMNS = ["timestamp", "session", "symbol", "open", "high", "low", "close", "volume"]


@dataclass
class MarketParams:
    """Everything that determines a synthetic market.  Frozen values are checked in."""

    seed: int = 20250917
    n_sessions: int = 39                       # 39 * 78 = 3042 bars per asset
    start_date: str = "2025-03-03"             # fictional Monday; weekdays only, no holidays
    symbols: List[str] = field(default_factory=lambda: list(SYMBOLS))
    sigma: List[float] = field(default_factory=lambda: [0.0009, 0.0012, 0.0015, 0.0020, 0.0028])
    drift: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    beta: float = 0.07                         # strength of the planted momentum effect (calibrated, see docs)
    momentum_bars: int = 3
    rho: float = 0.55                          # loading on the shared market factor
    gap_frac: float = 0.15                     # intra-session open gap as a fraction of sigma
    overnight_gap_mult: float = 3.0            # overnight gap std as a multiple of sigma
    wick_frac: float = 0.5                     # high/low extension as a fraction of sigma
    start_prices: List[float] = field(default_factory=lambda: [42.0, 118.0, 75.0, 23.5, 310.0])
    base_volume: List[float] = field(default_factory=lambda: [12000, 8000, 15000, 30000, 4000])
    label: str = "workshop"

    def to_dict(self) -> dict:
        return asdict(self)


STRESS_OVERRIDES = dict(
    seed=20250918,
    n_sessions=10,
    start_date="2025-04-28",
    sigma=[0.0018, 0.0024, 0.0030, 0.0040, 0.0056],   # doubled volatility
    beta=-0.05,                                          # reversed, weaker predictability
    label="stress",
)


def session_timestamps(start_date: str, n_sessions: int) -> pd.DatetimeIndex:
    """Bar START times for n consecutive weekday sessions.  No overnight bars."""
    days = pd.bdate_range(start=start_date, periods=n_sessions, freq="B")
    intraday = pd.to_timedelta(np.arange(BARS_PER_SESSION) * BAR_MINUTES, unit="m")
    open_offset = pd.to_timedelta(9 * 60 + 30, unit="m")
    stamps = [d + open_offset + off for d in days for off in intraday]
    return pd.DatetimeIndex(stamps).tz_localize(MARKET_TZ)


def _clip(x: np.ndarray, cap: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, -cap), cap)


def generate_market(params: Optional[MarketParams] = None) -> pd.DataFrame:
    """Generate the synthetic market as a long DataFrame with BAR_COLUMNS.

    Deterministic for a given MarketParams (numpy Generator with a fixed seed).
    """
    p = params or MarketParams()
    rng = np.random.default_rng(p.seed)
    n_assets = len(p.symbols)
    stamps = session_timestamps(p.start_date, p.n_sessions)
    n_bars = len(stamps)
    sessions = stamps.tz_convert(MARKET_TZ).strftime("%Y-%m-%d")
    sigma = np.asarray(p.sigma, dtype=float)
    drift = np.asarray(p.drift, dtype=float)
    cap = 3.0 * np.sqrt(p.momentum_bars) * sigma        # bound for the momentum feedback

    market_factor = rng.standard_normal(n_bars)
    own_noise = rng.standard_normal((n_bars, n_assets))
    gap_noise = rng.standard_normal((n_bars, n_assets))
    overnight_common = rng.standard_normal(n_bars)
    wick_hi = np.abs(rng.standard_normal((n_bars, n_assets)))
    wick_lo = np.abs(rng.standard_normal((n_bars, n_assets)))
    vol_noise = rng.standard_normal((n_bars, n_assets))

    log_close = np.empty((n_bars, n_assets))
    log_open = np.empty((n_bars, n_assets))
    r_hist: List[np.ndarray] = []                      # close-to-close log returns in this session
    prev_log_close = np.log(np.asarray(p.start_prices, dtype=float))

    for t in range(n_bars):
        first_bar = (t % BARS_PER_SESSION == 0)
        if first_bar:
            r_hist = []                                # momentum does not cross the session boundary
            gap = p.overnight_gap_mult * sigma * (0.6 * overnight_common[t] + 0.8 * gap_noise[t])
        else:
            gap = p.gap_frac * sigma * gap_noise[t]

        if len(r_hist) == 0:
            momentum = np.zeros(n_assets)
        else:
            momentum = np.sum(r_hist[-p.momentum_bars:], axis=0)
        shock = sigma * (p.rho * market_factor[t] + np.sqrt(1 - p.rho**2) * own_noise[t])
        r = drift + p.beta * _clip(momentum, cap) + shock
        if first_bar:
            r = r + gap                                 # the overnight move is part of the first bar's return
            log_open[t] = prev_log_close + gap
        else:
            log_open[t] = prev_log_close + gap          # small gap: next tradable price != last close
        log_close[t] = prev_log_close + r
        r_hist.append(r)
        prev_log_close = log_close[t]

    open_ = np.exp(log_open)
    close = np.exp(log_close)
    hi = np.maximum(open_, close) * np.exp(p.wick_frac * sigma * wick_hi)
    lo = np.minimum(open_, close) * np.exp(-p.wick_frac * sigma * wick_lo)

    # U-shaped intraday volume profile with lognormal noise
    bar_in_session = np.arange(n_bars) % BARS_PER_SESSION
    u_shape = 1.0 + 1.5 * ((bar_in_session - BARS_PER_SESSION / 2) / (BARS_PER_SESSION / 2)) ** 2
    volume = (np.asarray(p.base_volume)[None, :] * u_shape[:, None] * np.exp(0.35 * vol_noise)).round()

    frames = []
    for j, sym in enumerate(p.symbols):
        frames.append(pd.DataFrame({
            "timestamp": stamps,
            "session": sessions,
            "symbol": sym,
            "open": open_[:, j],
            "high": hi[:, j],
            "low": lo[:, j],
            "close": close[:, j],
            "volume": volume[:, j].astype("int64"),
        }))
    bars = pd.concat(frames, ignore_index=True)
    bars = bars.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return bars[BAR_COLUMNS]


def stress_params(base: Optional[MarketParams] = None) -> MarketParams:
    p = base or MarketParams()
    d = p.to_dict()
    d.update(STRESS_OVERRIDES)
    return MarketParams(**d)


# ----------------------------------------------------------------------------- I/O

def save_market(bars: pd.DataFrame, path) -> None:
    out = bars.copy()
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    out[["open", "high", "low", "close"]] = out[["open", "high", "low", "close"]].round(6)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def load_market(path) -> pd.DataFrame:
    """Load a bars CSV and restore tz-aware timestamps (New York time)."""
    bars = pd.read_csv(path, dtype={"session": str})
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True).dt.tz_convert(MARKET_TZ)
    bars = bars.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return validate_bars(bars)


def validate_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Raise if the bar table breaks the basic contract (positive prices, valid OHLC, no duplicates)."""
    missing = [c for c in BAR_COLUMNS if c not in bars.columns]
    if missing:
        raise ValueError(f"bars table is missing columns: {missing}")
    px = bars[["open", "high", "low", "close"]]
    if not np.isfinite(px.to_numpy()).all() or (px.to_numpy() <= 0).any():
        raise ValueError("prices must be finite and strictly positive")
    if (bars["high"] < bars[["open", "close"]].max(axis=1) - 1e-9).any():
        raise ValueError("high must be >= max(open, close)")
    if (bars["low"] > bars[["open", "close"]].min(axis=1) + 1e-9).any():
        raise ValueError("low must be <= min(open, close)")
    if bars.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("duplicate (timestamp, symbol) rows")
    return bars


def to_wide(bars: pd.DataFrame, column: str = "close") -> pd.DataFrame:
    """Pivot a long bars table into a timestamp x symbol table for one price column."""
    wide = bars.pivot(index="timestamp", columns="symbol", values=column).sort_index()
    wide.columns.name = None
    return wide


def session_series(bars: pd.DataFrame) -> pd.Series:
    """Session label for each timestamp (index = timestamp)."""
    s = bars.drop_duplicates("timestamp").set_index("timestamp")["session"].sort_index()
    s.name = "session"
    return s


def describe_market(bars: pd.DataFrame) -> pd.DataFrame:
    """Small summary table for the notebook's data inspection cell."""
    g = bars.groupby("symbol")
    out = pd.DataFrame({
        "bars": g.size(),
        "sessions": g["session"].nunique(),
        "first_close": g["close"].first().round(2),
        "last_close": g["close"].last().round(2),
        "min_close": g["close"].min().round(2),
        "max_close": g["close"].max().round(2),
        "avg_volume": g["volume"].mean().round(0),
    })
    out.index.name = "symbol"
    return out
