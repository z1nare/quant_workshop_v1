"""Portfolio rules: forecasts -> target weights, with explicit constraints.

Rule (long-only, unleveraged):
  1. candidates = symbols whose forecast is finite and > threshold
  2. rank by forecast (descending); ties broken by symbol name (ascending) -> deterministic
  3. keep the top `max_positions`
  4. each selected asset gets  min(exposure_budget / max_positions, per_asset_cap)
  5. everything else stays in cash (weights are NOT renormalised)

Example: budget 0.9, cap 0.3, two qualifying assets -> 0.3 each, cash 0.4.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Mapping

import math
import numpy as np
import pandas as pd


@dataclass
class PortfolioConfig:
    threshold: float = 0.0002        # forecast must exceed this (decimal return per 5-min bar; 0.0002 = 2 bps)
    max_positions: int = 3
    exposure_budget: float = 0.9     # at most 90% of capital invested
    per_asset_cap: float = 0.3       # at most 30% in any one asset
    cost_bps: float = 2.0            # simplified proportional cost per trade side, basis points of traded value

    def validate(self) -> "PortfolioConfig":
        if not math.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        if int(self.max_positions) < 1 or int(self.max_positions) != self.max_positions:
            raise ValueError("max_positions must be a positive integer")
        for name in ("exposure_budget", "per_asset_cap"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be between 0 and 1, got {v}")
        if self.cost_bps < 0:
            raise ValueError("cost_bps cannot be negative")
        return self

    @property
    def slot_weight(self) -> float:
        return min(self.exposure_budget / self.max_positions, self.per_asset_cap)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> "PortfolioConfig":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d}).validate()


def allocate_weights(predictions: Mapping[str, float] | pd.Series, config: PortfolioConfig) -> pd.Series:
    """Reference allocation.  Returns a Series of weights indexed by symbol (all symbols present, 0 if unselected)."""
    config.validate()
    preds = pd.Series(predictions, dtype=float)
    if preds.index.has_duplicates:
        raise ValueError("duplicate symbols in predictions")
    finite = preds[np.isfinite(preds.to_numpy())]
    candidates = finite[finite > config.threshold]
    ranked = sorted(candidates.items(), key=lambda kv: (-kv[1], kv[0]))
    chosen = [sym for sym, _ in ranked[: int(config.max_positions)]]
    weights = pd.Series(0.0, index=preds.index, name="weight")
    weights[chosen] = config.slot_weight
    return weights


def check_weights(weights: pd.Series | Mapping[str, float], config: PortfolioConfig) -> Dict[str, object]:
    """Constraint report used by the notebook checkpoint and the engine's safety layer."""
    w = pd.Series(weights, dtype=float)
    arr = w.to_numpy()
    report = {
        "all_finite": bool(np.isfinite(arr).all()),
        "long_only": bool((arr >= -1e-12).all()),
        "within_per_asset_cap": bool((arr <= config.per_asset_cap + 1e-12).all()),
        "within_budget": bool(arr.sum() <= config.exposure_budget + 1e-12),
        "positions": int((arr > 1e-12).sum()),
        "within_max_positions": bool((arr > 1e-12).sum() <= config.max_positions),
        "invested": float(arr.sum()),
        "cash": float(1.0 - arr.sum()),
    }
    report["ok"] = all(report[k] for k in ("all_finite", "long_only", "within_per_asset_cap", "within_budget", "within_max_positions"))
    return report


def equal_weight(symbols: Iterable[str], config: PortfolioConfig) -> pd.Series:
    """Benchmark: the exposure budget spread evenly over all symbols (each capped), rest in cash."""
    symbols = list(symbols)
    w = min(config.exposure_budget / len(symbols), config.per_asset_cap)
    return pd.Series(w, index=symbols, name="weight")
