"""Simulation engine shared by the evaluation backtest AND the accelerated replay.

Accounting model (deliberately simple, stated in the notebook):

* Decisions are made after bar t completes; fills happen at open[t+1] (next tradable price).
* Holdings are shares ("units"); between fills their value drifts with prices, so the
  trade needed at a fill is computed against the DRIFTED current weights, not the previous target.
* Cost = cost_bps / 10,000 * traded value, per side, deducted from cash at the fill.
* Target dollar amounts are sized on portfolio value net of the estimated cost of the trade
  itself, so cash can never go negative when the exposure budget is <= 100%.
* A target row that is entirely NaN means "no decision" (e.g. warm-up): the account holds.

The same `SimAccount.rebalance` is called by `run_backtest` (vectorised forecasts) and by
`run_replay` (bar-by-bar, progressive history), so the two cannot drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .portfolio import PortfolioConfig, allocate_weights, equal_weight

INITIAL_CASH = 100_000.0


@dataclass
class TradeRecord:
    timestamp: pd.Timestamp
    value_before: float
    trades: Dict[str, float]          # signed dollar amounts (+ buy, - sell) per symbol
    cost: float
    turnover: float                   # sum |trade| / value_before
    value_after: float


class SimAccount:
    """Cash + share holdings, marked at whatever prices you pass in.  Internals are numpy for speed."""

    def __init__(self, symbols: List[str], cash: float = INITIAL_CASH):
        self.symbols = list(symbols)
        self.cash = float(cash)
        self._units = np.zeros(len(self.symbols))

    @property
    def units(self) -> pd.Series:
        return pd.Series(self._units, index=self.symbols)

    def _px(self, prices) -> np.ndarray:
        if isinstance(prices, pd.Series):
            prices = prices.reindex(self.symbols)
        return np.asarray(prices, dtype=float)

    def position_values(self, prices: pd.Series) -> pd.Series:
        return pd.Series(self._units * self._px(prices), index=self.symbols)

    def value(self, prices: pd.Series) -> float:
        return float(self.cash + (self._units * self._px(prices)).sum())

    def weights(self, prices: pd.Series) -> pd.Series:
        pv = self._units * self._px(prices)
        v = self.cash + pv.sum()
        return pd.Series(pv / v if v > 0 else np.zeros_like(pv), index=self.symbols)

    def rebalance(self, target_weights: pd.Series, prices: pd.Series, cost_rate: float,
                  timestamp=None) -> TradeRecord:
        """Trade from the current (drifted) holdings to `target_weights` at `prices`."""
        px = self._px(prices)
        if not np.isfinite(px).all() or (px <= 0).any():
            raise ValueError("rebalance needs finite positive prices for every symbol")
        if isinstance(target_weights, pd.Series):
            target = target_weights.reindex(self.symbols).fillna(0.0).to_numpy(dtype=float)
        else:
            target = np.nan_to_num(np.asarray(target_weights, dtype=float))
        if (target < -1e-12).any() or target.sum() > 1 + 1e-9:
            raise ValueError("target weights must be long-only and sum to at most 1")
        current_value = self._units * px
        value_before = float(self.cash + current_value.sum())

        # size targets on value net of the trade's own cost (a few fixed-point passes converge exactly)
        investable = value_before
        for _ in range(4):
            trades = target * investable - current_value
            cost = float(np.abs(trades).sum() * cost_rate)
            investable = value_before - cost
        trades = target * investable - current_value
        cost = float(np.abs(trades).sum() * cost_rate)

        self._units = self._units + trades / px
        self._units[np.abs(self._units) < 1e-12] = 0.0
        self.cash = self.cash - float(trades.sum()) - cost
        if self.cash < -1e-6 * max(value_before, 1.0):
            raise AssertionError(f"cash went negative ({self.cash:.4f}); this should be impossible with budget <= 1")
        self.cash = max(self.cash, 0.0)
        turnover = float(np.abs(trades).sum() / value_before) if value_before > 0 else 0.0
        return TradeRecord(timestamp=timestamp, value_before=value_before, trades=dict(zip(self.symbols, trades.tolist())),
                           cost=cost, turnover=turnover, value_after=self.value(px))


@dataclass
class BacktestResult:
    equity: pd.Series                 # portfolio value at each fill time (after trading)
    cash: pd.Series
    weights: pd.DataFrame             # realised weights right after each fill
    turnover: pd.Series               # per fill, fraction of portfolio value traded
    costs: pd.Series                  # dollars paid per fill
    trades: pd.DataFrame              # signed dollar trades per symbol per fill
    held: pd.Series                   # True where no decision was applied (warm-up / hold)
    config: PortfolioConfig
    label: str = "strategy"

    @property
    def drawdown(self) -> pd.Series:
        return self.equity / self.equity.cummax() - 1.0

    @property
    def total_return(self) -> float:
        return float(self.equity.iloc[-1] / self.equity.iloc[0] - 1.0)

    @property
    def max_drawdown(self) -> float:
        return float(self.drawdown.min())

    @property
    def total_costs(self) -> float:
        return float(self.costs.sum())

    @property
    def avg_turnover(self) -> float:
        return float(self.turnover.mean())

    def summary(self) -> pd.Series:
        return pd.Series({
            "final value ($)": round(float(self.equity.iloc[-1]), 2),
            "total return (%)": round(100 * self.total_return, 3),
            "max drawdown (%)": round(100 * self.max_drawdown, 3),
            "total costs ($)": round(self.total_costs, 2),
            "avg turnover per bar (%)": round(100 * self.avg_turnover, 2),
            "bars": int(len(self.equity)),
        }, name=self.label)


def run_backtest(opens: pd.DataFrame, target_weights: pd.DataFrame, config: PortfolioConfig,
                 initial_cash: float = INITIAL_CASH, label: str = "strategy",
                 cost_bps: Optional[float] = None) -> BacktestResult:
    """Apply a table of decisions to a table of tradable prices.

    opens          : timestamp x symbol prices at which fills happen (bar opens)
    target_weights : timestamp x symbol target weights, indexed by DECISION bar t.
                     Applied at the next row of `opens` (open[t+1]).  All-NaN row = hold.
    """
    config.validate()
    cost_rate = (config.cost_bps if cost_bps is None else cost_bps) / 10_000.0
    symbols = list(opens.columns)
    opens = opens.sort_index()
    idx = opens.index
    acct = SimAccount(symbols, initial_cash)
    tw = target_weights.reindex(index=idx[:-1], columns=symbols)      # aligned to decision bars
    tw_arr = tw.to_numpy(dtype=float)
    has_decision = ~np.isnan(tw_arr).all(axis=1)
    px_arr = opens.to_numpy(dtype=float)
    n, k = len(idx), len(symbols)

    equity = np.empty(n); cash = np.empty(n); weights = np.zeros((n, k))
    turnover = np.zeros(n); costs = np.zeros(n); held = np.ones(n, dtype=bool); trades = np.zeros((n, k))
    equity[0] = cash[0] = initial_cash
    for i in range(1, n):
        prices = px_arr[i]
        if has_decision[i - 1]:
            rec = acct.rebalance(np.nan_to_num(tw_arr[i - 1]), prices, cost_rate, timestamp=idx[i])
            turnover[i], costs[i], held[i] = rec.turnover, rec.cost, False
            trades[i] = list(rec.trades.values())
        equity[i] = acct.value(prices)
        cash[i] = acct.cash
        weights[i] = acct.weights(prices).to_numpy()
    return BacktestResult(
        equity=pd.Series(equity, index=idx, name="equity"),
        cash=pd.Series(cash, index=idx, name="cash"),
        weights=pd.DataFrame(weights, index=idx, columns=symbols),
        turnover=pd.Series(turnover, index=idx, name="turnover"),
        costs=pd.Series(costs, index=idx, name="costs"),
        trades=pd.DataFrame(trades, index=idx, columns=symbols),
        held=pd.Series(held, index=idx, name="held"),
        config=config, label=label,
    )


def decisions_from_forecasts(forecasts: pd.DataFrame, config: PortfolioConfig,
                             allocate: Callable = None) -> pd.DataFrame:
    """Turn a timestamp x symbol forecast table into target weights using `allocate` (default: reference rule).

    Rows where every forecast is NaN become all-NaN rows = 'no decision, hold' (warm-up).
    """
    allocate = allocate or (lambda preds, cfg: allocate_weights(preds, cfg))
    cols = list(forecasts.columns)
    arr = forecasts.to_numpy(dtype=float)
    out = np.full(arr.shape, np.nan)
    for i in range(len(arr)):
        if not np.isnan(arr[i]).all():
            out[i] = pd.Series(allocate(pd.Series(arr[i], index=cols), config), dtype=float).reindex(cols).to_numpy()
    return pd.DataFrame(out, index=forecasts.index, columns=cols)


def equal_weight_decisions(index: pd.Index, symbols: List[str], config: PortfolioConfig,
                           hold_mask: Optional[pd.Series] = None) -> pd.DataFrame:
    """Equal-weight benchmark decisions with the same timing (and the same warm-up holds if given)."""
    w = equal_weight(symbols, config)
    out = pd.DataFrame([w.to_numpy()] * len(index), index=index, columns=symbols)
    if hold_mask is not None:
        out.loc[hold_mask.reindex(index).fillna(False).astype(bool)] = np.nan
    return out


def evaluate_strategy(opens: pd.DataFrame, forecasts: pd.DataFrame, config: PortfolioConfig,
                      allocate: Callable = None, label: str = "strategy") -> Dict[str, BacktestResult]:
    """Net, gross and equal-weight-benchmark backtests with identical timing and prices."""
    decisions = decisions_from_forecasts(forecasts, config, allocate)
    hold_mask = decisions.isna().all(axis=1)
    bench = equal_weight_decisions(decisions.index, list(opens.columns), config, hold_mask)
    return {
        "net": run_backtest(opens, decisions, config, label=f"{label} (net of costs)"),
        "gross": run_backtest(opens, decisions, config, label=f"{label} (gross, no costs)", cost_bps=0.0),
        "benchmark": run_backtest(opens, bench, config, label="equal weight (net of costs)"),
    }


def compare_table(results: Dict[str, BacktestResult]) -> pd.DataFrame:
    return pd.DataFrame({k: v.summary() for k, v in results.items()})


# --------------------------------------------------------------------------- replay

@dataclass
class ReplayStep:
    step: int
    decision_time: pd.Timestamp
    fill_time: Optional[pd.Timestamp]
    predictions: Dict[str, float]
    target_weights: Optional[Dict[str, float]]
    actual_weights: Dict[str, float]
    cash: float
    value: float
    orders: Dict[str, float]           # signed dollars actually traded at the fill
    cost: float
    message: str


@dataclass
class ReplayTrace:
    steps: List[ReplayStep]
    symbols: List[str]
    config: PortfolioConfig

    @property
    def equity(self) -> pd.Series:
        s = pd.Series({st.fill_time: st.value for st in self.steps if st.fill_time is not None}, name="equity")
        return s.sort_index()

    @property
    def drawdown(self) -> pd.Series:
        e = self.equity
        return e / e.cummax() - 1

    def frame(self) -> pd.DataFrame:
        rows = []
        for st in self.steps:
            rows.append({"decision_time": st.decision_time, "fill_time": st.fill_time, "value": st.value,
                         "cash": st.cash, "cost": st.cost, "message": st.message,
                         **{f"pred_{s}": st.predictions.get(s, np.nan) for s in self.symbols},
                         **{f"w_{s}": st.actual_weights.get(s, 0.0) for s in self.symbols}})
        return pd.DataFrame(rows)


def run_replay(bars: pd.DataFrame, predict_returns: Callable, allocate: Callable, model, config: PortfolioConfig,
               start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None,
               lookback_bars: int = 40, initial_cash: float = INITIAL_CASH,
               on_step: Optional[Callable[[ReplayStep], None]] = None) -> ReplayTrace:
    """Bar-by-bar replay that reveals history progressively to the strategy.

    At each decision bar t the strategy receives ONLY bars with timestamp <= t (the last
    `lookback_bars` of them), exactly like the live runner.  Fills happen at open[t+1]
    through the same SimAccount used by run_backtest.
    """
    config.validate()
    cost_rate = config.cost_bps / 10_000.0
    symbols = sorted(bars["symbol"].unique())
    opens = bars.pivot(index="timestamp", columns="symbol", values="open").sort_index()
    stamps = opens.index
    start = stamps[0] if start is None else pd.Timestamp(start)
    end = stamps[-1] if end is None else pd.Timestamp(end)
    positions = np.arange(len(stamps))
    sel = positions[(stamps >= start) & (stamps <= end)]
    acct = SimAccount(symbols, initial_cash)
    steps: List[ReplayStep] = []
    bars_sorted = bars.sort_values(["timestamp", "symbol"])
    for k, i in enumerate(sel):
        t = stamps[i]
        lo = stamps[max(0, i - lookback_bars + 1)]
        history = bars_sorted[(bars_sorted["timestamp"] >= lo) & (bars_sorted["timestamp"] <= t)]
        preds = pd.Series(predict_returns(history, model), dtype=float).reindex(symbols)
        if i + 1 >= len(stamps):
            fill_time, target, orders, cost, msg = None, None, {}, 0.0, "last bar: no next price to trade at"
            value = acct.value(opens.iloc[i]); aw = acct.weights(opens.iloc[i])
        else:
            fill_time = stamps[i + 1]
            prices = opens.iloc[i + 1]
            if preds.notna().any():
                target = pd.Series(allocate(preds, config), dtype=float).reindex(symbols).fillna(0.0)
                rec = acct.rebalance(target, prices, cost_rate, timestamp=fill_time)
                orders, cost = {s: v for s, v in rec.trades.items() if abs(v) > 1e-9}, rec.cost
                n_pos = int((target > 0).sum())
                msg = (f"{n_pos} position(s) targeted, {len(orders)} order(s), cost ${cost:.2f}"
                       if orders else f"{n_pos} position(s) targeted, already in line: no trades")
                target = target.to_dict()
            else:
                target, orders, cost = None, {}, 0.0
                msg = "warm-up: features not ready, holding current positions"
            value = acct.value(prices); aw = acct.weights(prices)
        st = ReplayStep(step=k, decision_time=t, fill_time=fill_time, predictions=preds.to_dict(),
                        target_weights=target, actual_weights=aw.to_dict(), cash=acct.cash, value=value,
                        orders=orders, cost=cost, message=msg)
        steps.append(st)
        if on_step is not None:
            on_step(st)
    return ReplayTrace(steps=steps, symbols=symbols, config=config)
