"""The supplied trading engine.  Students never edit this.

One `run_cycle()` = one decision after a completed 5-minute bar:

  1. market/session status          -> skip if closed or inside the close buffer
  2. completed bars from the broker -> skip stale symbols; skip cycle if nothing usable
  3. strategy.predict_returns       -> non-finite forecasts are excluded; all-NaN = warm-up skip
  4. strategy.allocate              -> validated against the portfolio constraints (safety layer)
  5. account, positions, OPEN ORDERS-> older open orders are cancelled and the cycle waits;
                                       orders from this same bar are treated as already submitted
  6. target vs ACTUAL positions     -> dollar differences, rebalance band, buying power
  7. sells first, wait for fills, then buys sized on the cash that is actually available
  8. every order carries a deterministic client_order_id "qsw-<bar>-<symbol>-<side>":
     the registry + broker lookup by client id prevent duplicates on reruns/restarts, and an
     uncertain submission is reconciled by that id before any retry.

Modes: "replay" (offline, ReplayBroker), "paper-preview" (real data/account, NO submission),
"paper" (submits to the Alpaca paper endpoint only when `submit=True`).
Exactly-once execution over an unreliable network is impossible; stable ids + reconciliation
is the honest substitute.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .broker import (Broker, BrokerError, OrderInfo, UncertainSubmission, PositionInfo)
from .features import MIN_HISTORY_BARS
from .market import BAR_MINUTES, MARKET_TZ
from .portfolio import PortfolioConfig, check_weights
from .state import StateStore

MODES = ("replay", "paper-preview", "paper")
CLIENT_ID_PREFIX = "qsw"


@dataclass
class ProposedOrder:
    symbol: str                 # workshop symbol (AURA ...)
    real_symbol: str            # broker symbol (SPY ...)
    side: str                   # buy / sell
    qty: Optional[float]
    notional: Optional[float]
    price: float
    dollars: float              # signed target change
    client_order_id: str
    reason: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class CycleResult:
    mode: str
    now: str
    status: str                             # traded | preview | skipped | no_trade | error
    reason: str
    decision_bar: Optional[str] = None
    predictions: Dict[str, float] = field(default_factory=dict)
    target_weights: Dict[str, float] = field(default_factory=dict)
    current_weights: Dict[str, float] = field(default_factory=dict)
    account: Dict[str, float] = field(default_factory=dict)
    positions: List[dict] = field(default_factory=list)
    proposed_orders: List[dict] = field(default_factory=list)
    submitted_orders: List[dict] = field(default_factory=list)
    skipped_symbols: Dict[str, str] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def bar_key(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).tz_convert(MARKET_TZ).strftime("%Y%m%d-%H%M")


def make_client_order_id(decision_bar: pd.Timestamp, real_symbol: str, side: str) -> str:
    return f"{CLIENT_ID_PREFIX}-{bar_key(decision_bar)}-{real_symbol}-{side}"


class TradingEngine:
    def __init__(self, broker: Broker, predict_returns: Callable, allocate: Callable, model,
                 config: PortfolioConfig, symbols: List[str], symbol_map: Optional[Dict[str, str]] = None,
                 state: Optional[StateStore] = None, mode: str = "replay", lookback_bars: int = 40,
                 rebalance_band: float = 0.0025, min_notional: float = 1.0, order_wait_seconds: float = 20.0,
                 poll_interval: float = 1.0, close_buffer_minutes: float = 5.0, stale_after_bars: int = 2,
                 sleep: Callable[[float], None] = time.sleep, strategy_label: str = "strategy"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.broker = broker
        self.predict_returns = predict_returns
        self.allocate = allocate
        self.model = model
        self.config = config.validate()
        self.symbols = list(symbols)
        self.symbol_map = dict(symbol_map or {s: s for s in symbols})
        missing = [s for s in self.symbols if s not in self.symbol_map]
        if missing:
            raise ValueError(f"symbol_map is missing {missing}")
        self.reverse_map = {v: k for k, v in self.symbol_map.items()}
        self.state = state or StateStore()
        self.mode = mode
        self.lookback_bars = lookback_bars
        self.rebalance_band = rebalance_band
        self.min_notional = min_notional
        self.order_wait_seconds = order_wait_seconds
        self.poll_interval = poll_interval
        self.close_buffer = timedelta(minutes=close_buffer_minutes)
        self.stale_after = timedelta(minutes=BAR_MINUTES * stale_after_bars)
        self.sleep = sleep
        self.strategy_label = strategy_label
        self._asset_cache: Dict[str, object] = {}
        self.cycles = 0

    # ------------------------------------------------------------------ helpers
    def _log(self, result: CycleResult, line: str) -> None:
        result.log.append(line)
        self.state.log(f"[{self.mode}] {line}")

    def _asset(self, real_symbol: str):
        if real_symbol not in self._asset_cache:
            self._asset_cache[real_symbol] = self.broker.asset(real_symbol)
        return self._asset_cache[real_symbol]

    def validate_instruments(self) -> Dict[str, str]:
        """Check every mapped instrument exists and is tradable.  Returns {symbol: message}."""
        report = {}
        for sym, real in self.symbol_map.items():
            try:
                a = self._asset(real)
                report[sym] = f"{real}: tradable={a.tradable}, fractionable={a.fractionable}"
                if not a.tradable:
                    raise BrokerError(f"{real} is not tradable")
            except BrokerError as e:
                report[sym] = f"{real}: ERROR {e}"
        return report

    def _finish(self, result: CycleResult, status: str, reason: str, equity: Optional[float] = None) -> CycleResult:
        result.status, result.reason = status, reason
        self._log(result, f"{status.upper()}: {reason}")
        self.state.write_snapshot({"mode": self.mode, "broker": getattr(self.broker, "name", "?"),
                                   "strategy": self.strategy_label, "last_cycle": result.to_dict(),
                                   "config": self.config.to_dict(), "symbol_map": self.symbol_map,
                                   "cycles": self.cycles},
                                  equity=equity, sim_time=result.now)
        return result

    # ------------------------------------------------------------------ the cycle
    def run_cycle(self, submit: Optional[bool] = None) -> CycleResult:
        """Run one decision cycle.  Orders are submitted only if mode == 'paper' and submit is True,
        or mode == 'replay' (offline simulation)."""
        self.cycles += 1
        if submit is None:
            submit = self.mode == "replay"
        if self.mode == "paper-preview":
            submit = False
        if self.mode == "paper" and not submit:
            pass  # explicit preview inside paper mode
        clock = self.broker.clock()
        now = pd.Timestamp(clock.timestamp)
        now = now.tz_localize("UTC") if now.tzinfo is None else now
        result = CycleResult(mode=self.mode, now=now.isoformat(), status="running", reason="")
        self._log(result, f"cycle {self.cycles} at {now.tz_convert(MARKET_TZ):%Y-%m-%d %H:%M:%S %Z}")

        # 1. market status
        if not clock.is_open:
            nxt = f"; next open {clock.next_open}" if clock.next_open else ""
            return self._finish(result, "skipped", f"market closed{nxt}")
        if clock.next_close is not None and pd.Timestamp(clock.next_close) - now < self.close_buffer:
            return self._finish(result, "skipped", f"inside the {self.close_buffer.seconds // 60}-minute close buffer")

        # 2. data
        real_symbols = [self.symbol_map[s] for s in self.symbols]
        try:
            hist = self.broker.bars(real_symbols, self.lookback_bars, now.to_pydatetime())
        except BrokerError as e:
            return self._finish(result, "error", f"could not fetch bars: {e}")
        if hist is None or len(hist) == 0:
            return self._finish(result, "skipped", "no completed bars returned by the broker")
        hist = hist.copy()
        hist["symbol"] = hist["symbol"].map(self.reverse_map).fillna(hist["symbol"])
        last_ts = hist.groupby("symbol")["timestamp"].max()
        latest_bar = last_ts.max()
        for sym in self.symbols:
            if sym not in last_ts.index:
                result.skipped_symbols[sym] = "no bars"
                continue
            age = now - (pd.Timestamp(last_ts[sym]) + pd.to_timedelta(BAR_MINUTES, unit='m'))
            if age > self.stale_after:
                result.skipped_symbols[sym] = f"stale data: last completed bar {last_ts[sym]} ({age} old)"
            n_session = int((hist[(hist.symbol == sym) & (hist.session == hist[hist.symbol == sym].session.iloc[-1])]).shape[0])
            if n_session < MIN_HISTORY_BARS and sym not in result.skipped_symbols:
                result.skipped_symbols[sym] = f"warm-up: {n_session}/{MIN_HISTORY_BARS} bars in this session"
        usable = [s for s in self.symbols if s not in result.skipped_symbols]
        for sym, why in result.skipped_symbols.items():
            self._log(result, f"{sym}: excluded ({why})")
        if not usable:
            return self._finish(result, "skipped", "no symbol has fresh, complete data (warm-up or stale feed)")
        decision_bar = pd.Timestamp(latest_bar)
        result.decision_bar = decision_bar.isoformat()

        # 3. forecasts
        try:
            preds = pd.Series(self.predict_returns(hist[hist.symbol.isin(usable)], self.model), dtype=float)
        except Exception as e:                                   # a bug in student code must not crash the runner
            return self._finish(result, "error", f"predict_returns raised {type(e).__name__}: {e}")
        preds = preds.reindex(self.symbols)
        bad = preds.index[~np.isfinite(preds.to_numpy())]
        for sym in bad:
            if sym not in result.skipped_symbols:
                result.skipped_symbols[sym] = "no valid forecast (NaN/inf)"
        preds[bad] = np.nan
        result.predictions = {k: (None if not np.isfinite(v) else float(v)) for k, v in preds.items()}
        if preds.notna().sum() == 0:
            return self._finish(result, "skipped", "no valid forecasts (warm-up or invalid model output): holding")

        # 4. allocation + safety check
        try:
            target = pd.Series(self.allocate(preds, self.config), dtype=float).reindex(self.symbols).fillna(0.0)
        except Exception as e:
            return self._finish(result, "error", f"allocate raised {type(e).__name__}: {e}")
        report = check_weights(target, self.config)
        if not report["ok"]:
            return self._finish(result, "error", f"allocation violates constraints: {report}")
        result.target_weights = {k: float(v) for k, v in target.items()}

        # 5. account, positions, open orders
        try:
            account = self.broker.account()
            positions = {self.reverse_map.get(p.symbol, p.symbol): p for p in self.broker.positions()}
            open_orders = self.broker.open_orders(real_symbols)
        except BrokerError as e:
            return self._finish(result, "error", f"could not read account state: {e}")
        result.account = {"equity": account.equity, "cash": account.cash, "buying_power": account.buying_power}
        result.positions = [{"symbol": s, "real_symbol": p.symbol, "qty": p.qty, "market_value": p.market_value,
                             "price": p.current_price} for s, p in positions.items()]
        equity = account.equity
        result.current_weights = {s: (positions[s].market_value / equity if s in positions and equity > 0 else 0.0)
                                  for s in self.symbols}
        this_bar_prefix = f"{CLIENT_ID_PREFIX}-{bar_key(decision_bar)}-"
        already_this_bar = set()
        stale_open = []
        for o in open_orders:
            if o.client_order_id.startswith(this_bar_prefix):
                already_this_bar.add(self.reverse_map.get(o.symbol, o.symbol))
            else:
                stale_open.append(o)
        if stale_open and submit:
            for o in stale_open:
                self._log(result, f"cancelling stale open order {o.client_order_id or o.id} ({o.symbol} {o.side} status={o.status})")
                try:
                    self.broker.cancel(o.id)
                except BrokerError as e:
                    self._log(result, f"cancel failed: {e}")
            return self._finish(result, "skipped", f"{len(stale_open)} stale open order(s) cancelled; waiting one cycle for positions to settle", equity)
        elif stale_open:
            self._log(result, f"{len(stale_open)} open order(s) from earlier bars exist; a live run would cancel them first")

        # 6. differences vs ACTUAL positions
        need_prices = [self.symbol_map[s] for s in self.symbols if s not in positions]
        prices = {s: positions[s].current_price for s in positions}
        if need_prices:
            try:
                fetched = self.broker.latest_prices(need_prices)
            except BrokerError as e:
                return self._finish(result, "error", f"could not fetch prices: {e}", equity)
            for real, px in fetched.items():
                prices[self.reverse_map.get(real, real)] = float(px)
        band = max(self.min_notional, self.rebalance_band * equity)
        proposed: List[ProposedOrder] = []
        for sym in self.symbols:
            real = self.symbol_map[sym]
            px = prices.get(sym)
            if px is None or not math.isfinite(px) or px <= 0:
                self._log(result, f"{sym}: no usable price, skipping")
                continue
            current_value = positions[sym].market_value if sym in positions else 0.0
            target_value = float(target[sym]) * equity
            diff = target_value - current_value
            if sym in already_this_bar:
                self._log(result, f"{sym}: order for this bar already open at the broker, not resubmitting")
                continue
            if abs(diff) < band:
                continue
            asset = self._asset(real)
            cid_side = "sell" if diff < 0 else "buy"
            cid = make_client_order_id(decision_bar, real, cid_side)
            if diff < 0:
                pos_qty = positions[sym].qty
                qty = min(abs(diff) / px, pos_qty)
                if target[sym] <= 0 or qty >= pos_qty * 0.995:
                    qty = pos_qty                                    # full exit: sell everything, no dust
                if not asset.fractionable:
                    qty = math.floor(qty)
                elif qty < pos_qty:
                    qty = min(round(qty, 6), pos_qty)               # never round above what we hold
                if qty <= 0:
                    continue
                proposed.append(ProposedOrder(sym, real, "sell", qty=qty, notional=None, price=px,
                                              dollars=-qty * px, client_order_id=cid,
                                              reason=f"target {target[sym]:.1%} vs current {current_value/equity:.1%}"))
            else:
                if asset.fractionable:
                    proposed.append(ProposedOrder(sym, real, "buy", qty=None, notional=round(diff, 2), price=px,
                                                  dollars=diff, client_order_id=cid,
                                                  reason=f"target {target[sym]:.1%} vs current {current_value/equity:.1%}"))
                else:
                    qty = math.floor(diff / px)
                    if qty <= 0:
                        continue
                    proposed.append(ProposedOrder(sym, real, "buy", qty=float(qty), notional=None, price=px,
                                                  dollars=qty * px, client_order_id=cid,
                                                  reason=f"target {target[sym]:.1%} vs current {current_value/equity:.1%} (whole shares)"))
        # buying-power check: buys must fit into cash + sell proceeds (no leverage)
        sells = [o for o in proposed if o.side == "sell"]
        buys = [o for o in proposed if o.side == "buy"]
        expected_cash = account.cash + sum(-o.dollars for o in sells)
        available = max(0.0, min(expected_cash, account.buying_power + sum(-o.dollars for o in sells)))
        total_buy = sum(o.dollars for o in buys)
        if total_buy > available + 1e-6:
            scale = available / total_buy if total_buy > 0 else 0.0
            self._log(result, f"insufficient buying power for {total_buy:.2f} of buys (available {available:.2f}); scaling buys by {scale:.2%}")
            for o in buys:
                o.dollars *= scale
                if o.notional is not None:
                    o.notional = round(o.dollars, 2)
                else:
                    o.qty = float(math.floor(o.dollars / o.price))
            buys = [o for o in buys if (o.notional or 0) >= self.min_notional or (o.qty or 0) >= 1]
            proposed = sells + buys
        result.proposed_orders = [o.to_dict() for o in proposed]
        for o in proposed:
            self._log(result, f"proposed {o.side.upper()} {o.real_symbol} ({o.symbol}) "
                              f"{'notional $%.2f' % o.notional if o.notional is not None else 'qty %.4f' % o.qty} "
                              f"@ ~{o.price:.2f} [{o.reason}]")

        if not proposed:
            return self._finish(result, "no_trade", "targets already within the rebalance band of current positions", equity)
        if not submit:
            return self._finish(result, "preview", f"{len(proposed)} order(s) proposed, nothing submitted", equity)

        # 7. sells first, then buys
        submitted = []
        sell_infos = []
        for o in sells:
            info = self._submit_with_reconciliation(o, result)
            if info is not None:
                submitted.append(info); sell_infos.append(info)
        if sell_infos:
            self._wait_for_orders(sell_infos, result)
            try:
                account = self.broker.account()
            except BrokerError as e:
                return self._finish(result, "error", f"could not re-read account after sells: {e}", equity)
        cash_now = max(0.0, min(account.cash, account.buying_power))
        buy_infos = []
        for o in buys:
            if o.notional is not None and o.notional > cash_now + 1e-6:
                self._log(result, f"{o.real_symbol}: buy trimmed from ${o.notional:.2f} to ${cash_now:.2f} of available cash")
                o.notional = round(cash_now, 2)
                if o.notional < self.min_notional:
                    self._log(result, f"{o.real_symbol}: buy skipped (below minimum notional)")
                    continue
            elif o.qty is not None and o.qty * o.price > cash_now + 1e-6:
                o.qty = float(math.floor(cash_now / o.price))
                if o.qty < 1:
                    self._log(result, f"{o.real_symbol}: buy skipped (cannot afford one share)")
                    continue
            info = self._submit_with_reconciliation(o, result)
            if info is not None:
                submitted.append(info); buy_infos.append(info)
                cash_now -= (o.notional if o.notional is not None else o.qty * o.price)
        if buy_infos:
            self._wait_for_orders(buy_infos, result)
        result.submitted_orders = [s.to_dict() for s in submitted]
        try:
            equity = self.broker.account().equity
        except BrokerError:
            pass
        n_filled = sum(1 for s in submitted if s.status == "filled")
        return self._finish(result, "traded", f"{len(submitted)} order(s) submitted, {n_filled} filled", equity)

    # ------------------------------------------------------------------ submission + reconciliation
    def _submit_with_reconciliation(self, o: ProposedOrder, result: CycleResult) -> Optional[OrderInfo]:
        cid = o.client_order_id
        if self.state.has_order(cid):
            self._log(result, f"{cid}: already in the local order registry, not resubmitting")
            existing = self._lookup(cid)
            return existing
        existing = self._lookup(cid)
        if existing is not None:
            self._log(result, f"{cid}: already exists at the broker (status={existing.status}), not resubmitting")
            self.state.record_order(cid, {**existing.to_dict(), "note": "found at broker before submission"})
            return existing
        for attempt in (1, 2):
            try:
                info = self.broker.submit_market(o.real_symbol, o.side, cid, qty=o.qty, notional=o.notional)
                self.state.record_order(cid, info.to_dict())
                self._log(result, f"submitted {o.side.upper()} {o.real_symbol} client_id={cid} status={info.status}")
                return info
            except UncertainSubmission as e:
                self._log(result, f"{cid}: uncertain submission ({e}); reconciling by client id")
                found = self._reconcile(cid)
                if found is not None:
                    self.state.record_order(cid, {**found.to_dict(), "note": "recovered after uncertain submission"})
                    self._log(result, f"{cid}: found at broker with status={found.status}; no retry needed")
                    return found
                if attempt == 1:
                    self._log(result, f"{cid}: not found at broker; retrying once")
                    continue
                self._log(result, f"{cid}: still unknown after retry; giving up this cycle")
                return None
            except BrokerError as e:
                self._log(result, f"{cid}: REJECTED at submission: {e}")
                self.state.record_order(cid, {"symbol": o.real_symbol, "side": o.side, "status": "rejected", "error": str(e)})
                return None
        return None

    def _lookup(self, cid: str) -> Optional[OrderInfo]:
        try:
            return self.broker.order_by_client_id(cid)
        except BrokerError:
            return None

    def _reconcile(self, cid: str, tries: int = 3) -> Optional[OrderInfo]:
        for _ in range(tries):
            found = self._lookup(cid)
            if found is not None:
                return found
            self.sleep(self.poll_interval)
        return None

    def _wait_for_orders(self, orders: List[OrderInfo], result: CycleResult) -> None:
        deadline = time.monotonic() + self.order_wait_seconds
        pending = [o for o in orders if not o.is_terminal]
        while pending and time.monotonic() < deadline:
            self.sleep(self.poll_interval)
            refreshed = []
            for o in pending:
                latest = self._lookup(o.client_order_id)
                if latest is not None:
                    o.status, o.filled_qty, o.filled_avg_price = latest.status, latest.filled_qty, latest.filled_avg_price
                if not o.is_terminal:
                    refreshed.append(o)
            pending = refreshed
        for o in orders:
            if o.status == "filled":
                self._log(result, f"filled {o.side.upper()} {o.symbol} qty={o.filled_qty} avg={o.filled_avg_price}")
            elif o.status == "partially_filled":
                self._log(result, f"PARTIAL fill {o.side.upper()} {o.symbol} filled_qty={o.filled_qty}; next cycle re-plans from actual positions")
            elif o.status == "rejected":
                self._log(result, f"REJECTED {o.side.upper()} {o.symbol}")
            elif not o.is_terminal:
                self._log(result, f"order {o.client_order_id} still {o.status} after {self.order_wait_seconds}s; it will be reconciled next cycle")
            self.state.record_order(o.client_order_id, o.to_dict())
