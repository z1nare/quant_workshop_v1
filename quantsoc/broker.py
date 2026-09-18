"""Broker adapters behind one small interface.

    AlpacaPaperBroker : the official alpaca-py SDK, paper endpoint only, IEX data feed
    ReplayBroker      : offline broker driven by the synthetic bars (fills at the next open)
    MockBroker        : scriptable in-memory broker for tests and the notebook's mocked cycle

The engine (engine.py) only ever talks to the `Broker` protocol, so every safety rule is
exercised identically offline and against the paper API.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Protocol, Set

import numpy as np
import pandas as pd

from .features import MIN_HISTORY_BARS
from .market import BAR_MINUTES, BARS_PER_SESSION, MARKET_TZ

OPEN_STATUSES = {"new", "accepted", "pending_new", "partially_filled", "accepted_for_bidding",
                 "pending_cancel", "pending_replace", "pending_review", "held", "calculated"}
TERMINAL_STATUSES = {"filled", "canceled", "expired", "rejected", "done_for_day", "replaced", "stopped", "suspended"}


class BrokerError(Exception):
    """The broker refused or could not perform the request (outcome KNOWN)."""


class UncertainSubmission(BrokerError):
    """A network/timeout problem: the order MAY have been accepted.  Reconcile before retrying."""


@dataclass
class ClockInfo:
    timestamp: datetime
    is_open: bool
    next_open: Optional[datetime] = None
    next_close: Optional[datetime] = None


@dataclass
class AccountInfo:
    equity: float
    cash: float
    buying_power: float


@dataclass
class PositionInfo:
    symbol: str
    qty: float
    market_value: float
    current_price: float


@dataclass
class OrderInfo:
    id: str
    client_order_id: str
    symbol: str
    side: str
    qty: Optional[float]
    notional: Optional[float]
    filled_qty: float
    filled_avg_price: Optional[float]
    status: str
    submitted_at: Optional[datetime] = None

    @property
    def is_open(self) -> bool:
        return self.status in OPEN_STATUSES

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["submitted_at"] = self.submitted_at.isoformat() if self.submitted_at else None
        return d


@dataclass
class AssetInfo:
    symbol: str
    tradable: bool
    fractionable: bool


class Broker(Protocol):
    name: str
    def clock(self) -> ClockInfo: ...
    def account(self) -> AccountInfo: ...
    def positions(self) -> List[PositionInfo]: ...
    def open_orders(self, symbols: List[str]) -> List[OrderInfo]: ...
    def order_by_client_id(self, client_order_id: str) -> Optional[OrderInfo]: ...
    def submit_market(self, symbol: str, side: str, client_order_id: str,
                      qty: Optional[float] = None, notional: Optional[float] = None) -> OrderInfo: ...
    def cancel(self, order_id: str) -> None: ...
    def bars(self, symbols: List[str], n_bars: int, end: datetime) -> pd.DataFrame: ...
    def latest_prices(self, symbols: List[str]) -> Dict[str, float]: ...
    def asset(self, symbol: str) -> AssetInfo: ...


# --------------------------------------------------------------------------- Alpaca (paper only)

class AlpacaPaperBroker:
    """Thin adapter over alpaca-py.  `paper=True` is hard-wired: this class cannot trade live money."""

    name = "alpaca-paper"

    def __init__(self, api_key: str, secret_key: str, feed: str = "iex"):
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.enums import DataFeed
        if not api_key or not secret_key:
            raise BrokerError("Alpaca API key and secret are required")
        self.trading = TradingClient(api_key, secret_key, paper=True)
        self.data = StockHistoricalDataClient(api_key, secret_key)
        self.feed = DataFeed(feed)

    # -- helpers
    @staticmethod
    def _wrap(fn, *args, **kwargs):
        import requests
        from alpaca.common.exceptions import APIError
        try:
            return fn(*args, **kwargs)
        except APIError as e:
            raise BrokerError(f"Alpaca API error: {e}") from e
        except (requests.ConnectionError, requests.Timeout) as e:
            raise BrokerError(f"network problem talking to Alpaca: {e}") from e

    def clock(self) -> ClockInfo:
        c = self._wrap(self.trading.get_clock)
        return ClockInfo(timestamp=c.timestamp, is_open=bool(c.is_open), next_open=c.next_open, next_close=c.next_close)

    def account(self) -> AccountInfo:
        a = self._wrap(self.trading.get_account)
        return AccountInfo(equity=float(a.equity), cash=float(a.cash), buying_power=float(a.buying_power))

    def positions(self) -> List[PositionInfo]:
        out = []
        for p in self._wrap(self.trading.get_all_positions):
            out.append(PositionInfo(symbol=p.symbol, qty=float(p.qty), market_value=float(p.market_value),
                                    current_price=float(p.current_price)))
        return out

    @staticmethod
    def _order(o) -> OrderInfo:
        return OrderInfo(id=str(o.id), client_order_id=o.client_order_id or "", symbol=o.symbol,
                         side=str(getattr(o.side, "value", o.side)),
                         qty=float(o.qty) if o.qty is not None else None,
                         notional=float(o.notional) if o.notional is not None else None,
                         filled_qty=float(o.filled_qty or 0),
                         filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price else None,
                         status=str(getattr(o.status, "value", o.status)), submitted_at=o.submitted_at)

    def open_orders(self, symbols: List[str]) -> List[OrderInfo]:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=list(symbols), limit=500)
        return [self._order(o) for o in self._wrap(self.trading.get_orders, req)]

    def order_by_client_id(self, client_order_id: str) -> Optional[OrderInfo]:
        from alpaca.common.exceptions import APIError
        try:
            return self._order(self.trading.get_order_by_client_id(client_order_id))
        except APIError as e:
            if getattr(e, "status_code", None) == 404 or "not found" in str(e).lower():
                return None
            raise BrokerError(f"Alpaca API error: {e}") from e

    def submit_market(self, symbol: str, side: str, client_order_id: str,
                      qty: Optional[float] = None, notional: Optional[float] = None) -> OrderInfo:
        import requests
        from alpaca.common.exceptions import APIError
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        req = MarketOrderRequest(symbol=symbol, side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                                 time_in_force=TimeInForce.DAY, client_order_id=client_order_id,
                                 qty=qty, notional=notional)
        try:
            return self._order(self.trading.submit_order(req))
        except APIError as e:
            raise BrokerError(f"order rejected by Alpaca: {e}") from e
        except (requests.ConnectionError, requests.Timeout) as e:
            raise UncertainSubmission(f"network problem during submission: {e}") from e

    def cancel(self, order_id: str) -> None:
        self._wrap(self.trading.cancel_order_by_id, order_id)

    def bars(self, symbols: List[str], n_bars: int, end: datetime) -> pd.DataFrame:
        """Completed 5-minute bars (IEX feed) as the workshop's long bar table."""
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        end = pd.Timestamp(end).tz_convert("UTC") if pd.Timestamp(end).tzinfo else pd.Timestamp(end, tz="UTC")
        start = end - pd.to_timedelta(4, unit='D')                      # spans the previous sessions comfortably
        req = StockBarsRequest(symbol_or_symbols=list(symbols), timeframe=TimeFrame(BAR_MINUTES, TimeFrameUnit.Minute),
                               start=start.to_pydatetime(), end=end.to_pydatetime(), feed=self.feed, limit=10_000)
        barset = self._wrap(self.data.get_stock_bars, req)
        df = barset.df
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["timestamp", "session", "symbol", "open", "high", "low", "close", "volume"])
        df = df.reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(MARKET_TZ)
        return completed_bars_only(df, end.tz_convert(MARKET_TZ), n_bars)

    def latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        from alpaca.data.requests import StockLatestTradeRequest
        req = StockLatestTradeRequest(symbol_or_symbols=list(symbols), feed=self.feed)
        trades = self._wrap(self.data.get_stock_latest_trade, req)
        return {sym: float(t.price) for sym, t in trades.items()}

    def asset(self, symbol: str) -> AssetInfo:
        a = self._wrap(self.trading.get_asset, symbol)
        return AssetInfo(symbol=symbol, tradable=bool(a.tradable), fractionable=bool(a.fractionable))


def completed_bars_only(df: pd.DataFrame, now: pd.Timestamp, n_bars: int) -> pd.DataFrame:
    """Keep regular-session bars whose END (start + 5 min) is <= now; label sessions by New York date."""
    df = df.copy()
    ts = df["timestamp"]
    local = ts.dt.tz_convert(MARKET_TZ)
    minutes = local.dt.hour * 60 + local.dt.minute
    in_session = (minutes >= 9 * 60 + 30) & (minutes < 16 * 60)
    completed = (ts + pd.to_timedelta(BAR_MINUTES, unit='m')) <= pd.Timestamp(now)
    df = df[in_session & completed]
    df["session"] = local[in_session & completed].dt.strftime("%Y-%m-%d")
    df = df[["timestamp", "session", "symbol", "open", "high", "low", "close", "volume"]]
    df = df.sort_values(["timestamp", "symbol"])
    keep = df.groupby("symbol").tail(n_bars)
    return keep.reset_index(drop=True)


# --------------------------------------------------------------------------- in-memory brokers

class MockBroker:
    """Scriptable in-memory broker.  Fills market orders immediately at `prices` unless told otherwise."""

    name = "mock"

    def __init__(self, cash: float = 100_000.0, prices: Optional[Dict[str, float]] = None,
                 bars_df: Optional[pd.DataFrame] = None, now: Optional[datetime] = None, is_open: bool = True):
        self.cash = float(cash)
        self.prices: Dict[str, float] = dict(prices or {})
        self.bars_df = bars_df
        self.now = now or datetime.now(timezone.utc)
        self.is_open = is_open
        self.next_open = None
        self.next_close = None
        self.holdings: Dict[str, float] = {}
        self.orders: Dict[str, OrderInfo] = {}
        self.submissions: List[dict] = []
        self.cancelled: List[str] = []
        # scripted behaviours
        self.reject_symbols: Set[str] = set()
        self.partial_fill: Dict[str, float] = {}          # symbol -> fraction filled
        self.pending_symbols: Set[str] = set()             # accepted but never fills
        self.uncertain_symbols: Set[str] = set()           # accepted, then raise UncertainSubmission
        self.fail_next_submit: bool = False                # raise UncertainSubmission WITHOUT accepting
        self.non_fractionable: Set[str] = set()
        self.untradable: Set[str] = set()
        self.buying_power_multiplier = 1.0

    # -- state helpers
    def clock(self) -> ClockInfo:
        return ClockInfo(timestamp=self.now, is_open=self.is_open, next_open=self.next_open, next_close=self.next_close)

    def equity(self) -> float:
        return self.cash + sum(q * self.prices[s] for s, q in self.holdings.items())

    def account(self) -> AccountInfo:
        return AccountInfo(equity=self.equity(), cash=self.cash, buying_power=self.cash * self.buying_power_multiplier)

    def positions(self) -> List[PositionInfo]:
        return [PositionInfo(symbol=s, qty=q, market_value=q * self.prices[s], current_price=self.prices[s])
                for s, q in self.holdings.items() if abs(q) > 1e-12]

    def open_orders(self, symbols: List[str]) -> List[OrderInfo]:
        return [o for o in self.orders.values() if o.symbol in symbols and o.is_open]

    def order_by_client_id(self, client_order_id: str) -> Optional[OrderInfo]:
        for o in self.orders.values():
            if o.client_order_id == client_order_id:
                return o
        return None

    def _fill(self, o: OrderInfo, fraction: float) -> None:
        px = self.prices[o.symbol]
        qty = o.qty if o.qty is not None else o.notional / px
        fq = qty * fraction
        if o.side == "buy":
            self.cash -= fq * px
            self.holdings[o.symbol] = self.holdings.get(o.symbol, 0.0) + fq
        else:
            self.cash += fq * px
            self.holdings[o.symbol] = self.holdings.get(o.symbol, 0.0) - fq
        o.filled_qty = fq
        o.filled_avg_price = px
        o.status = "filled" if fraction >= 1 - 1e-12 else "partially_filled"

    def submit_market(self, symbol: str, side: str, client_order_id: str,
                      qty: Optional[float] = None, notional: Optional[float] = None) -> OrderInfo:
        if self.fail_next_submit:
            self.fail_next_submit = False
            raise UncertainSubmission("simulated timeout before the broker acknowledged the order")
        if self.order_by_client_id(client_order_id) is not None:
            raise BrokerError(f"client_order_id {client_order_id} already exists")
        if symbol in self.untradable:
            raise BrokerError(f"asset {symbol} is not tradable")
        if (qty is None) == (notional is None):
            raise BrokerError("exactly one of qty / notional is required")
        px = self.prices[symbol]
        o = OrderInfo(id=str(uuid.uuid4()), client_order_id=client_order_id, symbol=symbol, side=side, qty=qty,
                      notional=notional, filled_qty=0.0, filled_avg_price=None, status="new", submitted_at=self.now)
        self.submissions.append({"symbol": symbol, "side": side, "qty": qty, "notional": notional, "client_order_id": client_order_id})
        if symbol in self.reject_symbols:
            o.status = "rejected"
            self.orders[o.id] = o
            raise BrokerError(f"order rejected by broker for {symbol}")
        need = (qty * px if qty is not None else notional)
        if side == "buy" and need > self.cash * self.buying_power_multiplier + 1e-9:
            o.status = "rejected"
            self.orders[o.id] = o
            raise BrokerError(f"insufficient buying power for {symbol}: need {need:.2f}, have {self.cash:.2f}")
        if side == "sell" and (qty or 0) > self.holdings.get(symbol, 0.0) * (1 + 1e-9) + 1e-9:
            o.status = "rejected"
            self.orders[o.id] = o
            raise BrokerError(f"insufficient qty available for {symbol}")
        self.orders[o.id] = o
        if symbol in self.pending_symbols:
            o.status = "accepted"
        elif symbol in self.partial_fill:
            self._fill(o, self.partial_fill[symbol])
        else:
            self._fill(o, 1.0)
        if symbol in self.uncertain_symbols:
            raise UncertainSubmission("simulated timeout AFTER the broker accepted the order")
        return o

    def cancel(self, order_id: str) -> None:
        o = self.orders[order_id]
        if o.is_open:
            o.status = "canceled"
        self.cancelled.append(order_id)

    def bars(self, symbols: List[str], n_bars: int, end: datetime) -> pd.DataFrame:
        if self.bars_df is None:
            return pd.DataFrame(columns=["timestamp", "session", "symbol", "open", "high", "low", "close", "volume"])
        df = self.bars_df[self.bars_df["symbol"].isin(symbols)]
        return completed_bars_only(df, pd.Timestamp(end), n_bars)

    def latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        return {s: self.prices[s] for s in symbols}

    def asset(self, symbol: str) -> AssetInfo:
        return AssetInfo(symbol=symbol, tradable=symbol not in self.untradable, fractionable=symbol not in self.non_fractionable)


class ReplayBroker(MockBroker):
    """Offline broker that walks through the synthetic bars one bar at a time.

    `cursor` points at the bar that has just COMPLETED (the decision bar).  "Now" is its end.
    Prices for fills are the OPEN of the next bar (the next tradable price), exactly as in the
    research backtest.  The market is 'closed' after the last bar of a session until `advance()`.
    """

    name = "replay"

    def __init__(self, bars: pd.DataFrame, start: Optional[pd.Timestamp] = None, cash: float = 100_000.0):
        super().__init__(cash=cash)
        self.bars_df = bars.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        self.stamps = pd.DatetimeIndex(sorted(self.bars_df["timestamp"].unique()))
        self.opens = self.bars_df.pivot(index="timestamp", columns="symbol", values="open").sort_index()
        self.sessions = self.bars_df.drop_duplicates("timestamp").set_index("timestamp")["session"].sort_index()
        self.cursor = 0 if start is None else int(np.searchsorted(self.stamps, pd.Timestamp(start)))
        self._sync()

    def _sync(self) -> None:
        t = self.stamps[self.cursor]
        self.now = (t + pd.to_timedelta(BAR_MINUTES, unit='m')).to_pydatetime()
        has_next = self.cursor + 1 < len(self.stamps)
        same_session = has_next and self.sessions.iloc[self.cursor + 1] == self.sessions.iloc[self.cursor]
        self.is_open = bool(same_session)
        if has_next:
            self.prices = self.opens.iloc[self.cursor + 1].to_dict()      # next tradable price
            self.next_open = self.stamps[self.cursor + 1].to_pydatetime()
        self.next_close = None

    @property
    def decision_time(self) -> pd.Timestamp:
        return self.stamps[self.cursor]

    @property
    def finished(self) -> bool:
        return self.cursor + 1 >= len(self.stamps)

    def advance(self) -> bool:
        """Move to the next completed bar.  Returns False when the data is exhausted."""
        if self.finished:
            return False
        self.cursor += 1
        self._sync()
        return True

    def bars(self, symbols: List[str], n_bars: int, end: datetime) -> pd.DataFrame:
        t = self.stamps[self.cursor]
        df = self.bars_df[(self.bars_df["timestamp"] <= t) & self.bars_df["symbol"].isin(symbols)]
        return df.groupby("symbol").tail(n_bars).reset_index(drop=True)
