#!/usr/bin/env python
"""Local application entry point: run a strategy bundle through the trading engine.

    python run_trader.py --mode replay                       # offline, synthetic data, default
    python run_trader.py --mode paper-preview                # real paper account + data, NO orders
    python run_trader.py --mode paper --i-understand-paper-orders

Without --bundle the reference bundle is built from the checked-in data (exports/reference_bundle).
State is written to ./state and read by `streamlit run dashboard.py`.  Ctrl+C stops gracefully.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quantsoc.artifacts import load_bundle
from quantsoc.broker import BrokerError, ReplayBroker
from quantsoc.engine import TradingEngine
from quantsoc.market import BAR_MINUTES, load_market
from quantsoc.modeling import build_dataset
from quantsoc.notebook_support import get_alpaca_credentials, symbol_map_from_env
from quantsoc.state import StateStore

STOP = {"requested": False}


def _on_signal(signum, frame):
    print(f"\n[runner] stop requested (signal {signum}); finishing the current cycle...")
    STOP["requested"] = True


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["replay", "paper-preview", "paper"], default="replay")
    p.add_argument("--bundle", default=None, help="exported bundle directory (default: exports/reference_bundle)")
    p.add_argument("--data", default=None, help="replay data CSV (default: data/workshop_market.csv, final-test part)")
    p.add_argument("--stress", action="store_true", help="replay the stress scenario instead")
    p.add_argument("--speed", type=float, default=50.0, help="replay acceleration: bars per second (default 50)")
    p.add_argument("--cycles", type=int, default=0, help="stop after N cycles (0 = until data ends / Ctrl+C)")
    p.add_argument("--interval", type=int, default=BAR_MINUTES * 60, help="seconds between paper cycles (default 300)")
    p.add_argument("--state-dir", default="state")
    p.add_argument("--reset-state", action="store_true", help="delete previous state before starting")
    p.add_argument("--i-understand-paper-orders", action="store_true", help="required for --mode paper")
    p.add_argument("--cancel-open-orders-on-stop", action="store_true")
    p.add_argument("--once", action="store_true", help="run a single cycle and exit")
    return p.parse_args(argv)


def load_or_build_bundle(path):
    if path is None:
        default = Path("exports") / "reference_bundle"
        if not (default / "manifest.json").exists():
            from quantsoc.reference_bundle import build_reference_bundle
            print(f"[runner] no bundle given; building the reference bundle at {default} ...")
            build_reference_bundle(default)
        path = default
    bundle = load_bundle(path)
    print(f"[runner] loaded bundle {bundle.path} (strategy source: {bundle.strategy_source}, "
          f"fund: {bundle.manifest.get('fund_name', '?')})")
    return bundle


def main(argv=None) -> int:
    args = parse_args(argv)
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    bundle = load_or_build_bundle(args.bundle)
    state = StateStore(args.state_dir)
    if args.reset_state:
        state.reset(); state = StateStore(args.state_dir)

    if args.mode == "replay":
        data = Path(args.data) if args.data else Path("data") / ("stress_market.csv" if args.stress else "workshop_market.csv")
        bars = load_market(data)
        start = None
        if not args.stress and not args.data:
            ds = build_dataset(bars)
            start = ds[ds["split"] == "test"]["timestamp"].min()          # replay only the untouched final test
        broker = ReplayBroker(bars, start=start)
        engine = TradingEngine(broker, bundle.predict_returns, bundle.allocate, bundle.model, bundle.config,
                               bundle.symbols, state=state, mode="replay", sleep=lambda s: None,
                               strategy_label=bundle.manifest.get("fund_name", "strategy"))
        print(f"[runner] OFFLINE REPLAY of {data} from {broker.decision_time} at {args.speed:g} bars/s. Ctrl+C to stop.")
        n = 0
        while not STOP["requested"]:
            res = engine.run_cycle()
            n += 1
            print(f"[replay] {broker.decision_time:%Y-%m-%d %H:%M}  {res.status:9s} {res.reason}")
            if args.once or (args.cycles and n >= args.cycles) or not broker.advance():
                break
            time.sleep(1.0 / max(args.speed, 0.01))
        acct = broker.account()
        print(f"[runner] replay finished after {n} cycles: equity ${acct.equity:,.2f}, cash ${acct.cash:,.2f}")
        return 0

    key, secret, source = get_alpaca_credentials()
    if not key:
        print("[runner] No Alpaca credentials found (APCA_API_KEY_ID / APCA_API_SECRET_KEY in .env or the environment).\n"
              "         Paper modes need them; the offline replay does not: python run_trader.py --mode replay")
        return 2
    if args.mode == "paper" and not args.i_understand_paper_orders:
        print("[runner] --mode paper submits orders to your Alpaca PAPER account. Add --i-understand-paper-orders to confirm.")
        return 2
    from quantsoc.broker import AlpacaPaperBroker
    broker = AlpacaPaperBroker(key, secret, feed=os.environ.get("QSW_DATA_FEED", "iex"))
    symbol_map = symbol_map_from_env()
    engine = TradingEngine(broker, bundle.predict_returns, bundle.allocate, bundle.model, bundle.config,
                           bundle.symbols, symbol_map=symbol_map, state=state, mode=args.mode,
                           strategy_label=bundle.manifest.get("fund_name", "strategy"))
    print(f"[runner] credentials from {source}; mode {args.mode.upper()}; symbol map {symbol_map}")
    try:
        for sym, msg in engine.validate_instruments().items():
            print(f"[runner]   {sym} -> {msg}")
        clock = broker.clock()
        print(f"[runner] market open: {clock.is_open}; next open {clock.next_open}; next close {clock.next_close}")
    except BrokerError as e:
        print(f"[runner] broker check failed: {e}")
        return 3
    n = 0
    while not STOP["requested"]:
        res = engine.run_cycle(submit=(args.mode == "paper"))
        n += 1
        print(f"[{args.mode}] {datetime.now(timezone.utc):%H:%M:%S}Z  {res.status:9s} {res.reason}")
        if args.once or (args.cycles and n >= args.cycles):
            break
        # sleep until shortly after the next 5-minute boundary
        now = time.time()
        wait = args.interval - (now % args.interval) + 5
        for _ in range(int(wait)):
            if STOP["requested"]:
                break
            time.sleep(1)
    open_orders = []
    try:
        open_orders = broker.open_orders([symbol_map[s] for s in bundle.symbols])
    except BrokerError:
        pass
    if open_orders:
        print(f"[runner] {len(open_orders)} open order(s) at stop:")
        for o in open_orders:
            print(f"         {o.client_order_id} {o.symbol} {o.side} status={o.status}")
            if args.cancel_open_orders_on_stop:
                broker.cancel(o.id); print("         -> cancelled")
    print("[runner] stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
