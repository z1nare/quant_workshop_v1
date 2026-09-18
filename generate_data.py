#!/usr/bin/env python
"""Regenerate the checked-in synthetic data deterministically.

    python generate_data.py            # writes data/workshop_market.csv and data/stress_market.csv
    python generate_data.py --check    # verify the checked-in files match the generator
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from quantsoc.market import MarketParams, generate_market, save_market, stress_params, load_market

DATA = Path(__file__).resolve().parent / "data"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    targets = {"workshop_market.csv": MarketParams(), "stress_market.csv": stress_params()}
    for name, params in targets.items():
        bars = generate_market(params)
        path = DATA / name
        if args.check:
            existing = load_market(path)
            fresh = load_market_from_frame(bars)
            same = existing.shape == fresh.shape and all(
                (existing[c].round(6) == fresh[c].round(6)).all() if c in ("open", "high", "low", "close")
                else (existing[c] == fresh[c]).all() for c in ["timestamp", "session", "symbol", "open", "high", "low", "close", "volume"])
            print(f"{name}: {'matches generator' if same else 'DIFFERS from generator'}")
            if not same:
                return 1
        else:
            save_market(bars, path)
            print(f"wrote {path} ({len(bars)} rows, seed {params.seed}, {params.n_sessions} sessions)")
    return 0


def load_market_from_frame(bars: pd.DataFrame) -> pd.DataFrame:
    import io
    buf = io.StringIO()
    tmp = bars.copy()
    tmp["timestamp"] = tmp["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    tmp[["open", "high", "low", "close"]] = tmp[["open", "high", "low", "close"]].round(6)
    tmp.to_csv(buf, index=False); buf.seek(0)
    return load_market(buf)


if __name__ == "__main__":
    sys.exit(main())
