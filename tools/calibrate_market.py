#!/usr/bin/env python
"""Preparation-time calibration: how does the planted signal behave across seeds?

    python tools/calibrate_market.py --seeds 10 --beta 0.10

Reports validation forecast quality and validation/test backtests (net of costs) per seed so the
instructor can see the spread.  Nothing here selects a seed for the classroom by outcome: the
classroom seed is fixed in quantsoc.market.MarketParams.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from quantsoc.market import MarketParams, generate_market, to_wide
from quantsoc.modeling import build_dataset, clean_rows, fit_model, predictions_wide, score_forecasts, xy
from quantsoc.backtest import evaluate_strategy
from quantsoc.solutions import REFERENCE_CONFIG


def run(seed: int, beta: float, cost_bps: float) -> dict:
    p = MarketParams(seed=seed, beta=beta)
    bars = generate_market(p)
    ds = build_dataset(bars)
    model = fit_model(clean_rows(ds, "train"))
    opens = to_wide(bars, "open")
    out = {"seed": seed}
    cfg = REFERENCE_CONFIG
    cfg = type(cfg)(**{**cfg.to_dict(), "cost_bps": cost_bps})
    for split in ("validation", "test"):
        rows = clean_rows(ds, split)
        X, y = xy(rows)
        sc = score_forecasts(y, model.predict(X))
        fc = predictions_wide(model, ds, split)
        res = evaluate_strategy(opens.loc[fc.index], fc, cfg)
        out[f"{split}_corr"] = round(sc.correlation, 3)
        out[f"{split}_mse_impr_%"] = round(sc.mse_improvement_pct, 2)
        out[f"{split}_gross_%"] = round(100 * res["gross"].total_return, 2)
        out[f"{split}_net_%"] = round(100 * res["net"].total_return, 2)
        out[f"{split}_bench_%"] = round(100 * res["benchmark"].total_return, 2)
        out[f"{split}_maxdd_%"] = round(100 * res["net"].max_drawdown, 2)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--beta", type=float, default=MarketParams().beta)
    ap.add_argument("--cost-bps", type=float, default=REFERENCE_CONFIG.cost_bps)
    ap.add_argument("--classroom", action="store_true", help="also include the classroom seed")
    a = ap.parse_args(argv)
    seeds = list(range(1, a.seeds + 1)) + ([MarketParams().seed] if a.classroom else [])
    rows = [run(s, a.beta, a.cost_bps) for s in seeds]
    df = pd.DataFrame(rows).set_index("seed")
    pd.set_option("display.width", 200)
    print(f"beta={a.beta}, cost={a.cost_bps} bps, config={REFERENCE_CONFIG}")
    print(df.to_string())
    print("\nmedian:\n" + df.median().round(2).to_string())
    print(f"\nseeds with net validation profit: {(df['validation_net_%'] > 0).sum()}/{len(df)}; "
          f"net test profit: {(df['test_net_%'] > 0).sum()}/{len(df)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
