import numpy as np
import pandas as pd
import pytest

from quantsoc.backtest import SimAccount, run_backtest, decisions_from_forecasts, evaluate_strategy
from quantsoc.portfolio import PortfolioConfig


def test_hand_calculated_two_asset_example():
    """Decision at t0 -> fill at open[t1]; drift; second decision computed against drifted holdings."""
    idx = pd.date_range("2025-01-06 09:30", periods=4, freq="5min", tz="America/New_York")
    opens = pd.DataFrame({"A": [100.0, 100.0, 110.0, 110.0], "B": [50.0, 50.0, 50.0, 40.0]}, index=idx)
    cfg = PortfolioConfig(threshold=0, exposure_budget=1.0, per_asset_cap=0.5, max_positions=2, cost_bps=10)
    tw = pd.DataFrame([[0.5, 0.5], [0.5, 0.0], [np.nan, np.nan]], index=idx[:3], columns=["A", "B"])
    res = run_backtest(opens, tw, cfg, initial_cash=10_000)
    # fill 1 at t1: value 10000, est cost = 10000*0.001 = 10 -> investable 9990 -> buy 4995 of each
    cost1 = 9990 * 0.001
    assert np.isclose(res.costs.iloc[1], cost1)
    assert np.isclose(res.equity.iloc[1], 10_000 - cost1)
    units_a, units_b = 4995 / 100, 4995 / 50
    cash1 = 10_000 - 9990 - cost1
    # t2: A goes to 110 -> A = 5494.5, B = 4995, value = cash1 + 10489.5
    value2 = cash1 + units_a * 110 + units_b * 50
    # target A 50%, B 0 against drifted holdings: est trades = 0.5*value2 - 5494.5 and -4995
    est_cost = (abs(0.5 * value2 - 5494.5) + 4995) * 0.001
    investable = value2 - est_cost
    trade_a = 0.5 * investable - 5494.5
    trade_b = -4995.0
    cost2 = (abs(trade_a) + abs(trade_b)) * 0.001
    assert np.isclose(res.costs.iloc[2], cost2)
    assert np.isclose(res.trades.iloc[2]["B"], trade_b)
    assert np.isclose(res.equity.iloc[2], value2 - cost2)
    assert np.isclose(res.turnover.iloc[2], (abs(trade_a) + abs(trade_b)) / value2)
    # t3: hold (NaN decision), B falls to 40 but we hold none of it; A flat
    assert res.held.iloc[3] and res.costs.iloc[3] == 0
    assert np.isclose(res.equity.iloc[3], res.equity.iloc[2])
    assert (res.cash >= 0).all()


def test_gross_equals_net_when_costs_zero(small_bars):
    from quantsoc.market import to_wide
    opens = to_wide(small_bars, "open").iloc[:200]
    cfg = PortfolioConfig(threshold=0, cost_bps=0)
    tw = pd.DataFrame(0.3, index=opens.index[:-1], columns=opens.columns[:3]).reindex(columns=opens.columns).fillna(0)
    a = run_backtest(opens, tw, cfg)
    b = run_backtest(opens, tw, cfg, cost_bps=0.0)
    pd.testing.assert_series_equal(a.equity, b.equity)
    assert a.total_costs == 0


def test_costs_reduce_equity_monotonically(small_bars):
    from quantsoc.market import to_wide
    opens = to_wide(small_bars, "open").iloc[:300]
    rng = np.random.default_rng(1)
    fc = pd.DataFrame(rng.normal(0, 0.001, (len(opens) - 1, 5)), index=opens.index[:-1], columns=opens.columns)
    cfg = PortfolioConfig(threshold=0)
    dec = decisions_from_forecasts(fc, cfg)
    prev = None
    for bps in (0, 1, 5, 10):
        r = run_backtest(opens, dec, cfg, cost_bps=bps)
        if prev is not None:
            assert r.equity.iloc[-1] < prev
        prev = r.equity.iloc[-1]


def test_sim_account_rejects_bad_inputs():
    acct = SimAccount(["A", "B"], 1000)
    with pytest.raises(ValueError):
        acct.rebalance(pd.Series({"A": 0.6, "B": 0.6}), pd.Series({"A": 10.0, "B": 10.0}), 0.0)
    with pytest.raises(ValueError):
        acct.rebalance(pd.Series({"A": 0.5, "B": 0.0}), pd.Series({"A": np.nan, "B": 10.0}), 0.0)


def test_full_budget_never_negative_cash():
    idx = pd.date_range("2025-01-06 09:30", periods=30, freq="5min", tz="America/New_York")
    rng = np.random.default_rng(3)
    opens = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (30, 2)), axis=0)), index=idx, columns=["A", "B"])
    tw = pd.DataFrame(rng.choice([0.0, 0.5, 1.0], size=(29, 2)), index=idx[:-1], columns=["A", "B"])
    tw = tw.div(tw.sum(axis=1).replace(0, 1), axis=0)
    res = run_backtest(opens, tw, PortfolioConfig(threshold=0, exposure_budget=1.0, per_asset_cap=1.0, cost_bps=25))
    assert (res.cash >= -1e-9).all()


def test_evaluate_strategy_benchmark_shares_timing(small_bars, small_dataset, small_model):
    from quantsoc.market import to_wide
    from quantsoc.modeling import predictions_wide
    fc = predictions_wide(small_model, small_dataset, "validation")
    opens = to_wide(small_bars, "open").loc[fc.index]
    res = evaluate_strategy(opens, fc, PortfolioConfig())
    assert set(res) == {"net", "gross", "benchmark"}
    assert res["net"].equity.index.equals(res["benchmark"].equity.index)
    assert res["gross"].equity.iloc[-1] >= res["net"].equity.iloc[-1]
    assert res["benchmark"].held.equals(res["net"].held)
