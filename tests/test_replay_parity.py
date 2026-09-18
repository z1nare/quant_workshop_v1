import numpy as np
import pandas as pd

from quantsoc.backtest import evaluate_strategy, run_replay
from quantsoc.market import to_wide
from quantsoc.modeling import predictions_wide
from quantsoc.portfolio import PortfolioConfig
from quantsoc import reference_strategy as rs


def test_replay_matches_vectorised_backtest(small_bars, small_dataset, small_model):
    cfg = PortfolioConfig(cost_bps=3)
    fc = predictions_wide(small_model, small_dataset, "test")
    opens = to_wide(small_bars, "open")
    start, end = fc.index[0], fc.index[-1]
    vec = evaluate_strategy(opens.loc[start:end], fc, cfg)["net"]
    trace = run_replay(small_bars, rs.predict_returns, rs.allocate, small_model, cfg, start=start, end=end, lookback_bars=30)
    rep = trace.equity
    common = vec.equity.index.intersection(rep.index)
    assert len(common) > 50
    assert np.allclose(vec.equity.loc[common].to_numpy(), rep.loc[common].to_numpy(), rtol=1e-10)


def test_replay_reveals_history_progressively(small_bars, small_model):
    seen = []

    def spy_predict(history, model):
        seen.append(history["timestamp"].max())
        return rs.predict_returns(history, model)

    stamps = sorted(small_bars["timestamp"].unique())
    trace = run_replay(small_bars, spy_predict, rs.allocate, small_model, PortfolioConfig(), start=stamps[20], end=stamps[40], lookback_bars=15)
    assert [pd.Timestamp(s) for s in seen] == [st.decision_time for st in trace.steps]
    assert all(s <= st.decision_time for s, st in zip(seen, trace.steps))


def test_replay_lookback_limits_history(small_bars, small_model):
    lens = []
    def spy(history, model):
        lens.append(history["timestamp"].nunique()); return rs.predict_returns(history, model)
    stamps = sorted(small_bars["timestamp"].unique())
    run_replay(small_bars, spy, rs.allocate, small_model, PortfolioConfig(), start=stamps[50], end=stamps[55], lookback_bars=20)
    assert max(lens) <= 20
