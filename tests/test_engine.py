"""Engine behaviour against the scriptable MockBroker.  No network, no credentials."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from quantsoc.broker import MockBroker, UncertainSubmission, BrokerError, completed_bars_only, OrderInfo
from quantsoc.engine import TradingEngine, make_client_order_id
from quantsoc.portfolio import PortfolioConfig
from quantsoc.state import StateStore
from quantsoc import reference_strategy as rs

SYMS = ["AURA", "BOLT", "CRUX", "DUNE", "ECHO"]
MAP = {"AURA": "SPY", "BOLT": "QQQ", "CRUX": "IWM", "DUNE": "DIA", "ECHO": "XLK"}


def _now_after(bars, n=20):
    """A 'now' that makes the first n bars of the first session completed."""
    stamps = sorted(bars["timestamp"].unique())
    return (pd.Timestamp(stamps[n - 1]) + pd.Timedelta(minutes=5)).to_pydatetime()


@pytest.fixture
def mock(small_bars):
    real = small_bars.copy()
    real["symbol"] = real["symbol"].map(MAP)
    now = _now_after(small_bars, 20)
    prices = small_bars[small_bars.timestamp == sorted(small_bars.timestamp.unique())[20]].set_index("symbol")["open"]
    m = MockBroker(cash=100_000, prices={MAP[s]: float(prices[s]) for s in SYMS}, bars_df=real, now=now, is_open=True)
    m.next_close = now + timedelta(hours=3)
    return m


def make_engine(broker, tmp_path, small_model, mode="paper", **kw):
    return TradingEngine(broker, rs.predict_returns, rs.allocate, small_model, PortfolioConfig(threshold=-1.0),
                         SYMS, symbol_map=MAP, state=StateStore(tmp_path / "state"), mode=mode, sleep=lambda s: None,
                         order_wait_seconds=0.1, poll_interval=0.0, **kw)


def fixed_predict(history, model):
    return pd.Series({"AURA": 0.002, "BOLT": 0.001, "CRUX": 0.0005, "DUNE": -0.001, "ECHO": np.nan})


def test_paper_preview_never_submits(mock, tmp_path, small_model):
    eng = make_engine(mock, tmp_path, small_model, mode="paper-preview")
    res = eng.run_cycle(submit=True)                  # even an explicit True cannot submit in preview mode
    assert res.status == "preview" and res.proposed_orders and mock.submissions == []


def test_run_all_default_is_preview_in_paper_mode(mock, tmp_path, small_model):
    eng = make_engine(mock, tmp_path, small_model, mode="paper")
    res = eng.run_cycle()                             # submit=None -> no submission in paper mode
    assert res.status == "preview" and mock.submissions == []


def test_paper_submit_buys_and_no_duplicates_on_rerun(mock, tmp_path, small_model):
    eng = make_engine(mock, tmp_path, small_model)
    eng.predict_returns = fixed_predict
    res = eng.run_cycle(submit=True)
    assert res.status == "traded"
    assert sorted(o["symbol"] for o in mock.submissions) == ["IWM", "QQQ", "SPY"]   # top 3 above threshold; ECHO NaN excluded
    n = len(mock.submissions)
    res2 = eng.run_cycle(submit=True)                 # same bar, positions now match -> nothing new
    assert len(mock.submissions) == n and res2.status in ("no_trade", "traded")
    cids = {o["client_order_id"] for o in mock.submissions}
    assert len(cids) == n and all(c.startswith("qsw-") for c in cids)


def test_restart_reconciliation_uses_registry_and_broker_lookup(mock, tmp_path, small_model):
    eng = make_engine(mock, tmp_path, small_model)
    eng.predict_returns = fixed_predict
    eng.run_cycle(submit=True)
    n = len(mock.submissions)
    # simulate a restart: new engine, same state dir, positions rolled back so it WANTS to trade again
    mock.holdings = {}; mock.cash = 100_000
    eng2 = make_engine(mock, tmp_path, small_model)
    eng2.predict_returns = fixed_predict
    res = eng2.run_cycle(submit=True)
    assert len(mock.submissions) == n                 # client ids already in the registry -> not resubmitted
    assert any("already in the local order registry" in l for l in res.log)
    # and with an EMPTY registry the broker lookup by client id still prevents duplicates
    eng3 = make_engine(mock, tmp_path / "fresh", small_model)
    eng3.predict_returns = fixed_predict
    res3 = eng3.run_cycle(submit=True)
    assert len(mock.submissions) == n
    assert any("already exists at the broker" in l for l in res3.log)


def test_uncertain_submission_reconciled_before_retry(mock, tmp_path, small_model):
    eng = make_engine(mock, tmp_path, small_model)
    eng.predict_returns = fixed_predict
    mock.uncertain_symbols = {"SPY"}                  # accepted, then timeout
    res = eng.run_cycle(submit=True)
    spy = [o for o in mock.submissions if o["symbol"] == "SPY"]
    assert len(spy) == 1                              # reconciled by client id, NOT resubmitted
    assert any("found at broker" in l for l in res.log)


def test_uncertain_submission_not_accepted_is_retried_once(mock, tmp_path, small_model):
    eng = make_engine(mock, tmp_path, small_model)
    eng.predict_returns = fixed_predict
    mock.fail_next_submit = True                      # timeout BEFORE acceptance
    res = eng.run_cycle(submit=True)
    assert any("retrying once" in l for l in res.log)
    assert res.status == "traded" and len(mock.submissions) == 3


def test_rejected_order_logged_and_others_proceed(mock, tmp_path, small_model):
    eng = make_engine(mock, tmp_path, small_model)
    eng.predict_returns = fixed_predict
    mock.reject_symbols = {"QQQ"}
    res = eng.run_cycle(submit=True)
    assert any("REJECTED" in l for l in res.log)
    assert {o["symbol"] for o in mock.submissions} == {"SPY", "QQQ", "IWM"}
    assert set(mock.holdings) == {"SPY", "IWM"}
    assert eng.state.orders()[make_client_order_id(pd.Timestamp(res.decision_bar), "QQQ", "buy")]["status"] == "rejected"


def test_partial_fill_replanned_next_cycle(mock, tmp_path, small_model):
    eng = make_engine(mock, tmp_path, small_model)
    eng.predict_returns = fixed_predict
    mock.partial_fill = {"SPY": 0.5}
    res = eng.run_cycle(submit=True)
    assert any("PARTIAL" in l for l in res.log)
    half = mock.holdings["SPY"]
    # next bar: engine sees the smaller SPY position and tops it up
    mock.partial_fill = {}
    for o in list(mock.orders.values()):
        if o.status == "partially_filled":
            o.status = "canceled"                      # broker closed the remainder
    mock.now = mock.now + timedelta(minutes=5)
    mock.bars_df = mock.bars_df                        # one more bar is now complete
    res2 = eng.run_cycle(submit=True)
    assert mock.holdings["SPY"] > half * 1.5


def test_sells_before_buys_and_buying_power_respected(mock, tmp_path, small_model):
    eng = make_engine(mock, tmp_path, small_model)
    eng.predict_returns = fixed_predict
    eng.run_cycle(submit=True)
    # now flip the view: sell SPY/QQQ/IWM, buy DUNE/ECHO with all the cash tied up
    eng.predict_returns = lambda h, m: pd.Series({"AURA": -1, "BOLT": -1, "CRUX": -1, "DUNE": 0.002, "ECHO": 0.001})
    mock.now = mock.now + timedelta(minutes=5)
    before = len(mock.submissions)
    res = eng.run_cycle(submit=True)
    new = mock.submissions[before:]
    sides = [o["side"] for o in new]
    assert sides[: sides.count("sell")] == ["sell"] * sides.count("sell")     # all sells first
    assert res.status == "traded" and mock.cash >= -1e-6
    assert set(mock.holdings) - {s for s, q in mock.holdings.items() if abs(q) < 1e-9} == {"DIA", "XLK"}


def test_insufficient_buying_power_scales_buys(mock, tmp_path, small_model):
    mock.cash = 20_000                                 # engine thinks equity 20k; wants 3 x 30% = 18k -> fits
    eng = make_engine(mock, tmp_path, small_model)
    eng.predict_returns = fixed_predict
    mock.buying_power_multiplier = 0.5                 # broker only allows 10k
    res = eng.run_cycle(submit=True)
    assert any("insufficient buying power" in l for l in res.log)
    assert res.status == "traded" and mock.cash >= -1e-6
    assert sum(o["notional"] for o in mock.submissions) <= 10_000 + 1e-6


def test_market_closed_skips(mock, tmp_path, small_model):
    mock.is_open = False; mock.next_open = mock.now + timedelta(hours=16)
    eng = make_engine(mock, tmp_path, small_model)
    res = eng.run_cycle(submit=True)
    assert res.status == "skipped" and "closed" in res.reason and mock.submissions == []


def test_close_buffer_skips(mock, tmp_path, small_model):
    mock.next_close = mock.now + timedelta(minutes=3)
    eng = make_engine(mock, tmp_path, small_model)
    res = eng.run_cycle(submit=True)
    assert res.status == "skipped" and "close buffer" in res.reason


def test_stale_data_skips(mock, tmp_path, small_model):
    mock.now = mock.now + timedelta(minutes=45)        # bars stop 45 min before 'now'
    mock.bars_df = mock.bars_df[mock.bars_df.timestamp < pd.Timestamp(mock.now) - pd.Timedelta(minutes=40)]
    eng = make_engine(mock, tmp_path, small_model)
    res = eng.run_cycle(submit=True)
    assert res.status == "skipped" and "stale" in res.reason and mock.submissions == []


def test_warm_up_skips(mock, tmp_path, small_model, small_bars):
    mock.now = _now_after(small_bars, 5)               # only 5 completed bars in the session
    eng = make_engine(mock, tmp_path, small_model)
    res = eng.run_cycle(submit=True)
    assert res.status == "skipped" and "warm-up" in res.reason


def test_invalid_forecasts_all_nan_skips(mock, tmp_path, small_model):
    eng = make_engine(mock, tmp_path, small_model)
    eng.predict_returns = lambda h, m: pd.Series({s: np.nan for s in SYMS})
    res = eng.run_cycle(submit=True)
    assert res.status == "skipped" and mock.submissions == []


def test_student_bug_does_not_crash_runner(mock, tmp_path, small_model):
    eng = make_engine(mock, tmp_path, small_model)
    eng.predict_returns = lambda h, m: 1 / 0
    res = eng.run_cycle(submit=True)
    assert res.status == "error" and "ZeroDivisionError" in res.reason


def test_allocation_violating_constraints_blocked(mock, tmp_path, small_model):
    eng = make_engine(mock, tmp_path, small_model)
    eng.allocate = lambda p, c: pd.Series({s: 0.5 for s in SYMS})       # 250% exposure
    res = eng.run_cycle(submit=True)
    assert res.status == "error" and "constraints" in res.reason and mock.submissions == []


def test_stale_open_orders_cancelled_and_cycle_waits(mock, tmp_path, small_model):
    eng = make_engine(mock, tmp_path, small_model)
    eng.predict_returns = fixed_predict
    mock.pending_symbols = {"SPY"}
    eng.run_cycle(submit=True)                         # SPY stays 'accepted'
    mock.pending_symbols = set()
    mock.now = mock.now + timedelta(minutes=5)
    res = eng.run_cycle(submit=True)
    assert res.status == "skipped" and "stale open order" in res.reason and mock.cancelled


def test_same_bar_open_order_not_resubmitted(mock, tmp_path, small_model):
    eng = make_engine(mock, tmp_path, small_model)
    eng.predict_returns = fixed_predict
    mock.pending_symbols = {"SPY"}
    eng.run_cycle(submit=True)
    n = len(mock.submissions)
    eng2 = make_engine(mock, tmp_path / "other", small_model)   # empty registry, same bar
    eng2.predict_returns = fixed_predict
    res = eng2.run_cycle(submit=True)
    assert len(mock.submissions) == n
    assert any("already open at the broker" in l for l in res.log)


def test_non_fractionable_uses_whole_shares(mock, tmp_path, small_model):
    mock.non_fractionable = {"SPY"}
    eng = make_engine(mock, tmp_path, small_model)
    eng.predict_returns = fixed_predict
    eng.run_cycle(submit=True)
    spy = [o for o in mock.submissions if o["symbol"] == "SPY"][0]
    assert spy["notional"] is None and float(spy["qty"]).is_integer()


def test_validate_instruments_reports_untradable(mock, tmp_path, small_model):
    mock.untradable = {"XLK"}
    eng = make_engine(mock, tmp_path, small_model)
    rep = eng.validate_instruments()
    assert "ERROR" in rep["ECHO"] and "tradable=True" in rep["AURA"]


def test_completed_bars_only_drops_forming_bar_and_extended_hours():
    ts = pd.date_range("2025-03-03 09:00", periods=100, freq="5min", tz="America/New_York")
    df = pd.DataFrame({"timestamp": ts, "symbol": "SPY", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1})
    now = pd.Timestamp("2025-03-03 10:02", tz="America/New_York")
    out = completed_bars_only(df, now, 100)
    assert out["timestamp"].min() == pd.Timestamp("2025-03-03 09:30", tz="America/New_York")
    assert out["timestamp"].max() == pd.Timestamp("2025-03-03 09:55", tz="America/New_York")   # 10:00 bar still forming
    assert (out["session"] == "2025-03-03").all()


def test_state_snapshot_written_every_cycle(mock, tmp_path, small_model):
    eng = make_engine(mock, tmp_path, small_model, mode="paper-preview")
    eng.run_cycle()
    snap = eng.state.read_snapshot()
    assert snap["mode"] == "paper-preview" and snap["last_cycle"]["status"] == "preview"
    assert eng.state.tail_log(3)
