"""Headless checks of the interactive controls (play/pause/step, re-execution cleanup, one-shot submit)
and of the Streamlit dashboard via streamlit's AppTest.  No browser, no network."""
import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from quantsoc import widgets
from quantsoc.backtest import run_replay
from quantsoc.market import to_wide
from quantsoc.modeling import predictions_wide
from quantsoc.portfolio import PortfolioConfig, allocate_weights
from quantsoc.state import StateStore
from quantsoc import reference_strategy as rs


@pytest.fixture(scope="module")
def trace(small_bars, small_dataset, small_model):
    fc = predictions_wide(small_model, small_dataset, "test")
    return run_replay(small_bars, rs.predict_returns, rs.allocate, small_model, PortfolioConfig(), start=fc.index[0], end=fc.index[60], lookback_bars=30)


pytestmark = pytest.mark.skipif(not widgets.WIDGETS_AVAILABLE, reason="ipywidgets not installed")


def _controls(box):
    row = box.children[0]
    play, slider, speed, back, fwd = row.children
    return play, slider, speed, back, fwd, box.children[1], box.children[2]


def test_replay_monitor_play_pause_step(trace):
    box = widgets.replay_monitor(trace)
    play, slider, speed, back, fwd, fig_out, html_out = _controls(box)
    assert slider.value == 0 and "simulated time" in html_out.value
    fwd.click(); fwd.click()
    assert slider.value == 2 and f"{trace.steps[2].decision_time:%Y-%m-%d %H:%M}" in html_out.value
    back.click()
    assert slider.value == 1
    slider.value = 40                                   # what Play does as it advances (jslink'd)
    assert f"{trace.steps[40].decision_time:%H:%M}" in html_out.value
    play.playing = True; play.playing = False           # pause leaves the position untouched
    assert slider.value == 40
    speed.value = 100
    assert play.interval == 100
    slider.value = len(trace.steps) - 1
    assert "last bar" in html_out.value or "fill at" in html_out.value


def test_rerunning_a_widget_cell_cleans_up_the_old_one(trace):
    first = widgets.replay_monitor(trace)
    play1 = _controls(first)[0]
    second = widgets.replay_monitor(trace)
    assert widgets._ACTIVE["replay_monitor"][0] is _controls(second)[0]
    assert play1 not in widgets._ACTIVE["replay_monitor"]
    # the old slider no longer drives anything: its observers were removed
    old_slider = _controls(first)[1]
    assert not old_slider._trait_notifiers.get("value", {}).get("change")


def test_portfolio_manager_and_cost_explorer_recalculate_on_release_only(small_bars, small_dataset, small_model):
    from quantsoc.backtest import decisions_from_forecasts
    fc = predictions_wide(small_model, small_dataset, "validation").dropna(how="all")
    scen = {"a": fc.iloc[10], "nan": pd.Series(np.nan, index=fc.columns)}
    box = widgets.portfolio_manager(scen, allocate=allocate_weights)
    scen_dd, thr = box.children[0].children
    assert thr.continuous_update is False
    scen_dd.value = "nan"
    import html
    info = html.unescape(box.children[3].value)
    assert "'positions': 0" in info and "'cash': 1.0" in info
    opens = to_wide(small_bars, "open").loc[fc.index]
    dec = decisions_from_forecasts(fc, PortfolioConfig())
    cbox = widgets.cost_explorer(opens, dec, PortfolioConfig())
    slider, apply = cbox.children[0].children
    before = cbox.children[1].value
    slider.value = 10.0
    assert cbox.children[1].value == before                # slider alone does nothing
    apply.click()
    assert cbox.children[1].value != before                # Apply recomputes


def test_paper_panel_submits_once_per_preview(small_bars, small_model, tmp_path):
    from quantsoc.broker import MockBroker
    from quantsoc.engine import TradingEngine
    stamps = sorted(small_bars.timestamp.unique())
    now = (pd.Timestamp(stamps[20]) + pd.Timedelta(minutes=5)).to_pydatetime()
    prices = small_bars[small_bars.timestamp == stamps[20]].set_index("symbol")["open"].to_dict()
    mock = MockBroker(cash=100_000, prices=prices, bars_df=small_bars, now=now, is_open=True)
    mock.next_close = now + timedelta(hours=3)
    eng = TradingEngine(mock, rs.predict_returns, rs.allocate, small_model, PortfolioConfig(threshold=-1), list(prices),
                        state=StateStore(tmp_path), mode="paper", sleep=lambda s: None, order_wait_seconds=0.1, poll_interval=0)
    panel = widgets.paper_panel(eng)
    preview, submit = panel.children[0].children
    assert submit.disabled and mock.submissions == []
    preview.click()
    assert not submit.disabled and mock.submissions == []
    submit.click()
    n = len(mock.submissions)
    assert n > 0 and submit.disabled
    submit.click()                                          # a lingering second click cannot submit again
    assert len(mock.submissions) == n
    # a new panel (re-run cell) invalidates the old preview token
    panel2 = widgets.paper_panel(eng)
    p2, s2 = panel2.children[0].children
    assert s2.disabled
    submit.click()
    assert len(mock.submissions) == n


def test_state_monitor_renders_snapshot(tmp_path):
    store = StateStore(tmp_path)
    store.write_snapshot({"mode": "replay", "broker": "replay", "cycles": 3,
                          "last_cycle": {"status": "traded", "reason": "1 order", "predictions": {"AURA": 0.001}, "target_weights": {"AURA": 0.3},
                                         "current_weights": {"AURA": 0.29}, "proposed_orders": [], "submitted_orders": []}}, equity=100_500.0)
    box = widgets.state_monitor(tmp_path)
    assert "TRADED" in box.children[1].value.upper() and "AURA" in box.children[1].value


def test_streamlit_dashboard_with_and_without_state(tmp_path, monkeypatch, repo_root):
    from streamlit.testing.v1 import AppTest
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(repo_root / "dashboard.py"), default_timeout=30).run()
    assert not at.exception and any("No trading state" in str(i.value) for i in at.info)
    store = StateStore(tmp_path / "state")
    store.write_snapshot({"mode": "replay", "broker": "replay", "strategy": "Test Fund", "cycles": 5, "config": {}, "symbol_map": {},
                          "last_cycle": {"status": "no_trade", "reason": "within band", "now": "2025-04-15T14:00:00-04:00",
                                         "account": {"equity": 100_100.0, "cash": 70_000.0, "buying_power": 70_000.0},
                                         "predictions": {"AURA": 0.0003, "BOLT": None}, "target_weights": {"AURA": 0.3, "BOLT": 0.0},
                                         "current_weights": {"AURA": 0.3, "BOLT": 0.0}, "positions": [{"symbol": "AURA", "qty": 10}],
                                         "proposed_orders": [], "submitted_orders": [], "skipped_symbols": {"BOLT": "stale data"}}},
                         equity=100_100.0)
    store.log("hello log")
    at = AppTest.from_file(str(repo_root / "dashboard.py"), default_timeout=30).run()
    assert not at.exception
    text = " ".join(str(m.value) for m in at.markdown)
    assert "NO_TRADE" in text and "OFFLINE REPLAY" in text
    assert any("stale data" in str(w.value) for w in at.warning)
