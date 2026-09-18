import numpy as np
import pandas as pd
import pytest

from quantsoc.portfolio import PortfolioConfig, allocate_weights, check_weights, equal_weight

SYMS = ["AURA", "BOLT", "CRUX", "DUNE", "ECHO"]


def test_example_from_spec_two_qualify():
    cfg = PortfolioConfig(threshold=0.0, exposure_budget=0.9, per_asset_cap=0.3, max_positions=3)
    w = allocate_weights({"AURA": 0.001, "BOLT": 0.002, "CRUX": -0.001, "DUNE": 0.0, "ECHO": -0.002}, cfg)
    assert w["AURA"] == 0.3 and w["BOLT"] == 0.3 and w[["CRUX", "DUNE", "ECHO"]].eq(0).all()
    assert np.isclose(1 - w.sum(), 0.4)


def test_all_below_threshold_all_cash():
    w = allocate_weights({s: 0.0001 for s in SYMS}, PortfolioConfig(threshold=0.0002))
    assert w.eq(0).all() and check_weights(w, PortfolioConfig())["cash"] == 1.0


def test_top_three_when_more_qualify():
    w = allocate_weights({"AURA": 0.005, "BOLT": 0.004, "CRUX": 0.003, "DUNE": 0.002, "ECHO": 0.001}, PortfolioConfig(threshold=0))
    assert list(w[w > 0].index) == ["AURA", "BOLT", "CRUX"] and np.isclose(w.sum(), 0.9)


def test_non_finite_predictions_excluded():
    w = allocate_weights({"AURA": np.nan, "BOLT": np.inf, "CRUX": 0.003, "DUNE": -np.inf, "ECHO": 0.001}, PortfolioConfig(threshold=0))
    assert w["AURA"] == 0 and w["BOLT"] == 0 and w["DUNE"] == 0 and w["CRUX"] == 0.3 and w["ECHO"] == 0.3


def test_caps_bind():
    cfg = PortfolioConfig(threshold=0, exposure_budget=1.0, per_asset_cap=0.25, max_positions=3)
    w = allocate_weights({s: 0.01 for s in SYMS}, cfg)
    assert (w[w > 0] == 0.25).all() and np.isclose(w.sum(), 0.75)
    cfg2 = PortfolioConfig(threshold=0, exposure_budget=0.6, per_asset_cap=0.5, max_positions=3)
    w2 = allocate_weights({s: 0.01 for s in SYMS}, cfg2)
    assert np.allclose(w2[w2 > 0], 0.2)


def test_deterministic_tie_breaking():
    w = allocate_weights({"ECHO": 0.001, "DUNE": 0.001, "CRUX": 0.001, "BOLT": 0.001, "AURA": 0.001}, PortfolioConfig(threshold=0))
    assert sorted(w[w > 0].index) == ["AURA", "BOLT", "CRUX"]
    w2 = allocate_weights(pd.Series({"AURA": 0.001, "BOLT": 0.001, "CRUX": 0.001, "DUNE": 0.001, "ECHO": 0.001}), PortfolioConfig(threshold=0))
    pd.testing.assert_series_equal(w.sort_index(), w2.sort_index(), check_names=False)


def test_threshold_changes_selection():
    preds = {"AURA": 0.0005, "BOLT": 0.0003, "CRUX": 0.0001, "DUNE": -0.0001, "ECHO": 0.0}
    assert (allocate_weights(preds, PortfolioConfig(threshold=0.0002)) > 0).sum() == 2
    assert (allocate_weights(preds, PortfolioConfig(threshold=0.0004)) > 0).sum() == 1
    assert (allocate_weights(preds, PortfolioConfig(threshold=0.001)) > 0).sum() == 0


@pytest.mark.parametrize("bad", [dict(exposure_budget=1.2), dict(per_asset_cap=-0.1), dict(max_positions=0),
                                 dict(threshold=float("nan")), dict(cost_bps=-1)])
def test_invalid_config_rejected(bad):
    with pytest.raises(ValueError):
        PortfolioConfig(**bad).validate()


def test_duplicate_symbols_rejected():
    with pytest.raises(ValueError):
        allocate_weights(pd.Series([0.1, 0.2], index=["AURA", "AURA"]), PortfolioConfig())


def test_equal_weight_benchmark_respects_caps():
    w = equal_weight(SYMS, PortfolioConfig(exposure_budget=0.9, per_asset_cap=0.3))
    assert np.allclose(w, 0.18) and check_weights(w, PortfolioConfig(max_positions=5))["ok"]
