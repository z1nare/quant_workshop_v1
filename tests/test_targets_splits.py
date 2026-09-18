import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from quantsoc.features import FEATURE_NAMES
from quantsoc.modeling import (SPLIT_NAMES, build_dataset, clean_rows, fit_model, split_boundaries, tradable_target, xy,
                               score_forecasts)
from quantsoc.market import session_series, to_wide


def test_target_alignment_open_to_open(small_bars, small_dataset):
    opens = to_wide(small_bars, "open")
    d = small_dataset.set_index(["timestamp", "symbol"])
    i, sym = 25, "BOLT"
    t = opens.index[i]
    expected = opens[sym].iloc[i + 2] / opens[sym].iloc[i + 1] - 1
    assert np.isclose(d.loc[(t, sym), "target"], expected)


def test_target_does_not_cross_session_boundary(small_bars):
    opens, sess = to_wide(small_bars, "open"), session_series(small_bars)
    tgt = tradable_target(opens, sess)
    last_two = tgt.groupby(sess).tail(2)
    assert last_two.isna().all().all()
    rest = tgt.drop(last_two.index)
    assert rest.notna().all().all()


def test_splits_chronological_same_timestamps_all_assets(small_dataset):
    for s in SPLIT_NAMES:
        part = small_dataset[small_dataset.split == s]
        assert part.groupby("timestamp")["symbol"].nunique().eq(5).all()
    tr, va, te = (small_dataset[small_dataset.split == s]["timestamp"] for s in SPLIT_NAMES)
    assert tr.max() < va.min() < va.max() < te.min()
    n = small_dataset["timestamp"].nunique()
    assert abs(tr.nunique() / n - 0.6) < 0.01 and abs(va.nunique() / n - 0.2) < 0.01


def test_split_boundary_examples_removed(small_dataset):
    bounds = split_boundaries(small_dataset["timestamp"])
    ts = sorted(small_dataset["timestamp"].unique())
    train_end = bounds["train"][1]
    i = ts.index(train_end)
    crossing = small_dataset[small_dataset["timestamp"].isin([ts[i], ts[i - 1]])]
    assert crossing["crosses_split"].all()
    assert not clean_rows(small_dataset, "train")["timestamp"].isin([ts[i], ts[i - 1]]).any()
    ok = small_dataset[small_dataset["timestamp"] == ts[i - 2]]
    assert not ok["crosses_split"].any()


def test_preprocessing_fit_on_train_only(small_dataset):
    train = clean_rows(small_dataset, "train")
    model = fit_model(train)
    scaler = model.named_steps["standardscaler"]
    X, _ = xy(train)
    assert np.allclose(scaler.mean_, X.mean(axis=0))
    val = clean_rows(small_dataset, "validation")
    Xv, _ = xy(val)
    assert not np.allclose(scaler.mean_, np.vstack([X, Xv]).mean(axis=0))


def test_signal_learnable_over_tradable_target():
    from quantsoc.market import MarketParams, generate_market
    bars = generate_market(MarketParams())
    ds = build_dataset(bars)
    model = fit_model(clean_rows(ds, "train"))
    Xv, yv = xy(clean_rows(ds, "validation"))
    sc = score_forecasts(yv, model.predict(Xv))
    assert sc.correlation > 0.05 and sc.mse < sc.baseline_mse
    assert model.named_steps["linearregression"].coef_[FEATURE_NAMES.index("mom_3")] > 0


def test_zero_baseline_mse_is_mean_square(small_dataset, small_model):
    Xv, yv = xy(clean_rows(small_dataset, "validation"))
    sc = score_forecasts(yv, np.zeros_like(yv))
    assert np.isclose(sc.mse, sc.baseline_mse)
