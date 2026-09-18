import numpy as np
import pandas as pd

from quantsoc.features import FEATURE_NAMES, WARMUP_BARS, build_features, latest_features, bar_returns, momentum
from quantsoc.market import session_series, to_wide


def test_feature_order_fixed():
    assert FEATURE_NAMES == ["ret_1", "mom_3", "vol_12"]


def test_causality_future_changes_do_not_affect_past(small_bars):
    base = build_features(small_bars)
    cut = small_bars["timestamp"].sort_values().unique()[len(small_bars["timestamp"].unique()) // 2]
    tampered = small_bars.copy()
    future = tampered["timestamp"] > cut
    tampered.loc[future, ["open", "high", "low", "close"]] *= np.random.default_rng(0).uniform(0.5, 1.5, future.sum())[:, None]
    other = build_features(tampered)
    past = base["timestamp"] <= cut
    pd.testing.assert_frame_equal(base[past].reset_index(drop=True), other[past].reset_index(drop=True))


def test_features_only_use_completed_bars_of_same_session(small_bars):
    table = build_features(small_bars)
    first = table.groupby(["session", "symbol"]).head(1)
    assert first[FEATURE_NAMES].isna().all().all()          # nothing is inherited from the previous day
    warm = table.groupby(["session", "symbol"]).head(WARMUP_BARS)
    assert warm["vol_12"].isna().all()
    ready = table.groupby(["session", "symbol"]).nth(WARMUP_BARS)
    assert ready[FEATURE_NAMES].notna().all().all()


def test_feature_values_match_definitions(small_bars):
    closes, sess = to_wide(small_bars), session_series(small_bars)
    table = build_features(small_bars).set_index(["timestamp", "symbol"])
    sym = "CRUX"
    c = closes[sym]
    t = closes.index[20]                                    # inside the first session, past warm-up
    assert np.isclose(table.loc[(t, sym), "ret_1"], c.iloc[20] / c.iloc[19] - 1)
    assert np.isclose(table.loc[(t, sym), "mom_3"], c.iloc[20] / c.iloc[17] - 1)
    r = c.pct_change()
    assert np.isclose(table.loc[(t, sym), "vol_12"], r.iloc[9:21].std())


def test_latest_features_one_row_per_symbol(small_bars):
    latest = latest_features(small_bars)
    assert list(latest.index) == sorted(small_bars["symbol"].unique())
    assert list(latest.columns) == ["timestamp"] + FEATURE_NAMES
    assert (latest["timestamp"] == small_bars["timestamp"].max()).all()


def test_missing_bar_handled_without_inventing_data(small_bars):
    # remove one bar for one symbol: features still compute for the others, and NaN pattern stays sane
    t = small_bars["timestamp"].unique()[30]
    trimmed = small_bars[~((small_bars.timestamp == t) & (small_bars.symbol == "AURA"))]
    table = build_features(trimmed)
    row = table[(table.timestamp == t)]
    assert set(row.symbol) == {"BOLT", "CRUX", "DUNE", "ECHO"} or row["ret_1"][row.symbol == "AURA"].isna().all()
