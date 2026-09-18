import numpy as np
import pandas as pd

from quantsoc.market import (BARS_PER_SESSION, MARKET_TZ, SYMBOLS, MarketParams, generate_market, load_market,
                             save_market, stress_params, validate_bars)


def test_generation_is_deterministic(small_params):
    a = generate_market(small_params)
    b = generate_market(small_params)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_differs(small_params):
    a = generate_market(small_params)
    b = generate_market(MarketParams(seed=small_params.seed + 1, n_sessions=small_params.n_sessions))
    assert not np.allclose(a["close"], b["close"])


def test_prices_positive_and_ohlc_valid(small_bars):
    px = small_bars[["open", "high", "low", "close"]]
    assert (px > 0).all().all()
    assert (small_bars["high"] >= small_bars[["open", "close"]].max(axis=1)).all()
    assert (small_bars["low"] <= small_bars[["open", "close"]].min(axis=1)).all()
    validate_bars(small_bars)


def test_session_structure_no_overnight_bars(small_bars, small_params):
    per_symbol = small_bars.groupby("symbol").size()
    assert (per_symbol == small_params.n_sessions * BARS_PER_SESSION).all()
    stamps = small_bars.drop_duplicates("timestamp")["timestamp"]
    local = stamps.dt.tz_convert(MARKET_TZ)
    minutes = local.dt.hour * 60 + local.dt.minute
    assert minutes.min() == 9 * 60 + 30 and minutes.max() == 15 * 60 + 55
    # consecutive bars inside a session are exactly 5 minutes apart; gaps only at session boundaries
    diffs = stamps.diff().dropna()
    assert set(diffs.unique()) <= {pd.Timedelta(minutes=5)} | set(d for d in diffs.unique() if d > pd.Timedelta(hours=17))
    assert sorted(small_bars["symbol"].unique()) == SYMBOLS


def test_open_is_not_the_previous_close(small_bars):
    aura = small_bars[small_bars.symbol == "AURA"].reset_index(drop=True)
    same_session = aura["session"].shift(1) == aura["session"]
    gap = (aura["open"] / aura["close"].shift(1) - 1)[same_session]
    assert (gap.abs() > 0).mean() > 0.99          # the next tradable price differs from the last close


def test_planted_momentum_is_present_in_closes(small_bars):
    from quantsoc.features import bar_returns, momentum
    from quantsoc.market import session_series, to_wide
    closes, sess = to_wide(small_bars), session_series(small_bars)
    r = bar_returns(closes, sess)
    m = momentum(closes, sess, 3)
    corrs = []
    for s in closes.columns:
        x, y = m[s].groupby(sess).shift(1), r[s]
        ok = x.notna() & y.notna()
        corrs.append(np.corrcoef(x[ok], y[ok])[0, 1])
    assert np.mean(corrs) > 0.05


def test_stress_scenario_has_higher_volatility_and_reversed_beta():
    p = stress_params()
    assert p.beta < 0 and all(s2 > s1 for s1, s2 in zip(MarketParams().sigma, p.sigma))
    bars = generate_market(MarketParams(**{**p.to_dict(), "n_sessions": 4}))
    validate_bars(bars)


def test_csv_round_trip(tmp_path, small_bars):
    path = tmp_path / "m.csv"
    save_market(small_bars, path)
    back = load_market(path)
    assert back.shape == small_bars.shape
    assert back["timestamp"].dt.tz is not None
    assert np.allclose(back["close"], small_bars["close"].round(6))
    assert (back["timestamp"] == small_bars["timestamp"]).all()


def test_checked_in_data_matches_generator(repo_root):
    from quantsoc.market import load_market
    bars = load_market(repo_root / "data" / "workshop_market.csv")
    fresh = generate_market(MarketParams())
    assert len(bars) == len(fresh) == 5 * 39 * 78
    assert np.allclose(bars["close"].to_numpy(), fresh["close"].round(6).to_numpy())
