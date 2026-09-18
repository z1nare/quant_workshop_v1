import numpy as np
import pandas as pd
import pytest

from quantsoc.exercises import ExerciseIncomplete, ExerciseTracker, SPECS
from quantsoc.market import session_series, to_wide
from quantsoc.modeling import clean_rows, xy
from quantsoc.portfolio import PortfolioConfig
from quantsoc import solutions as S


@pytest.fixture
def ex(small_bars, small_dataset, tmp_path):
    t = ExerciseTracker(auto_reference=False)
    closes, sess = to_wide(small_bars), session_series(small_bars)
    t.set_context(1, closes=closes, sessions=sess)
    t.set_context(2, closes=closes, sessions=sess)
    Xtr, ytr = xy(clean_rows(small_dataset, "train")); Xv, yv = xy(clean_rows(small_dataset, "validation"))
    t.set_context(3, X_train=Xtr, y_train=ytr, X_val=Xv, y_val=yv)
    for k6 in ("6a", "6b"):
        t.set_context(k6, strategy_path=tmp_path / "strategy.py", model=S.fit_linear_model(Xtr, ytr),
                      histories=[small_bars[small_bars.timestamp <= small_bars.timestamp.unique()[k]] for k in (5, 30, 200)])
    return t


def test_wrong_answers_rejected(ex, capsys):
    closes, sess = ex.contexts["1"]["closes"], ex.contexts["1"]["sessions"]
    assert ex.attempt(1, lambda c, s: c.pct_change(), closes, sess) is None          # crosses sessions
    assert ex.attempt(1, lambda c, s: c.diff(), closes, sess) is None                 # not a return
    assert ex.attempt(2, lambda c, s: c.groupby(s).pct_change(fill_method=None), closes, sess) is None   # 1-bar, not 3
    assert ex.states["1"].status == "failed" and not ex.passed(1)


def test_correct_answers_pass(ex):
    closes, sess = ex.contexts["1"]["closes"], ex.contexts["1"]["sessions"]
    assert ex.attempt(1, S.compute_bar_returns, closes, sess) is not None and ex.passed(1)
    assert ex.attempt(2, lambda c, s: c / c.groupby(s).shift(3) - 1, closes, sess) is not None and ex.passed(2)


def test_not_started_and_errors_reported_not_raised(ex, capsys):
    def todo(c, s):
        raise NotImplementedError("later")
    assert ex.attempt(1, todo, None, None) is None and ex.states["1"].status == "not_started"
    assert ex.attempt(1, lambda c, s: 1 / 0, None, None) is None and ex.states["1"].status == "error"
    with pytest.raises(ExerciseIncomplete):
        ex.get(1)


def test_model_check_detects_leakage(ex):
    c = ex.contexts["3"]
    leaky = S.fit_linear_model(np.vstack([c["X_train"], c["X_val"]]), np.concatenate([c["y_train"], c["y_val"]]))
    assert ex.attempt(3, lambda: leaky) is None and "validation" in ex.states["3"].message
    from sklearn.linear_model import LinearRegression
    plain = LinearRegression().fit(c["X_train"], c["y_train"])
    assert ex.attempt(3, lambda: plain) is not None            # unscaled OLS gives the same forecasts


def test_allocation_check_battery(ex):
    def broken(preds, cfg):
        w = S.my_allocate(preds, cfg); return w / max(w.sum(), 1e-9) * cfg.exposure_budget    # renormalises
    assert ex.attempt(4, lambda: broken) is None
    assert ex.attempt(4, lambda: S.my_allocate) is not None


def test_config_check(ex):
    assert ex.attempt(5, lambda: (PortfolioConfig(threshold=0.0003), "short")) is None
    assert ex.attempt(5, lambda: (PortfolioConfig(exposure_budget=2.0), "a long enough justification sentence here")) is None
    assert ex.attempt(5, lambda: (PortfolioConfig(threshold=0.0003), S.REFERENCE_JUSTIFICATION)) is not None


def test_strategy_check_and_reference_route(ex, tmp_path):
    from quantsoc.artifacts import load_strategy_module
    p = tmp_path / "strategy.py"
    p.write_text(S.STRATEGY_PY.replace("preds[latest.index[ready]] = model.predict(X)", "preds[latest.index[ready]] = 0.0"), encoding="utf-8")
    assert ex.attempt("6a", lambda: load_strategy_module(p, "bad_strategy")) is None
    assert ex.attempt("6b", lambda: load_strategy_module(p, "bad_strategy2")) is not None   # allocate untouched -> ok
    mod = ex.use_reference("6a")
    assert ex.states["6a"].status == "reference" and callable(mod.predict_returns) and ex.any_reference
    assert "reference" in ex.summary().loc["6a", "status"]


def test_auto_reference_env(monkeypatch, small_bars):
    monkeypatch.setenv("QSW_AUTO_REFERENCE", "1")
    t = ExerciseTracker()
    t.set_context(1, closes=to_wide(small_bars), sessions=session_series(small_bars))
    def todo(c, s):
        raise NotImplementedError
    out = t.attempt(1, todo, None, None)
    assert out is not None and t.states["1"].status == "reference"


def test_every_exercise_has_hints_and_solution():
    for k, spec in SPECS.items():
        assert len(spec.hints) >= 3 and len(spec.solution_source()) > 20
