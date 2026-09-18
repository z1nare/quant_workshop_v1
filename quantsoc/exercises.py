"""Exercise tracker: attempts, checks, progressive hints and LABELLED reference solutions.

Design rules
* `attempt()` never raises: it reports a clear status (not started / error / failed / passed).
* `get()` raises ExerciseIncomplete with a short, friendly message, so later cells cannot
  silently run on missing work, and the untouched student notebook never reports completion.
* `use_reference()` is the explicit recovery route.  It prints a banner, records the fact,
  and the export manifest carries that label.  Nothing is swapped in silently.
* Setting the environment variable QSW_AUTO_REFERENCE=1 makes attempt() fall back to the
  reference automatically (used by CI to exercise the recovery route from a clean kernel).
"""
from __future__ import annotations

import inspect
import os
import textwrap
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from . import features as F
from . import solutions as S
from .portfolio import PortfolioConfig, allocate_weights


class ExerciseIncomplete(Exception):
    """Raised by ExerciseTracker.get when an exercise has not been passed."""


@dataclass
class ExerciseSpec:
    key: str
    title: str
    hints: List[str]
    reference: Callable                 # (context) -> result
    checker: Callable                   # (result, context) -> (ok, message)
    solution_source: Callable           # () -> str


@dataclass
class ExerciseState:
    status: str = "not_started"         # not_started | error | failed | passed | reference
    message: str = ""
    result: Any = None
    hints_shown: int = 0


def _src(fn) -> str:
    return textwrap.dedent(inspect.getsource(fn))


# --------------------------------------------------------------------------- checkers

def _frame_close(a, b, atol=1e-10) -> bool:
    if not isinstance(a, pd.DataFrame):
        return False
    try:
        a = a.reindex(index=b.index, columns=b.columns)
    except Exception:
        return False
    an, bn = a.isna().to_numpy(), b.isna().to_numpy()
    if not np.array_equal(an, bn):
        return False
    return np.allclose(a.to_numpy()[~an], b.to_numpy()[~bn], atol=atol, rtol=1e-8)


def check_ex1(result, ctx):
    ref = F.bar_returns(ctx["closes"], ctx["sessions"])
    if result is None:
        return False, "The function returned None. Return the DataFrame of returns."
    if not isinstance(result, pd.DataFrame):
        return False, f"Expected a DataFrame (timestamp x symbol), got {type(result).__name__}."
    if result.shape != ref.shape:
        return False, f"Shape {result.shape} differs from the expected {ref.shape}."
    if result.isna().sum().sum() < ref.isna().sum().sum():
        return False, ("Too few NaNs: the first bar of EVERY session should be NaN (there is no previous bar in "
                       "that session). Did you compute returns across the overnight gap?")
    if not _frame_close(result, ref):
        return False, "Values differ from close[t]/close[t-1] - 1 within each session. Check the direction and the -1."
    return True, "Per-bar returns match the library implementation (session boundaries handled)."


def check_ex2(result, ctx):
    ref = F.momentum(ctx["closes"], ctx["sessions"], 3)
    if result is None or not isinstance(result, pd.DataFrame):
        return False, "Return a DataFrame (timestamp x symbol) of 3-bar momentum."
    if result.shape != ref.shape:
        return False, f"Shape {result.shape} differs from the expected {ref.shape}."
    one_bar = F.momentum(ctx["closes"], ctx["sessions"], 1)
    if _frame_close(result, one_bar):
        return False, "That is the 1-bar return. Momentum here is the return over the last THREE completed bars."
    if result.isna().sum().sum() < ref.isna().sum().sum():
        return False, "Too few NaNs: the first three bars of every session cannot have 3-bar momentum."
    if not _frame_close(result, ref):
        return False, "Values differ from close[t]/close[t-3] - 1 within each session."
    return True, "3-bar momentum matches the library implementation."


def check_ex3(result, ctx):
    X_train, y_train, X_val = ctx["X_train"], ctx["y_train"], ctx["X_val"]
    if result is None or not hasattr(result, "predict"):
        return False, "Return the fitted model object (it needs a .predict method)."
    try:
        pred = np.asarray(result.predict(X_val), dtype=float).ravel()
    except Exception as e:
        return False, f"model.predict(X_val) failed: {type(e).__name__}: {e}"
    if pred.shape[0] != X_val.shape[0]:
        return False, "predict returned the wrong number of values."
    ref = S.fit_linear_model(X_train, y_train)
    ref_pred = ref.predict(X_val)
    if np.allclose(pred, ref_pred, atol=1e-9):
        return True, "Model reproduces the reference fit on the training data (validation forecasts match)."
    # allow a plain LinearRegression without scaling: same predictions up to numerical noise
    if np.allclose(pred, ref_pred, atol=1e-7):
        return True, "Model matches the reference fit (tiny numerical differences are fine)."
    leak = S.fit_linear_model(np.vstack([X_train, X_val]), np.concatenate([y_train, ctx["y_val"]]))
    if np.allclose(pred, leak.predict(X_val), atol=1e-7):
        return False, "These forecasts match a model fitted on train AND validation. Fit on the training rows only."
    return False, ("Validation forecasts differ from a linear model fitted on the training rows. "
                   "Use LinearRegression (optionally after StandardScaler) fitted with X_train, y_train.")


ALLOC_CASES = [
    ("all below threshold", {"AURA": 0.0001, "BOLT": -0.001, "CRUX": 0.0, "DUNE": 0.00005, "ECHO": -0.0002}),
    ("two qualify", {"AURA": 0.002, "BOLT": 0.0001, "CRUX": 0.001, "DUNE": -0.001, "ECHO": 0.0}),
    ("four qualify", {"AURA": 0.002, "BOLT": 0.0015, "CRUX": 0.001, "DUNE": 0.0009, "ECHO": -0.001}),
    ("non-finite", {"AURA": float("nan"), "BOLT": 0.0015, "CRUX": float("inf"), "DUNE": 0.0009, "ECHO": 0.0006}),
    ("ties", {"AURA": 0.001, "BOLT": 0.001, "CRUX": 0.001, "DUNE": 0.001, "ECHO": 0.0005}),
]
ALLOC_CONFIGS = [PortfolioConfig(), PortfolioConfig(threshold=0.0005, max_positions=2, exposure_budget=0.5, per_asset_cap=0.2),
                 PortfolioConfig(per_asset_cap=0.5, exposure_budget=1.0)]


def check_allocate_fn(fn):
    for cfg in ALLOC_CONFIGS:
        for name, preds in ALLOC_CASES:
            ref = allocate_weights(preds, cfg)
            try:
                got = fn(pd.Series(preds, dtype=float), cfg)
            except Exception as e:
                return False, f"case '{name}' raised {type(e).__name__}: {e}"
            try:
                got = pd.Series(got, dtype=float).reindex(ref.index)
            except Exception:
                return False, f"case '{name}': return a Series of weights indexed by symbol."
            if got.isna().any():
                return False, f"case '{name}': weights contain NaN (unselected assets must be 0, not NaN)."
            if not np.allclose(got.to_numpy(), ref.to_numpy(), atol=1e-12):
                return False, (f"case '{name}' with threshold={cfg.threshold}, max={cfg.max_positions}, "
                               f"budget={cfg.exposure_budget}, cap={cfg.per_asset_cap}: expected "
                               f"{ref.to_dict()} but got {got.to_dict()}")
    return True, "Allocation matches the rule on every edge case (threshold, caps, NaN, ties, fewer than N)."


def check_ex4(result, ctx):
    if not callable(result):
        return False, "Return the allocation FUNCTION itself (my_allocate), not its output."
    return check_allocate_fn(result)


def check_ex5(result, ctx):
    if not isinstance(result, tuple) or len(result) != 2:
        return False, "Return a tuple: (PortfolioConfig, justification string)."
    cfg, why = result
    if not isinstance(cfg, PortfolioConfig):
        return False, "The first element must be a PortfolioConfig."
    try:
        cfg.validate()
    except ValueError as e:
        return False, f"Configuration is invalid: {e}"
    if not isinstance(why, str) or len(why.strip()) < 30:
        return False, "Write at least one full sentence explaining WHY you chose this configuration (validation evidence)."
    if cfg.max_positions > 5:
        return False, "max_positions cannot exceed the number of assets (5)."
    return True, f"Frozen: {cfg.to_dict()}"


def _module_ok(result):
    if result is None:
        return False, "strategy.py could not be imported (check the cell output above for a syntax error)."
    return True, ""


def check_ex6a(result, ctx):
    """result = imported strategy module; ctx has histories (list of long DataFrames) and model."""
    from . import reference_strategy as R
    ok, msg = _module_ok(result)
    if not ok:
        return ok, msg
    if not callable(getattr(result, "predict_returns", None)):
        return False, "strategy.py must define `predict_returns(history, model)`."
    for hist in ctx["histories"]:
        try:
            got = pd.Series(result.predict_returns(hist, ctx["model"]), dtype=float)
        except NotImplementedError:
            return False, "predict_returns is not implemented yet (it still raises NotImplementedError)."
        except Exception as e:
            return False, f"predict_returns raised {type(e).__name__}: {e}"
        ref = R.predict_returns(hist, ctx["model"])
        got = got.reindex(ref.index)
        if not np.array_equal(got.isna().to_numpy(), ref.isna().to_numpy()):
            return False, ("NaN pattern differs from the research pipeline: symbols whose features are not ready "
                           "(warm-up) must be NaN, every other symbol must get a forecast.")
        okm = ~ref.isna().to_numpy()
        if okm.any() and not np.allclose(got.to_numpy()[okm], ref.to_numpy()[okm], atol=1e-10):
            return False, ("forecasts differ from the research pipeline for the same history. "
                           "Use latest_features(history) and model.predict on the FEATURE_NAMES columns.")
    return True, "predict_returns reproduces the research forecasts for identical histories."


def check_ex6b(result, ctx):
    ok, msg = _module_ok(result)
    if not ok:
        return ok, msg
    if not callable(getattr(result, "allocate", None)):
        return False, "strategy.py must define `allocate(predictions, config)` (run the 6b cell)."
    try:
        result.allocate(pd.Series(ALLOC_CASES[0][1], dtype=float), PortfolioConfig())
    except NotImplementedError:
        return False, "allocate is not implemented yet (it still raises NotImplementedError)."
    except Exception:
        pass
    ok, msg = check_allocate_fn(result.allocate)
    return ok, ("allocate: " + msg) if not ok else "allocate in strategy.py matches the rule on every edge case."


# --------------------------------------------------------------------------- registry

def _ref1(ctx): return S.compute_bar_returns(ctx["closes"], ctx["sessions"])
def _ref2(ctx): return S.compute_momentum(ctx["closes"], ctx["sessions"], 3)
def _ref3(ctx): return S.fit_linear_model(ctx["X_train"], ctx["y_train"])
def _ref4(ctx): return S.my_allocate
def _ref5(ctx): return (S.REFERENCE_CONFIG, S.REFERENCE_JUSTIFICATION)
def _ref6(ctx):
    """Write the complete reference strategy.py (both functions) and import it."""
    from .artifacts import load_strategy_module
    path = ctx["strategy_path"]
    path.write_text(S.STRATEGY_PY, encoding="utf-8")
    return load_strategy_module(path, name="strategy")


SPECS: Dict[str, ExerciseSpec] = {
    "1": ExerciseSpec("1", "Per-bar returns", [
        "A return is (price now / price before) - 1. pandas has pct_change() for exactly this.",
        "Returns must not cross the overnight gap: group by session first -> closes.groupby(sessions).pct_change(...)",
        "Pass fill_method=None to pct_change so pandas does not silently fill gaps.",
    ], _ref1, check_ex1, lambda: _src(S.compute_bar_returns)),
    "2": ExerciseSpec("2", "3-bar momentum", [
        "Momentum over 3 bars compares close[t] with close[t-3]: it is pct_change with periods=3.",
        "Still within a session: closes.groupby(sessions).pct_change(periods=3, fill_method=None).",
        "Equivalent: closes / closes.groupby(sessions).shift(3) - 1.",
    ], _ref2, check_ex2, lambda: _src(S.compute_momentum)),
    "3": ExerciseSpec("3", "Fit a linear model", [
        "from sklearn.linear_model import LinearRegression; model = LinearRegression().fit(X_train, y_train)",
        "Scaling is optional for OLS but keeps coefficients comparable: make_pipeline(StandardScaler(), LinearRegression()).",
        "Fit ONLY with X_train, y_train. Never pass validation rows to fit().",
    ], _ref3, check_ex3, lambda: _src(S.fit_linear_model)),
    "4": ExerciseSpec("4", "Allocation rule", [
        "Step 1: keep forecasts that are finite AND greater than config.threshold.",
        "Step 2: sort candidates by forecast (highest first); break ties by symbol name; keep the top config.max_positions.",
        "Step 3: each chosen asset gets min(config.exposure_budget / config.max_positions, config.per_asset_cap); others 0.",
    ], _ref4, check_ex4, lambda: _src(S.my_allocate)),
    "5": ExerciseSpec("5", "Choose and freeze a configuration", [
        "Try 2-3 thresholds on VALIDATION with the cost slider; look at net equity, drawdown and turnover.",
        "Higher threshold -> fewer trades -> lower costs, but you also skip weaker opportunities.",
        "Return (PortfolioConfig(...), 'one or two sentences of evidence from validation').",
    ], _ref5, check_ex5, lambda: f"my_config = {S.REFERENCE_CONFIG!r}\njustification = {S.REFERENCE_JUSTIFICATION!r}"),
    "6a": ExerciseSpec("6a", "strategy.py: predict_returns(history, model)", [
        "latest_features(history) returns one row per symbol with the FEATURE_NAMES columns (NaN during warm-up).",
        "ready = latest[FEATURE_NAMES].notna().all(axis=1) picks the rows you can predict on.",
        "preds[latest.index[ready]] = model.predict(latest.loc[ready, FEATURE_NAMES].to_numpy(dtype=float))",
    ], _ref6, check_ex6a, lambda: S.STRATEGY_PY),
    "6b": ExerciseSpec("6b", "strategy.py: allocate(predictions, config)", [
        "Copy the body of your Exercise 4 function; it already has the right inputs and outputs.",
        "The file must not use notebook variables: everything comes from the arguments or quantsoc imports.",
        "Re-run the 6a cell first if you want a clean file (6b appends to it).",
    ], _ref6, check_ex6b, lambda: S.STRATEGY_PY),
}


class ExerciseTracker:
    def __init__(self, auto_reference: Optional[bool] = None):
        self.states: Dict[str, ExerciseState] = {k: ExerciseState() for k in SPECS}
        self.contexts: Dict[str, dict] = {}
        self.auto_reference = (os.environ.get("QSW_AUTO_REFERENCE", "") == "1") if auto_reference is None else auto_reference

    # -- context registration (set by the notebook before each exercise)
    def set_context(self, key, **context) -> None:
        self.contexts[str(key)] = context

    # -- attempts
    def attempt(self, key, fn: Optional[Callable] = None, *args, value=None, **kwargs):
        key = str(key)
        spec, st = SPECS[key], self.states[key]
        print(f"Exercise {key}: {spec.title}")
        try:
            result = fn(*args, **kwargs) if fn is not None else value
        except NotImplementedError as e:
            st.status, st.message = "not_started", f"not started ({e})"
            print(f"  [ ] Not started yet. Hints: ex.hint({key})   Reference: ex.show_solution({key}) / ex.use_reference({key})")
            return self._auto(key)
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            line = tb[-1].lineno if tb else "?"
            st.status, st.message = "error", f"{type(e).__name__}: {e} (line {line})"
            print(f"  [x] Your code raised {type(e).__name__}: {e} (line {line})")
            return self._auto(key)
        ok, msg = spec.checker(result, self.contexts.get(key, {}))
        if ok:
            st.status, st.message, st.result = "passed", msg, result
            print(f"  [ok] {msg}")
            return result
        st.status, st.message, st.result = "failed", msg, None
        print(f"  [x] Not yet: {msg}")
        print(f"       Hints: ex.hint({key})   Reference: ex.show_solution({key}) / ex.use_reference({key})")
        return self._auto(key)

    def _auto(self, key):
        if self.auto_reference:
            print("  QSW_AUTO_REFERENCE=1: falling back to the reference solution (labelled).")
            return self.use_reference(key)
        return None

    def check(self, key, value):
        """Check a value directly (used for exercises that are not functions)."""
        return self.attempt(key, None, value=value)

    # -- results
    def get(self, key):
        key = str(key)
        st = self.states[key]
        if st.status in ("passed", "reference"):
            return st.result
        raise ExerciseIncomplete(
            f"Exercise {key} ({SPECS[key].title}) is not complete yet ({st.status}: {st.message or 'run the exercise cell'}). "
            f"Finish it above, or run ex.use_reference({key}) to continue with the labelled reference solution.")

    def passed(self, key) -> bool:
        return self.states[str(key)].status in ("passed", "reference")

    def require(self, *keys) -> None:
        """Stop with one friendly message if any of the exercises is not complete (guards later cells)."""
        for key in keys:
            self.get(key)

    # -- help
    def hint(self, key, level: Optional[int] = None) -> None:
        key = str(key)
        spec, st = SPECS[key], self.states[key]
        if level is None:
            st.hints_shown = min(st.hints_shown + 1, len(spec.hints))
            level = st.hints_shown
        level = max(1, min(int(level), len(spec.hints)))
        for i in range(level):
            print(f"Hint {i + 1}/{len(spec.hints)}: {spec.hints[i]}")

    def show_solution(self, key) -> None:
        key = str(key)
        print(f"--- Reference solution for Exercise {key} ({SPECS[key].title}) ---")
        print(SPECS[key].solution_source())

    def use_reference(self, key):
        key = str(key)
        spec, st = SPECS[key], self.states[key]
        result = spec.reference(self.contexts.get(key, {}))
        st.status, st.result = "reference", result
        st.message = "REFERENCE solution in use (not the participant's own work)"
        print(f"  [ref] Exercise {key}: using the workshop REFERENCE solution. This is recorded and labelled in your export.")
        return result

    # -- reporting
    def summary(self) -> pd.DataFrame:
        rows = [{"exercise": k, "title": SPECS[k].title, "status": v.status, "message": v.message}
                for k, v in self.states.items()]
        return pd.DataFrame(rows).set_index("exercise")

    def status_dict(self) -> Dict[str, str]:
        return {k: v.status for k, v in self.states.items()}

    @property
    def any_reference(self) -> bool:
        return any(v.status == "reference" for v in self.states.values())

    def all_done(self) -> bool:
        return all(v.status in ("passed", "reference") for v in self.states.values())
