# Verification report

What was actually built and tested for the workshop "Build Your First Quant Trading Strategy", as of
2026-09-18. Everything below was executed on a macOS (x86_64) machine with a `uv`-managed CPython 3.12.14
virtual environment. No Alpaca account, no network broker call and no real or paper order was involved at
any point: **all broker behaviour was exercised through the scriptable mock and the offline replay broker.**

## Environment actually used

| component | version used here | Colab reference (2026.07 runtime) | requirement range |
|---|---|---|---|
| Python | 3.12.14 | 3.12.13 | >=3.11 |
| numpy | 2.5.3 | (Colab-provided) | >=1.26,<3 |
| pandas | 2.3.3 | (Colab-provided) | >=2.2,<3 |
| scikit-learn | 1.9.1 | (Colab-provided) | >=1.5,<2 |
| matplotlib | 3.11.2 | (Colab-provided) | >=3.8,<4 |
| ipywidgets | 8.1.9 | (Colab-provided) | >=8.1,<9 |
| alpaca-py | 0.44.0 | installed by the setup cell | >=0.40,<1 |
| streamlit | 1.64.0 | not used in Colab | >=1.40,<2 |
| joblib | 1.6.0 | n/a | >=1.3 |

Additional compatibility runs (unit + engine + widget tests, notebooks excluded):
* Colab-like pin set: numpy 2.0.2, pandas 2.2.2, scikit-learn 1.6.1, matplotlib 3.10.0, ipywidgets 8.1.5: **see "Compatibility runs" below**.
* Forward set: pandas 3.0.x with current numpy/scikit-learn: **see "Compatibility runs" below**.

The exact package versions inside the Colab image were **not** read from a live Colab session (see "Not verified").

## Automated tests (pytest)

`pytest -q` in the repository root runs 102 tests. Result of the final full run: **see the summary at the end
of this file**. Groups and what they cover:

| file | tests | what is checked |
|---|---|---|
| `tests/test_market.py` | 9 | deterministic generation, different seeds differ, positive prices and valid OHLC, 78 bars per session with no overnight bars, open ≠ previous close, planted momentum present, stress parameters, CSV round trip, checked-in data matches the generator |
| `tests/test_features.py` | 6 | fixed feature order, **causality** (perturbing every future bar leaves earlier features bit-identical), session warm-up NaN pattern, values match their definitions, `latest_features` shape/order, a missing bar does not invent data |
| `tests/test_targets_splits.py` | 7 | target = `open[t+2]/open[t+1]−1`, targets never cross a session, 60/20/20 chronological split on shared timestamps, boundary-crossing examples removed, scaler fitted on training rows only, signal learnable on the tradable target, zero baseline MSE |
| `tests/test_portfolio.py` | 13 | the spec example (two qualify → 30/30/40 % cash), all-below-threshold, top-3 of many, non-finite exclusion, caps binding, deterministic ties, threshold changes, invalid configs rejected, duplicate symbols rejected, equal-weight benchmark |
| `tests/test_backtest.py` | 6 | **hand-calculated** two-asset example with drift and costs (costs, cash, turnover, hold rows), gross == net at zero cost, monotone cost effect, bad inputs rejected, cash never negative at 100 % budget, benchmark shares timing |
| `tests/test_replay_parity.py` | 3 | replay equity == vectorised backtest equity (rtol 1e-10), history revealed progressively, lookback respected |
| `tests/test_artifacts.py` | 8 | bundle round trip (model, config, strategy), ZIP contents and **exclusion of `.env`/`state/`**, export refuses pasted credentials, reference label recorded, schema/feature-order mismatch rejected, exported app runs in a subprocess from the bundle, strategy module imports with no notebook globals |
| `tests/test_engine.py` | 22 | preview never submits; Run-all default never submits; duplicate prevention on rerun; **restart reconciliation** via registry and broker lookup; uncertain submission reconciled before retry (accepted and not-accepted cases); rejected order; partial fill re-planned; sells before buys; insufficient buying power scaled; market closed; close buffer; stale data; warm-up; all-NaN forecasts; student bug does not crash the runner; constraint-violating allocation blocked; stale open orders cancelled then wait; same-bar open order not resubmitted; whole shares for non-fractionable assets; instrument validation; completed-bars filter; state snapshot written |
| `tests/test_exercises.py` | 9 | wrong answers rejected with specific reasons (cross-session returns, diff instead of return, 1-bar vs 3-bar), leakage (train+validation fit) detected, renormalised allocation rejected, config check, strategy.py check, labelled reference route, `QSW_AUTO_REFERENCE` fallback, hints/solutions present |
| `tests/test_widgets_dashboard.py` | 6 | replay monitor play/pause/step/speed, **re-running a widget cell cleans up the previous one**, sliders recalculate on release/Apply only, paper panel submits exactly once per preview and never from a stale panel, state monitor, Streamlit dashboard renders with and without state (AppTest) |
| `tests/test_release_tool.py` | 1 | `tools/set_release.py` rewrites `release.py`, notebook source, both notebooks and the README links consistently |
| `tests/test_notebooks.py` | 10 | notebooks in sync with the source; all 7 exercise cells in order; stubs vs solutions; cell count; **no unguarded order submission** in any cell; no outputs/credentials committed; release settings centralised; *(slow)* instructor notebook executes from a clean kernel; *(slow)* student recovery route executes from a clean kernel; *(slow)* untouched student notebook stops at the first checkpoint with `ExerciseIncomplete`, cell-by-cell gives only friendly stops |

CI (`.github/workflows/ci.yml`) runs the same suite on Ubuntu with Python 3.11 and 3.12, checks the data and
notebooks against their generators, and installs the exported reference bundle into a fresh virtualenv and
runs 40 replay cycles. CI needs no credentials. It has **not** been executed on GitHub yet (no repository exists).

## Notebook execution from a clean kernel (offline)

Measured with `tools/run_notebook.py` (nbclient, fresh `python3` kernel, `MPLBACKEND=Agg`, no credentials):

| route | result | wall time | slowest cells |
|---|---|---|---|
| `workshop_instructor.ipynb` (all solutions) | all 65 cells executed, 7× `[ok]`, parity `True`, replay == backtest `True`, offline banner shown | 35–43 s | demo animation ≈ 4–5 s, replay ≈ 8–15 s |
| `workshop_student.ipynb` with `QSW_AUTO_REFERENCE=1` (recovery route) | all 55 cells executed, 6× `[ref]` (6a's reference writes the whole `strategy.py`, so 6b then passes on its own), export labelled `REFERENCE` | ≈ 42 s | same |
| `workshop_student.ipynb` untouched (Run-all) | stops at cell #10 (checkpoint 1) with `ExerciseIncomplete`; 0× `[ok]`, nothing exported | ≈ 10 s | n/a |
| `workshop_student.ipynb` untouched, cell by cell (`--allow-errors`) | 18 friendly `[stop]` messages, **no tracebacks, no NameError**, nothing exported | ≈ 12 s | n/a |

Cell counts: student 55 cells (7 exercise cells: 1, 2, 3, 4, 5, 6a, 6b), instructor 65 cells (10 presenter-note
cells added). The agenda target was "approximately 44"; the extra cells are one-line guard/checkpoint cells
that keep each exercise self-contained.

## Observed research results (classroom seed 20250917, β = 0.07)

Reference configuration: threshold 3 bps, max 3 positions, 90 % budget, 30 % cap.

| split | forecast corr. | MSE improvement vs zero | hit rate | net return @2 bps | gross | equal-weight benchmark | max drawdown (net / benchmark) |
|---|---|---|---|---|---|---|---|
| validation (8 sessions) | 0.123 | 1.40 % | 53.2 % | +2.21 % | +3.30 % | +2.78 % | −0.87 % / −2.36 % |
| final test (8 sessions) | 0.161 | 2.45 % | 54.9 % | +4.09 % | +5.10 % | +4.39 % | −0.69 % / −2.33 % |
| final test @5 bps | n/a | n/a | n/a | +2.59 % | n/a | +4.35 % | −1.13 % |
| stress (10 sessions, σ×2, β = −0.05) | −0.061 | −2.88 % | n/a | **−4.76 %** | −1.50 % | +2.24 % | −7.78 % |

Model coefficients (per 1 std of feature): `ret_1` −0.18 bps, `mom_3` +2.58 bps, `vol_12` +0.08 bps.
Note that the equal-weight benchmark out-returns the strategy on the final test because the synthetic market
drifted up in that window while the strategy was mostly in cash; the strategy's drawdown is a third of the
benchmark's. This is discussed in the notebook; the seed was **not** chosen for its test outcome.

Calibration across seeds (`tools/calibrate_market.py --seeds 8 --classroom`, 2 bps): median validation
correlation 0.12, MSE improvement 1.5 %, gross +3.2 %, net +1.9 %; net validation profit in 7/9 seeds, net
test profit in 8/9. Stronger (β = 0.10: 9/9 profitable, corr 0.19) and weaker (β = 0.05: corr 0.08, 6/9)
settings were measured and rejected as too easy / too marginal.

## Local application and dashboard

* `python run_trader.py --mode replay --cycles 120 --speed 1000` from the repository root: builds the reference
  bundle (fit ≈ 1 s), replays 120 cycles in ≈ 10 s, writes `state/trading_state.json`, `orders.json`,
  `decisions.log`. Ctrl+C handling was tested by signal (`SIGINT` → finishes the cycle, lists open orders).
* Exported bundle in a **fresh virtualenv** (only `requirements.txt` installed): 40 replay cycles in 4 s
  (first run 78 s because macOS scanned the new binaries; second run 4 s), state files written. The same
  check runs in CI.
* `streamlit run dashboard.py` was opened in a browser and inspected visually: mode heading, freshness
  indicator (goes STALE after the runner stops), cycle count, portfolio value, equity chart with a non-zero
  axis, last-decision status with reason, positions / proposed / submitted tables, readable decision log.
  Binding is `localhost:8501` via `.streamlit/config.toml`.

## Visual inspection of notebook figures

Every figure produced by the instructor notebook was rendered to PNG and reviewed: return histograms,
split timeline, signal scatter + bucket means, forecast-vs-realised + deciles, coefficient bars, allocation
bar with cash and cap line, validation performance panel (equity / drawdown / turnover with benchmark),
gross-vs-net cost chart, replay monitor frame (revealed vs unrevealed equity, drawdown, prediction/position
tables, decision log), final-test panel and the stress comparison. All have titles, axis labels with units
and legends. The widget-rendered figures were produced through the same functions the widgets call.

## Widget interactions

Play/pause/step/speed, Apply-only recalculation, one-shot submit and cleanup on re-execution were tested
**headlessly** through ipywidgets' Python model (`tests/test_widgets_dashboard.py`). The rendering of those
widgets in a real Colab front end was **not** observed (see below).

## Not verified: manual checks remaining

1. **Google Colab UI.** The built-in browser is not signed into a Google account and signing in was not
   attempted, so nothing was run inside Colab: the setup cell's GitHub download, `%%writefile` in Colab,
   widget rendering (`Play`, `Output`), the JS animation, the ZIP download via `google.colab.files`, and Colab
   Secrets. Local clean-kernel execution is **not** a substitute. Use `docs/colab_acceptance_checklist.md`.
2. **Colab package versions** (the pinned ranges include what Colab has shipped in 2025–2026; the Colab-like
   pin run below is the closest evidence).
3. **Alpaca paper endpoint.** `AlpacaPaperBroker` was written against alpaca-py 0.44.0's actual class
   signatures (inspected locally) and the documented paper URL / IEX feed, but **no call to Alpaca was made**.
   The engine's broker behaviour is verified only against the mock. An instructor with paper keys should run
   `python run_trader.py --mode paper-preview` during market hours before the session.
4. **GitHub Actions** has not run (no repository/remote yet).
5. **Windows** local install instructions were written but not executed on Windows.

## Publishing configuration you must supply

* `quantsoc/release.py` and the notebook setup cell contain placeholders `YOUR-GITHUB-ORG / first-quant-strategy / v0.1.0`.
  After pushing and tagging, run `python tools/set_release.py --owner … --repo … --revision …` and commit.
* The README Colab badge points at the same placeholders and is rewritten by the tool.
* Optional: choose the real-instrument mapping in `.env.example` (`QSW_SYMBOL_MAP`, default liquid ETFs).

## Compatibility runs

Unit, engine, artifact, exercise, widget, dashboard and notebook-structure tests (`pytest -m "not slow"`, 99 tests)
were repeated in two fresh virtual environments:

| environment | numpy | pandas | scikit-learn | matplotlib | ipywidgets | result |
|---|---|---|---|---|---|---|
| Colab-like pin set | 2.0.2 | 2.2.2 | 1.6.1 | 3.10.0 | 8.1.5 | **99 passed** (15.7 s) |
| forward set (pandas 3) | 2.5.3 | 3.0.5 | 1.9.1 | 3.11.2 | 8.1.9 | **99 passed** (16.9 s) |

So the code base works across pandas 2.2 → 3.0 and numpy 2.0 → 2.5 even though `requirements.txt` pins `pandas<3`
for safety.

## Final full test run

`pytest -q` from the repository root, Python 3.12.14, no other processes running:

    102 passed in 98.76s

This includes the three clean-kernel notebook executions. One caveat discovered during verification: when the
notebook's `workspace/` directory was deleted by a concurrent job while a notebook was executing, the kernel
stalled instead of erroring; `tools/run_notebook.py` therefore has a watchdog (`--watchdog`, default 900 s) and
the notebook tests clean `workspace/` only before each run. Do not run two notebook executions in the same
checkout at the same time.
