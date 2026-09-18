# Timing contract

This document fixes *when* information exists and *when* money can move. Every module
(`quantsoc.features`, `quantsoc.modeling`, `quantsoc.backtest`, `quantsoc.engine`) implements exactly this.

## Bars

* Interval: 5 minutes. A bar labelled `T` covers `[T, T + 5 min)`. Alpaca labels bars the same way.
* Sessions: 09:30–16:00 America/New_York, 78 bars per session, weekdays only. There are **no** bars between
  16:00 and 09:30; the synthetic generator never creates overnight bars.
* A bar is **complete** (its close is observable) at `T + 5 min`. The engine only ever uses completed bars:
  `completed_bars_only()` drops any bar whose end is after "now".

## Decision at bar *t*

| Step | Time | Uses |
|---|---|---|
| Bar *t* completes | `t + 5 min` | nothing yet |
| Features computed | `t + 5 min` | closes of bars `≤ t` **in the same session** (`ret_1`, `mom_3`, `vol_12`) |
| Forecast | `t + 5 min` | features → `model.predict` |
| Allocation | `t + 5 min` | forecasts → target weights |
| Orders sent | `t + 5 min + ε` | target weights vs current positions |
| **Fill** | `open[t+1]` | the first tradable price after the decision (never `close[t]`) |
| Return accrues | `open[t+1] → open[t+2]` | holdings drift; the next decision's fill closes the interval |

Therefore the **target** used for training, scoring and selection is

```
target[t] = open[t+2] / open[t+1] − 1
```

and it is `NaN` whenever `t+1` or `t+2` belongs to another session. In the research table an example is also
dropped from a split if `t+1` or `t+2` fall in a different split (`crosses_split`).

## Warm-up and missing observations

* `ret_1` needs 1 previous bar in the session, `mom_3` needs 3, `vol_12` needs 12 returns → the first 12 bars of a
  session have at least one `NaN` feature. `MIN_HISTORY_BARS = 13`.
* All forecasts `NaN` (warm-up) ⇒ the engine/backtest **holds** existing positions (no decision, no trades).
* Some forecasts `NaN` (one symbol missing/stale) ⇒ that symbol is excluded from the allocation; the others proceed.
* A symbol whose latest completed bar is older than 2 intervals is treated as **stale** and excluded for the cycle.
* Nothing is interpolated or forward-filled.

## Why the planted signal survives the contract

The generator plants momentum in **close-to-close** log returns. The tradable target `open[t+2]/open[t+1]` equals
`exp(r[t+1] + gap[t+2] − gap[t+1]) − 1`, i.e. the one-step-ahead close-to-close return plus small open gaps, so the
relationship between `mom_3[t]` and the target is preserved (attenuated only by the gap noise).
`tests/test_targets_splits.py::test_signal_learnable_over_tradable_target` checks this.

## Live operation

The runner wakes shortly after each 5-minute boundary, requests IEX 5-minute bars, keeps completed regular-session bars,
computes features within the current session (so the first ~65 minutes of each day are warm-up), and submits
market orders that fill at the prevailing price (the real-world counterpart of `open[t+1]`).
