"""Interactive controls for the notebook, with static fallbacks that keep every learning objective.

Rules followed here
* Sliders use continuous_update=False (recalculate on release) or an explicit Apply button.
* Expensive results (backtests, replay traces) are computed once and cached; controls only re-render.
* Every builder registers its widgets/observers in _ACTIVE and `cleanup()` closes the previous
  instance of the same kind, so re-running a cell never leaves stale callbacks or timers.
* Nothing here submits broker orders from a slider or Play widget.  The paper panel has a
  one-shot, token-guarded Submit button that Run All cannot press.
"""
from __future__ import annotations

import html
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, clear_output, display

from . import viz
from .backtest import BacktestResult, ReplayTrace, run_backtest
from .features import FEATURE_NAMES
from .portfolio import PortfolioConfig, allocate_weights, check_weights

try:
    import ipywidgets as W
    WIDGETS_AVAILABLE = True
except Exception:                                   # pragma: no cover
    W = None
    WIDGETS_AVAILABLE = False

_ACTIVE: Dict[str, list] = {}


def cleanup(kind: Optional[str] = None) -> None:
    """Close widgets and stop timers created earlier by this module (all kinds, or one kind)."""
    kinds = [kind] if kind else list(_ACTIVE)
    for k in kinds:
        for w in _ACTIVE.pop(k, []):
            try:
                if hasattr(w, "playing"):
                    w.playing = False
                w.unobserve_all()
                w.close()
            except Exception:
                pass


def _register(kind: str, *widgets) -> None:
    cleanup(kind)
    _ACTIVE[kind] = list(widgets)


def _show(out, fig) -> None:
    with out:
        clear_output(wait=True)
        display(fig)
    plt.close(fig)


def _table(df: pd.DataFrame, title: str = "", fmt: Optional[Dict[str, str]] = None) -> str:
    if df is None or len(df) == 0:
        return f"<b>{title}</b><p style='color:gray'>none</p>"
    styled = df.copy()
    for col, f in (fmt or {}).items():
        if col in styled:
            styled[col] = styled[col].map(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v))
    return f"<b>{title}</b>" + styled.to_html(border=0, classes="qsw", escape=True)


# --------------------------------------------------------------------------- 1. asset explorer

def asset_explorer(closes: pd.DataFrame, returns: pd.DataFrame):
    if not WIDGETS_AVAILABLE:
        display(viz.normalised_prices(closes)); display(viz.return_histograms(returns)); return None
    view = W.Dropdown(options=["normalised prices", "return distributions"], description="view")
    which = W.SelectMultiple(options=list(closes.columns), value=tuple(closes.columns), description="assets", rows=5)
    out = W.Output()

    def render(_=None):
        syms = list(which.value) or list(closes.columns)
        fig = viz.normalised_prices(closes, syms) if view.value.startswith("norm") else viz.return_histograms(returns, syms)
        _show(out, fig)

    view.observe(render, "value"); which.observe(render, "value")
    _register("asset_explorer", view, which, out)
    render()
    return W.VBox([W.HBox([view, which]), out])


# --------------------------------------------------------------------------- 2. signal explorer

def signal_explorer(train_rows: pd.DataFrame):
    if not WIDGETS_AVAILABLE:
        for f in FEATURE_NAMES:
            display(viz.feature_vs_target(train_rows, f))
        return None
    feat = W.Dropdown(options=FEATURE_NAMES, value="mom_3", description="feature")
    buckets = W.IntSlider(value=10, min=4, max=20, step=1, description="buckets", continuous_update=False)
    out = W.Output()

    def render(_=None):
        _show(out, viz.feature_vs_target(train_rows, feat.value, n_buckets=buckets.value))

    feat.observe(render, "value"); buckets.observe(render, "value")
    _register("signal_explorer", feat, buckets, out)
    render()
    return W.VBox([W.HBox([feat, buckets]), out])


# --------------------------------------------------------------------------- 3. portfolio manager

def portfolio_manager(scenarios: Dict[str, pd.Series], allocate: Callable = None, base: Optional[PortfolioConfig] = None):
    """scenarios: name -> Series of forecasts (include an all-NaN one and an all-below-threshold one)."""
    allocate = allocate or allocate_weights
    base = base or PortfolioConfig()

    def figure(name, threshold_bps, max_positions, budget, cap):
        cfg = PortfolioConfig(threshold=threshold_bps / 1e4, max_positions=max_positions, exposure_budget=budget,
                              per_asset_cap=cap, cost_bps=base.cost_bps)
        preds = scenarios[name]
        w = pd.Series(allocate(preds, cfg), dtype=float).reindex(preds.index).fillna(0.0)
        rep = check_weights(w, cfg)
        fig = viz.allocation_bar(w, preds, cfg, title=f"'{name}': {rep['positions']} position(s), cash {rep['cash']:.0%}")
        return fig, rep

    if not WIDGETS_AVAILABLE:
        for name in scenarios:
            fig, rep = figure(name, base.threshold * 1e4, base.max_positions, base.exposure_budget, base.per_asset_cap)
            display(fig); print(rep)
        return None
    scen = W.Dropdown(options=list(scenarios), description="forecasts")
    thr = W.FloatSlider(value=base.threshold * 1e4, min=-5, max=15, step=0.5, description="threshold (bps)", continuous_update=False)
    mx = W.IntSlider(value=base.max_positions, min=1, max=5, description="max positions", continuous_update=False)
    bud = W.FloatSlider(value=base.exposure_budget, min=0.0, max=1.0, step=0.05, description="budget", continuous_update=False)
    cap = W.FloatSlider(value=base.per_asset_cap, min=0.05, max=1.0, step=0.05, description="per-asset cap", continuous_update=False)
    out = W.Output(); info = W.HTML()

    def render(_=None):
        fig, rep = figure(scen.value, thr.value, mx.value, bud.value, cap.value)
        _show(out, fig)
        info.value = f"<code>{html.escape(str(rep))}</code>"

    for w in (scen, thr, mx, bud, cap):
        w.observe(render, "value")
    _register("portfolio_manager", scen, thr, mx, bud, cap, out, info)
    render()
    return W.VBox([W.HBox([scen, thr]), W.HBox([mx, bud, cap]), out, info])


# --------------------------------------------------------------------------- 4. cost explorer

def cost_explorer(opens: pd.DataFrame, decisions: pd.DataFrame, config: PortfolioConfig):
    """Gross vs net validation equity for a chosen cost level.  Decisions are precomputed (cached)."""
    gross = run_backtest(opens, decisions, config, cost_bps=0.0, label="gross")
    cache: Dict[float, BacktestResult] = {0.0: gross}

    def net_for(bps: float) -> BacktestResult:
        if bps not in cache:
            cache[bps] = run_backtest(opens, decisions, config, cost_bps=bps, label=f"net {bps:g} bps")
        return cache[bps]

    if not WIDGETS_AVAILABLE:
        for bps in (1.0, 2.0, 5.0, 10.0):
            display(viz.gross_vs_net(gross, net_for(bps), bps))
        return None
    slider = W.FloatSlider(value=config.cost_bps, min=0, max=15, step=0.5, description="cost (bps)", continuous_update=False)
    apply = W.Button(description="Apply", button_style="primary")
    out = W.Output(); info = W.HTML()

    def render(_=None):
        net = net_for(float(slider.value))
        _show(out, viz.gross_vs_net(gross, net, slider.value))
        info.value = (f"gross return {gross.total_return:+.2%} &nbsp; net return {net.total_return:+.2%} &nbsp; "
                      f"costs ${net.total_costs:,.0f} &nbsp; max drawdown {net.max_drawdown:.2%}")

    apply.on_click(render)
    _register("cost_explorer", slider, apply, out, info)
    render()
    return W.VBox([W.HBox([slider, apply]), info, out])


# --------------------------------------------------------------------------- 5. replay monitor

def _frame_figure(trace: ReplayTrace, k: int) -> plt.Figure:
    eq = trace.equity
    steps = trace.steps[: k + 1]
    times = [s.fill_time for s in steps if s.fill_time is not None]
    vals = [s.value for s in steps if s.fill_time is not None]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    x_all = np.arange(len(eq))
    ax1.plot(x_all, eq.to_numpy(), color="#dddddd", lw=1, label="full replay (not yet revealed)")
    ax1.plot(np.arange(len(vals)), vals, color=viz.PALETTE[0], lw=1.5, label="portfolio value so far")
    ax1.set_ylabel("value ($)"); ax1.set_title("Accelerated replay: equity"); ax1.legend(loc="upper left")
    if vals:
        dd = (np.array(vals) / np.maximum.accumulate(vals) - 1) * 100
        ax2.fill_between(np.arange(len(vals)), dd, 0, color=viz.PALETTE[3], alpha=0.3, label="drawdown")
    ax2.set_ylabel("drawdown (%)"); ax2.set_xlabel("step (5-minute bars)"); ax2.legend(loc="lower left")
    fig.tight_layout()
    return fig


def _frame_html(trace: ReplayTrace, k: int, tail_log: int = 8) -> str:
    st = trace.steps[k]
    syms = trace.symbols
    preds = pd.DataFrame({"forecast (bps)": [st.predictions.get(s) for s in syms],
                          "target weight": [None if st.target_weights is None else st.target_weights.get(s, 0.0) for s in syms],
                          "actual weight": [st.actual_weights.get(s, 0.0) for s in syms]}, index=syms)
    preds["forecast (bps)"] = preds["forecast (bps)"].map(lambda v: "n/a" if v is None or not np.isfinite(v) else f"{v * 1e4:+.1f}")
    preds["target weight"] = preds["target weight"].map(lambda v: "hold" if v is None else f"{v:.0%}")
    preds["actual weight"] = preds["actual weight"].map(lambda v: f"{v:.1%}")
    orders = pd.DataFrame([{"symbol": s, "side": "BUY" if v > 0 else "SELL", "dollars": f"${abs(v):,.0f}"}
                           for s, v in st.orders.items()])
    log_lines = [f"{s.decision_time:%m-%d %H:%M}  {s.message}" for s in trace.steps[max(0, k - tail_log + 1): k + 1]]
    head = (f"<div style='font-family:monospace'><b>simulated time</b> {st.decision_time:%Y-%m-%d %H:%M} (bar completed) "
            f"&rarr; fill at {'-' if st.fill_time is None else f'{st.fill_time:%H:%M}'} &nbsp;|&nbsp; "
            f"<b>value</b> ${st.value:,.2f} &nbsp;|&nbsp; <b>cash</b> ${st.cash:,.2f} ({st.cash / st.value:.0%}) "
            f"&nbsp;|&nbsp; <b>cost this step</b> ${st.cost:.2f}</div>")
    return (head + "<div style='display:flex;gap:24px'>" + _table(preds, "Predictions & positions") +
            _table(orders, "Orders at this fill") + "</div><b>Decision log</b><pre style='font-size:11px'>" +
            html.escape("\n".join(log_lines)) + "</pre>")


def replay_frame_static(trace: ReplayTrace, k: int):
    """Static fallback: show one replay step (figure + tables)."""
    display(_frame_figure(trace, k)); display(HTML(_frame_html(trace, k)))


def replay_monitor(trace: ReplayTrace, interval_ms: int = 300):
    """Play / pause / step / speed controls over a precomputed replay trace."""
    n = len(trace.steps)
    if not WIDGETS_AVAILABLE:
        for k in (0, n // 3, 2 * n // 3, n - 1):
            replay_frame_static(trace, k)
        return None
    play = W.Play(value=0, min=0, max=n - 1, step=1, interval=interval_ms, description="play")
    slider = W.IntSlider(value=0, min=0, max=n - 1, description="step", continuous_update=False)
    W.jslink((play, "value"), (slider, "value"))
    speed = W.Dropdown(options=[("slow (1 bar/s)", 1000), ("normal (3 bars/s)", 300), ("fast (10 bars/s)", 100)],
                       value=interval_ms, description="speed")
    back = W.Button(description="< step"); fwd = W.Button(description="step >")
    fig_out = W.Output(); html_out = W.HTML()
    state = {"busy": False}

    def render(_=None):
        if state["busy"]:
            return
        state["busy"] = True
        try:
            k = int(slider.value)
            _show(fig_out, _frame_figure(trace, k))
            html_out.value = _frame_html(trace, k)
        finally:
            state["busy"] = False

    def on_speed(change):
        play.interval = change["new"]

    back.on_click(lambda _: setattr(slider, "value", max(0, slider.value - 1)))
    fwd.on_click(lambda _: setattr(slider, "value", min(n - 1, slider.value + 1)))
    slider.observe(render, "value"); speed.observe(on_speed, "value")
    _register("replay_monitor", play, slider, speed, back, fwd, fig_out, html_out)
    render()
    return W.VBox([W.HBox([play, slider, speed, back, fwd]), fig_out, html_out])


# --------------------------------------------------------------------------- 6. paper preview / submit

def paper_panel(engine, on_result: Optional[Callable] = None):
    """Preview proposed orders, then optionally submit them ONCE with an explicit click.

    The Submit button is disabled until a preview exists, is tied to that preview's token, and
    disables itself after one click.  Re-running the cell creates a new panel and invalidates
    the old token, so a lingering callback cannot submit.
    """
    from .engine import TradingEngine
    tokens = {"current": None, "used": set()}

    def preview_cycle():
        res = engine.run_cycle(submit=False)
        return res

    def submit_cycle():
        return engine.run_cycle(submit=True)

    if not WIDGETS_AVAILABLE:
        print("ipywidgets not available: use engine.run_cycle(submit=False) to preview and, only if you want to "
              "submit, engine.run_cycle(submit=True) in a cell you run by hand.")
        return None
    preview = W.Button(description="Preview proposed orders", button_style="info")
    submit = W.Button(description="Submit these orders (paper)", button_style="danger", disabled=True)
    out = W.Output()

    def show(res):
        with out:
            clear_output(wait=True)
            print(f"mode={res.mode}  status={res.status}  reason={res.reason}")
            print(f"account: {res.account}")
            if res.proposed_orders:
                display(pd.DataFrame(res.proposed_orders)[["symbol", "real_symbol", "side", "qty", "notional", "price", "reason"]])
            if res.submitted_orders:
                display(pd.DataFrame(res.submitted_orders)[["symbol", "side", "qty", "notional", "status", "filled_qty", "client_order_id"]])
            print("\n".join(res.log[-15:]))

    def on_preview(_):
        res = preview_cycle()
        show(res)
        tokens["current"] = object() if res.status == "preview" and engine.mode == "paper" else None
        submit.disabled = tokens["current"] is None
        if on_result:
            on_result(res)

    def on_submit(btn):
        tok = tokens["current"]
        if tok is None or tok in tokens["used"]:
            with out:
                print("Nothing to submit: preview first (each preview allows exactly one submission).")
            return
        tokens["used"].add(tok); tokens["current"] = None
        btn.disabled = True
        res = submit_cycle()
        show(res)
        if on_result:
            on_result(res)

    preview.on_click(on_preview); submit.on_click(on_submit)
    _register("paper_panel", preview, submit, out)
    return W.VBox([W.HBox([preview, submit]), out])


# --------------------------------------------------------------------------- 7. inline state monitor

def state_monitor(state_dir="state", auto_refresh_s: Optional[int] = None):
    """Cloud dashboard: render the runner's on-disk state inside the notebook (no web server)."""
    from .state import StateStore
    store = StateStore(state_dir)

    def render_html() -> str:
        snap = store.read_snapshot()
        if not snap:
            return "<p>No trading state yet. Run a cycle first.</p>"
        last = snap.get("last_cycle", {})
        hist = pd.DataFrame(snap.get("equity_history", []))
        parts = [f"<div style='font-family:monospace'><b>mode</b> {html.escape(str(snap.get('mode')))} "
                 f"&nbsp; <b>broker</b> {html.escape(str(snap.get('broker')))} &nbsp; <b>written</b> {snap.get('written_at')} "
                 f"&nbsp; <b>cycles</b> {snap.get('cycles')}<br><b>last status</b> {last.get('status')}: {html.escape(str(last.get('reason')))}</div>"]
        if last.get("predictions"):
            df = pd.DataFrame({"forecast (bps)": pd.Series(last["predictions"]).map(lambda v: "n/a" if v is None else f"{v * 1e4:+.1f}"),
                               "target weight": pd.Series(last.get("target_weights", {})).map(lambda v: f"{v:.0%}"),
                               "current weight": pd.Series(last.get("current_weights", {})).map(lambda v: f"{v:.1%}")})
            parts.append(_table(df, "Forecasts and weights"))
        if last.get("proposed_orders"):
            parts.append(_table(pd.DataFrame(last["proposed_orders"])[["symbol", "side", "qty", "notional", "price", "reason"]], "Proposed orders"))
        if last.get("submitted_orders"):
            parts.append(_table(pd.DataFrame(last["submitted_orders"])[["symbol", "side", "status", "filled_qty", "client_order_id"]], "Submitted orders"))
        parts.append("<b>Log</b><pre style='font-size:11px'>" + html.escape("\n".join(store.tail_log(12))) + "</pre>")
        return "".join(parts)

    def render_fig():
        snap = store.read_snapshot()
        hist = pd.DataFrame(snap.get("equity_history", [])) if snap else pd.DataFrame()
        fig, ax = plt.subplots(figsize=(9, 3))
        if len(hist):
            ax.plot(range(len(hist)), hist["equity"], color=viz.PALETTE[0], label="equity")
        ax.set_title("Equity recorded by the runner"); ax.set_xlabel("cycle"); ax.set_ylabel("value ($)"); ax.legend(loc="upper left")
        return fig

    if not WIDGETS_AVAILABLE:
        display(HTML(render_html())); display(render_fig()); return None
    refresh = W.Button(description="Refresh", button_style="info")
    body = W.HTML(); fig_out = W.Output()

    def render(_=None):
        body.value = render_html(); _show(fig_out, render_fig())

    refresh.on_click(render)
    _register("state_monitor", refresh, body, fig_out)
    render()
    return W.VBox([refresh, body, fig_out])
