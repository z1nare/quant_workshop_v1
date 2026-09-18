"""Local dashboard: READS the runner's state files; it never trades.

    streamlit run dashboard.py                      # binds to localhost:8501 by default (see .streamlit/config.toml)
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

STATE_DIR = Path("state")
st.set_page_config(page_title="QuantSoc trading monitor", layout="wide")


def load_state():
    p = STATE_DIR / "trading_state.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def tail_log(n=40):
    p = STATE_DIR / "decisions.log"
    return p.read_text(encoding="utf-8").splitlines()[-n:] if p.exists() else []


@st.fragment(run_every="5s")
def body():
    snap = load_state()
    if not snap:
        st.info("No trading state found in ./state yet. Start the runner: `python run_trader.py --mode replay`")
        return
    last = snap.get("last_cycle", {})
    written = snap.get("written_at")
    age = None
    if written:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(written)).total_seconds()
    mode = snap.get("mode", "?")
    labels = {"replay": "OFFLINE REPLAY (synthetic data, simulated fills)",
              "paper-preview": "ALPACA PAPER PREVIEW (no orders submitted)",
              "paper": "ALPACA PAPER TRADING (paper endpoint only)"}
    fresh = age is not None and age < (30 if mode == "replay" else 600)
    st.markdown(f"## Mode: {labels.get(mode, mode)}")
    c1, c2, c3 = st.columns(3)
    c1.metric("last state update", f"{age:.0f} s ago" if age is not None else "?",
              delta="live" if fresh else "STALE - runner not writing", delta_color="normal" if fresh else "inverse",
              help="Time since the runner last wrote its state; stale if the runner stopped.")
    c2.metric("runner cycles", snap.get("cycles", 0))
    hist_all = snap.get("equity_history", [])
    eq = last.get("account", {}).get("equity") or (hist_all[-1]["equity"] if hist_all else None)
    c3.metric("portfolio value", f"${eq:,.2f}" if eq else "n/a")
    st.markdown(f"**strategy:** {snap.get('strategy')} &nbsp;|&nbsp; **broker:** {snap.get('broker')} &nbsp;|&nbsp; "
                f"**sim/real time of last cycle:** {last.get('now')}")
    status = last.get("status", "?")
    color = {"traded": "green", "preview": "blue", "no_trade": "gray", "skipped": "orange", "error": "red"}.get(status, "gray")
    st.markdown(f"### Last decision: :{color}[{status.upper()}] - {last.get('reason', '')}")

    left, right = st.columns([3, 2])
    with left:
        hist = pd.DataFrame(snap.get("equity_history", []))
        if len(hist):
            hist["time"] = pd.to_datetime(hist["time"], utc=True, errors="coerce")
            chart = (alt.Chart(hist).mark_line().encode(
                x=alt.X("time:T", title="cycle time (simulated for replay, real for paper)"),
                y=alt.Y("equity:Q", title="portfolio value ($)", scale=alt.Scale(zero=False)),
                tooltip=["time:T", alt.Tooltip("equity:Q", format=",.2f")]).properties(height=260, title="Equity recorded by the runner"))
            st.altair_chart(chart, use_container_width=True)
            peak = hist["equity"].cummax()
            st.caption(f"drawdown now: {(hist['equity'].iloc[-1] / peak.iloc[-1] - 1):.2%}")
        preds = last.get("predictions") or {}
        if preds:
            df = pd.DataFrame({
                "forecast (bps)": pd.Series(preds).map(lambda v: None if v is None else round(v * 1e4, 2)),
                "target weight": pd.Series(last.get("target_weights", {})),
                "current weight": pd.Series(last.get("current_weights", {})),
            })
            st.dataframe(df, use_container_width=True)
        if last.get("skipped_symbols"):
            st.warning("Excluded symbols: " + "; ".join(f"{k}: {v}" for k, v in last["skipped_symbols"].items()))
    with right:
        st.markdown("**Positions**")
        pos = last.get("positions") or []
        if pos:
            st.dataframe(pd.DataFrame(pos), use_container_width=True)
        else:
            st.caption("no positions")
        st.markdown("**Proposed orders**")
        po = last.get("proposed_orders") or []
        if po:
            st.dataframe(pd.DataFrame(po)[["symbol", "side", "qty", "notional", "price", "reason"]], use_container_width=True)
        else:
            st.caption("none")
        st.markdown("**Submitted orders**")
        so = last.get("submitted_orders") or []
        if so:
            st.dataframe(pd.DataFrame(so)[["symbol", "side", "status", "filled_qty", "client_order_id"]], use_container_width=True)
        else:
            st.caption("none")
    st.markdown("**Decision log (latest last)**")
    st.code("\n".join(tail_log()), language="text")
    st.caption("Configuration: " + json.dumps(snap.get("config", {})) + " | symbol map: " + json.dumps(snap.get("symbol_map", {})))


st.title("QuantSoc trading monitor")
st.caption("Reads ./state written by run_trader.py. This page never places orders. Teaching project - not investment advice.")
body()
