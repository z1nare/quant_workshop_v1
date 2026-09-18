"""Plot helpers for the notebook.  Every figure has a title, labelled axes (with units) and a legend.

All functions return the Figure so the notebook can display or save it.  Nothing here hides
a learning objective; these are the repetitive plotting parts only.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import BacktestResult
from .features import FEATURE_NAMES
from .portfolio import PortfolioConfig

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3, "figure.figsize": (10, 4.5)})


def _bps(x, pos=None):
    return f"{x * 1e4:.0f}"


def normalised_prices(closes: pd.DataFrame, symbols: Optional[Iterable[str]] = None, title="Normalised close prices") -> plt.Figure:
    symbols = list(symbols) if symbols is not None else list(closes.columns)
    fig, ax = plt.subplots()
    for i, s in enumerate(symbols):
        series = closes[s] / closes[s].iloc[0] * 100
        ax.plot(range(len(series)), series.to_numpy(), label=s, color=PALETTE[i % len(PALETTE)], lw=1)
    ax.set_title(f"{title} (start = 100)")
    ax.set_xlabel("bar number (5-minute bars, sessions joined end to end)")
    ax.set_ylabel("price index (start = 100)")
    ax.legend(loc="upper left", ncol=len(symbols))
    return fig


def return_histograms(returns: pd.DataFrame, symbols: Optional[Iterable[str]] = None, bins: int = 60) -> plt.Figure:
    symbols = list(symbols) if symbols is not None else list(returns.columns)
    fig, axes = plt.subplots(1, len(symbols), figsize=(2.6 * len(symbols) + 2, 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, s, c in zip(axes, symbols, PALETTE):
        r = returns[s].dropna() * 1e4
        ax.hist(r, bins=bins, color=c, alpha=0.8)
        ax.set_title(f"{s}\nstd {r.std():.1f} bps")
        ax.set_xlabel("5-min return (bps)")
    axes[0].set_ylabel("number of bars")
    fig.suptitle("Distribution of per-bar returns (1 bp = 0.01%)")
    fig.tight_layout()
    return fig


def split_timeline(closes: pd.DataFrame, boundaries: Dict[str, tuple], symbol: str) -> plt.Figure:
    fig, ax = plt.subplots()
    x = np.arange(len(closes))
    ax.plot(x, closes[symbol].to_numpy(), color="black", lw=1, label=f"{symbol} close")
    colors = {"train": "#cfe2f3", "validation": "#fff2cc", "test": "#f4cccc"}
    idx = closes.index
    for name, (start, end) in boundaries.items():
        i0, i1 = idx.get_loc(start), idx.get_loc(end)
        ax.axvspan(i0, i1, color=colors[name], alpha=0.8, label=f"{name}: bars {i0}-{i1}")
    ax.set_title("Chronological split: train (60%) -> validation (20%) -> final test (20%)")
    ax.set_xlabel("bar number")
    ax.set_ylabel("price ($)")
    ax.legend(loc="upper left")
    return fig


def bucket_means(x: pd.Series, y: pd.Series, n_buckets: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    df["bucket"] = pd.qcut(df["x"], n_buckets, labels=False, duplicates="drop")
    g = df.groupby("bucket")
    out = pd.DataFrame({"feature_mean": g["x"].mean(), "target_mean": g["y"].mean(),
                        "target_se": g["y"].std() / np.sqrt(g.size()), "n": g.size()})
    return out


def feature_vs_target(rows: pd.DataFrame, feature: str, target: str = "target", n_buckets: int = 10,
                      title_suffix: str = "(training rows only)") -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    x, y = rows[feature], rows[target]
    ax1.scatter(x * 1e4, y * 1e4, s=4, alpha=0.25, color=PALETTE[0], label="one (bar, asset) example")
    ax1.set_xlabel(f"{feature} (bps)")
    ax1.set_ylabel("next tradable return (bps)")
    ax1.set_title(f"Every example {title_suffix}")
    ax1.legend(loc="upper left")
    bm = bucket_means(x, y, n_buckets)
    ax2.errorbar(bm["feature_mean"] * 1e4, bm["target_mean"] * 1e4, yerr=1.96 * bm["target_se"] * 1e4,
                 fmt="o-", color=PALETTE[1], capsize=3, label="bucket mean +/- 95% CI")
    ax2.axhline(0, color="grey", lw=1)
    ax2.set_xlabel(f"{feature} (bps, bucket mean)")
    ax2.set_ylabel("mean next tradable return (bps)")
    ax2.set_title(f"Same data, {len(bm)} equal-count buckets")
    ax2.legend(loc="upper left")
    corr = float(np.corrcoef(x.dropna(), y[x.notna()].dropna())[0, 1]) if x.notna().sum() > 2 else float("nan")
    fig.suptitle(f"Can you spot the signal?  correlation({feature}, target) = {corr:+.3f}")
    fig.tight_layout()
    return fig


def forecast_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, split: str = "validation") -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.scatter(y_pred * 1e4, y_true * 1e4, s=4, alpha=0.25, color=PALETTE[2], label="example")
    lim = np.percentile(np.abs(y_true) * 1e4, 99)
    ax1.plot([-lim, lim], [-lim, lim], color="grey", lw=1, ls="--", label="perfect forecast")
    ax1.set_xlim(np.percentile(y_pred * 1e4, [0.5, 99.5]))
    ax1.set_ylim(-lim, lim)
    ax1.set_xlabel("forecast (bps)")
    ax1.set_ylabel("realised next tradable return (bps)")
    ax1.set_title(f"Forecast vs realised ({split})")
    ax1.legend(loc="upper left")
    bm = bucket_means(pd.Series(y_pred), pd.Series(y_true), 10)
    ax2.errorbar(bm["feature_mean"] * 1e4, bm["target_mean"] * 1e4, yerr=1.96 * bm["target_se"] * 1e4,
                 fmt="o-", color=PALETTE[3], capsize=3, label="bucket mean +/- 95% CI")
    ax2.axhline(0, color="grey", lw=1)
    ax2.set_xlabel("forecast (bps, bucket mean)")
    ax2.set_ylabel("mean realised return (bps)")
    ax2.set_title("Forecast deciles: does a higher forecast pay off on average?")
    ax2.legend(loc="upper left")
    fig.tight_layout()
    return fig


def allocation_bar(weights: pd.Series, predictions: Optional[pd.Series] = None, config: Optional[PortfolioConfig] = None,
                   title: str = "Target allocation") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = weights.astype(float)
    labels = list(w.index) + ["CASH"]
    vals = list(w.to_numpy() * 100) + [(1 - w.sum()) * 100]
    colors = [PALETTE[i % len(PALETTE)] if v > 0 else "#cccccc" for i, v in enumerate(vals[:-1])] + ["#999999"]
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    if config is not None:
        ax.axhline(config.per_asset_cap * 100, color="red", ls="--", lw=1, label=f"per-asset cap {config.per_asset_cap:.0%}")
        ax.legend(loc="upper right")
    if predictions is not None:
        for i, s in enumerate(w.index):
            p = predictions.get(s, np.nan)
            ax.text(i, -8, "NaN" if not np.isfinite(p) else f"{p * 1e4:+.1f} bps", ha="center", fontsize=8, color="dimgray")
    ax.set_ylim(-12, 105)
    ax.set_ylabel("weight (% of portfolio value)")
    ax.set_title(title + "  (numbers under bars = forecasts)")
    return fig


def performance_panel(results: Dict[str, BacktestResult], title: str = "Backtest") -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1]})
    for i, (name, r) in enumerate(results.items()):
        x = np.arange(len(r.equity))
        axes[0].plot(x, r.equity.to_numpy() / r.equity.iloc[0] * 100, label=r.label, color=PALETTE[i % len(PALETTE)], lw=1.2)
        axes[1].fill_between(x, r.drawdown.to_numpy() * 100, 0, alpha=0.25, color=PALETTE[i % len(PALETTE)], label=r.label)
        axes[2].plot(x, r.turnover.rolling(12).mean().to_numpy() * 100, color=PALETTE[i % len(PALETTE)], lw=1, label=r.label)
    axes[0].set_title(f"{title}: equity (start = 100)")
    axes[0].set_ylabel("portfolio value (index)")
    axes[0].legend(loc="upper left")
    axes[1].set_title("Drawdown: distance below the previous peak")
    axes[1].set_ylabel("drawdown (%)")
    axes[2].set_title("Turnover: value traded per bar (12-bar average)")
    axes[2].set_ylabel("turnover (% of portfolio)")
    axes[2].set_xlabel("bar number within the evaluation window")
    fig.tight_layout()
    return fig


def gross_vs_net(gross: BacktestResult, net: BacktestResult, cost_bps: float) -> plt.Figure:
    fig, ax = plt.subplots()
    x = np.arange(len(gross.equity))
    ax.plot(x, gross.equity / gross.equity.iloc[0] * 100, label="gross (no costs)", color=PALETTE[2])
    ax.plot(x, net.equity / net.equity.iloc[0] * 100, label=f"net of {cost_bps:g} bps per trade", color=PALETTE[3])
    ax.fill_between(x, net.equity / net.equity.iloc[0] * 100, gross.equity / gross.equity.iloc[0] * 100,
                    color=PALETTE[3], alpha=0.15, label="paid away in costs")
    ax.set_title(f"Where did the profit go?  total costs ${net.total_costs:,.0f}, avg turnover {net.avg_turnover:.0%} per bar")
    ax.set_xlabel("bar number (validation window)")
    ax.set_ylabel("portfolio value (start = 100)")
    ax.legend(loc="upper left")
    return fig


def coefficient_bars(coefs: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3))
    vals = coefs.iloc[:, 0] * 1e4
    ax.barh(coefs.index, vals, color=[PALETTE[0] if v >= 0 else PALETTE[3] for v in vals])
    ax.axvline(0, color="grey", lw=1)
    ax.set_xlabel("effect on forecast (bps per 1 std of the feature)")
    ax.set_title("What the linear model learned")
    return fig


def demo_animation(equity: pd.Series, weights: pd.DataFrame, n_frames: int = 60):
    """A short, self-contained animation of the finished system (equity growing + allocation changing)."""
    from matplotlib.animation import FuncAnimation
    idx = np.linspace(0, len(equity) - 1, n_frames).astype(int)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2), dpi=72, gridspec_kw={"width_ratios": [2, 1]})
    line, = ax1.plot([], [], color=PALETTE[0], lw=1.5, label="portfolio value")
    ax1.set_xlim(0, len(equity)); ax1.set_ylim(equity.min() * 0.999, equity.max() * 1.001)
    ax1.set_title("The finished system, replayed at high speed")
    ax1.set_xlabel("bar number (5-minute bars)"); ax1.set_ylabel("portfolio value ($)")
    ax1.legend(loc="upper left")
    syms = list(weights.columns) + ["CASH"]
    bars = ax2.bar(syms, [0] * len(syms), color=PALETTE[:len(weights.columns)] + ["#999999"])
    ax2.set_ylim(0, 100); ax2.set_ylabel("weight (%)"); ax2.set_title("Current allocation")
    stamp = ax1.text(0.02, 0.05, "", transform=ax1.transAxes, fontsize=9)

    def update(k):
        i = idx[k]
        line.set_data(np.arange(i + 1), equity.to_numpy()[: i + 1])
        w = weights.iloc[i]
        vals = list(w.to_numpy() * 100) + [(1 - w.sum()) * 100]
        for b, v in zip(bars, vals):
            b.set_height(v)
        stamp.set_text(f"{equity.index[i]:%Y-%m-%d %H:%M}   value ${equity.iloc[i]:,.0f}")
        return (line, *bars, stamp)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=120, blit=False)
    plt.close(fig)
    return anim


def stress_comparison(normal: BacktestResult, stress: BacktestResult) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, r, name in zip(axes, [normal, stress], ["Final test (normal regime)", "Stress scenario"]):
        x = np.arange(len(r.equity))
        ax.plot(x, r.equity / r.equity.iloc[0] * 100, color=PALETTE[0], label="frozen strategy (net)")
        ax.set_title(f"{name}: return {r.total_return:+.2%}, max drawdown {r.max_drawdown:.2%}")
        ax.set_xlabel("bar number"); ax.set_ylabel("portfolio value (start = 100)")
        ax.legend(loc="upper left")
    fig.suptitle("Your edge disappeared: same strategy, different market mechanism")
    fig.tight_layout()
    return fig
