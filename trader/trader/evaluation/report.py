"""Static HTML report ([C]) with plotly: equity curves, action timeline, weights, metrics table."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCALAR_KEYS = ["accuracy", "trade_frequency", "precision_LONG", "precision_SHORT", "precision_NO_TRADE",
               "recall_LONG", "recall_SHORT", "expectancy_atr", "win_rate", "profit_factor", "sharpe", "sortino",
               "total_log_return", "max_drawdown", "exposure", "n_trades_sequential",
               "avg_favorable_excursion_atr", "avg_adverse_excursion_atr"]


def _fmt(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def write_report(path: str, results: dict, decisions: pd.DataFrame, bars: pd.DataFrame, weights_by_pop: dict, manifest: dict) -> None:
    """results: {split: {model_name: metrics}}; decisions: organism test-split frame."""
    figs = []
    # equity curves per split
    for split, models in results.items():
        fig = go.Figure()
        for name, m in models.items():
            eq = m.get("equity_curve")
            if eq:
                fig.add_trace(go.Scatter(y=eq, mode="lines", name=name))
        fig.update_layout(title=f"Sequential equity (log return) — {split}", height=360, margin=dict(l=40, r=20, t=50, b=30),
                          xaxis_title="bars", yaxis_title="cum. log return")
        figs.append(fig)
    # organism decisions over price
    if len(decisions):
        d = decisions
        px = bars["close"].reindex(d.index)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.03)
        fig.add_trace(go.Scatter(x=d.index, y=px, mode="lines", name="close", line=dict(color="#888")), row=1, col=1)
        for act, color, sym in (("LONG", "#2ca02c", "triangle-up"), ("SHORT", "#d62728", "triangle-down")):
            s = d[d["action"] == act]
            fig.add_trace(go.Scatter(x=s.index, y=px.reindex(s.index), mode="markers", name=act,
                                     marker=dict(color=color, symbol=sym, size=7)), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["arousal"], name="arousal", line=dict(color="#ff7f0e")), row=2, col=1)
        for col, color in (("share_long", "#2ca02c"), ("share_short", "#d62728"), ("share_avoid", "#1f77b4")):
            fig.add_trace(go.Scatter(x=d.index, y=d[col], name=col, line=dict(color=color, width=1)), row=3, col=1)
        fig.update_layout(title="Organism decisions on the test split", height=640, margin=dict(l=40, r=20, t=50, b=30))
        figs.append(fig)
    # weights
    fig = go.Figure()
    for pop, w in weights_by_pop.items():
        fig.add_trace(go.Histogram(x=w, name=pop, opacity=0.6, nbinsx=40))
    fig.update_layout(barmode="overlay", title="KC→MBON weights after training (per population)", height=320,
                      margin=dict(l=40, r=20, t=50, b=30))
    figs.append(fig)

    rows = []
    for split, models in results.items():
        for name, m in models.items():
            rows.append(f"<tr><td>{split}</td><td>{name}</td>" + "".join(f"<td>{_fmt(m.get(k))}</td>" for k in SCALAR_KEYS) + "</tr>")
    table = ("<table><thead><tr><th>split</th><th>model</th>" + "".join(f"<th>{k}</th>" for k in SCALAR_KEYS) +
             "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")
    extra = []
    for split, models in results.items():
        fly = models.get("organism")
        if fly:
            extra.append(f"<h3>{split}: no-trade quality</h3><pre>{json.dumps(fly.get('no_trade'), indent=2)}</pre>"
                         f"<h3>{split}: regimes</h3><pre>{json.dumps(fly.get('regimes'), indent=2)}</pre>"
                         f"<h3>{split}: calibration</h3><pre>{json.dumps(fly.get('calibration'), indent=2)}</pre>")
    html = ["<!doctype html><html><head><meta charset='utf-8'><title>Trader Fly report</title>",
            "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>",
            "<style>body{font-family:system-ui,sans-serif;margin:24px;max-width:1300px}table{border-collapse:collapse;font-size:12px}"
            "td,th{border:1px solid #ccc;padding:3px 6px;text-align:right}th{background:#f3f3f3}pre{background:#f7f7f7;padding:8px;font-size:12px}</style></head><body>",
            f"<h1>Trader Fly — {manifest.get('experiment')}</h1>",
            f"<p>run {manifest.get('created')} · git {manifest.get('git_commit')} · data {manifest.get('data_range')} · fingerprint {manifest.get('data_fingerprint')} · seed {manifest['config'].get('seed')}</p>",
            "<p><b>Research prototype.</b> No live trading. Connectome category of this network: "
            f"{manifest['versions'].get('topology_category')}. Metrics are software checks on the given data, not a trading recommendation.</p>",
            "<h2>Metrics</h2><div style='overflow-x:auto'>", table, "</div>"]
    for i, fig in enumerate(figs):
        html.append(fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"fig{i}"))
    html += extra
    html.append("</body></html>")
    with open(path, "w") as f:
        f.write("\n".join(html))
