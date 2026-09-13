"""Evaluation metrics ([C]). Computed from per-bar decisions and their forward outcomes.

Positions are held for exactly `horizon` bars (one decision per bar, overlapping
positions allowed at the decision level; the equity curve uses non-overlapping
sequential trades: after a trade opens, new signals are ignored until it closes).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..labels.excursion import ACTION_NAMES, LONG, NO_TRADE, SHORT

ANN_BARS = {"1m": 525600, "5m": 105120, "15m": 35040, "1h": 8760, "4h": 2190, "1d": 365}


def decisions_frame(records, ds) -> pd.DataFrame:
    idx = pd.to_datetime([r.t for r in records], utc=True)
    df = pd.DataFrame({
        "action": [r.action for r in records], "label": [r.label for r in records],
        "arousal": [r.arousal for r in records], "gated": [r.gated for r in records],
        "share_long": [r.pop_share.get("long", np.nan) for r in records],
        "share_short": [r.pop_share.get("short", np.nan) for r in records],
        "share_avoid": [r.pop_share.get("avoid", np.nan) for r in records],
    }, index=idx)
    out = ds.outcomes.reindex(idx)
    df["fwd_return_atr"] = out["fwd_return_atr"].to_numpy()
    df["fwd_return"] = out["fwd_return"].to_numpy()
    df["mfe_long"] = out["mfe_long"].to_numpy(); df["mae_long"] = out["mae_long"].to_numpy()
    df["atr_regime"] = ds.z["atr_regime"].reindex(idx).to_numpy()
    df["atr_rel"] = ds.atr_rel.reindex(idx).to_numpy()
    return df


def classification_metrics(df: pd.DataFrame) -> dict:
    m = {}
    for a in ("LONG", "SHORT", "NO_TRADE"):
        pred = df["action"] == a
        true = df["label"] == a
        m[f"precision_{a}"] = float((pred & true).sum() / pred.sum()) if pred.sum() else float("nan")
        m[f"recall_{a}"] = float((pred & true).sum() / true.sum()) if true.sum() else float("nan")
        m[f"count_{a}"] = int(pred.sum())
    m["accuracy"] = float((df["action"] == df["label"]).mean())
    m["trade_frequency"] = float((df["action"] != "NO_TRADE").mean())
    return m


def trade_metrics(df: pd.DataFrame, horizon: int, timeframe: str, cost_atr: float) -> dict:
    """Per-decision PnL in ATR units (all signals) + sequential equity curve in log-return."""
    direction = df["action"].map({"LONG": 1.0, "SHORT": -1.0, "NO_TRADE": 0.0})
    pnl_atr = direction * df["fwd_return_atr"] - cost_atr * (direction != 0)
    traded = pnl_atr[direction != 0]
    m = {"n_trades_signals": int(len(traded))}
    if len(traded):
        wins, losses = traded[traded > 0], traded[traded <= 0]
        m["expectancy_atr"] = float(traded.mean())
        m["win_rate"] = float((traded > 0).mean())
        m["profit_factor"] = float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else float("inf")
        fav = np.where(direction[direction != 0] > 0, df.loc[direction != 0, "mfe_long"], df.loc[direction != 0, "mae_long"])
        adv = np.where(direction[direction != 0] > 0, df.loc[direction != 0, "mae_long"], df.loc[direction != 0, "mfe_long"])
        m["avg_favorable_excursion_atr"] = float(np.nanmean(fav))
        m["avg_adverse_excursion_atr"] = float(np.nanmean(adv))
    # sequential non-overlapping trades
    eq, pos_until, rets = 0.0, -1, []
    per_bar = np.zeros(len(df))
    acts, fr, atr_rel = df["action"].to_numpy(), df["fwd_return"].to_numpy(), np.nan_to_num(df["atr_rel"].to_numpy())
    n_seq = 0
    for i in range(len(df)):
        if i < pos_until or acts[i] == "NO_TRADE" or not np.isfinite(fr[i]):
            continue
        d = 1.0 if acts[i] == "LONG" else -1.0
        r = d * fr[i] - cost_atr * atr_rel[i]              # cost in log-return units: ATR fraction of price
        per_bar[min(i + horizon, len(df) - 1)] += r
        rets.append(r); pos_until = i + horizon; n_seq += 1
    equity = np.cumsum(per_bar)
    m["n_trades_sequential"] = n_seq
    m["total_log_return"] = float(equity[-1]) if len(equity) else 0.0
    peak = np.maximum.accumulate(equity) if len(equity) else np.array([0.0])
    m["max_drawdown"] = float((equity - peak).min()) if len(equity) else 0.0
    m["exposure"] = float(n_seq * horizon / max(1, len(df)))
    if len(rets) > 1:
        r = np.array(rets)
        per_year = ANN_BARS.get(timeframe, 8760) / horizon * m["exposure"]
        m["sharpe"] = float(r.mean() / (r.std() + 1e-12) * np.sqrt(max(per_year, 1)))
        down = r[r < 0]
        m["sortino"] = float(r.mean() / (down.std() + 1e-12) * np.sqrt(max(per_year, 1))) if len(down) > 1 else float("inf")
    m["equity_curve"] = equity.tolist()
    return m


def calibration(df: pd.DataFrame, bins: int = 5) -> list[dict]:
    """Does a larger directional share mean a better outcome? Bucketed by |share_long - share_short|."""
    d = df[df["action"] != "NO_TRADE"].copy()
    if d.empty:
        return []
    d["conf"] = (d["share_long"] - d["share_short"]).abs()
    d["hit"] = ((d["action"] == "LONG") & (d["fwd_return_atr"] > 0)) | ((d["action"] == "SHORT") & (d["fwd_return_atr"] < 0))
    try:
        d["bucket"] = pd.qcut(d["conf"], bins, duplicates="drop")
    except ValueError:
        return []
    g = d.groupby("bucket", observed=True)
    return [{"bucket": str(k), "n": int(len(v)), "hit_rate": float(v["hit"].mean())} for k, v in g]


def regime_breakdown(df: pd.DataFrame, cost_atr: float) -> dict:
    d = df.copy()
    d["regime"] = pd.cut(d["atr_regime"], [-np.inf, -0.3, 0.3, np.inf], labels=["low_vol", "mid_vol", "high_vol"])
    out = {}
    for k, v in d.groupby("regime", observed=True):
        direction = v["action"].map({"LONG": 1.0, "SHORT": -1.0, "NO_TRADE": 0.0})
        pnl = direction * v["fwd_return_atr"] - cost_atr * (direction != 0)
        tr = pnl[direction != 0]
        out[str(k)] = {"bars": int(len(v)), "trade_frequency": float((direction != 0).mean()),
                       "expectancy_atr": float(tr.mean()) if len(tr) else float("nan")}
    return out


def no_trade_quality(df: pd.DataFrame, cost_atr: float) -> dict:
    """Did the organism abstain when trading would have lost? Compare abstained bars vs traded bars."""
    abst = df[df["action"] == "NO_TRADE"]
    best_dir = np.abs(abst["fwd_return_atr"]) - cost_atr        # best-case directional edge if it had traded
    return {"abstained_bars": int(len(abst)),
            "abstained_best_case_edge_atr": float(best_dir.mean()) if len(abst) else float("nan"),
            "abstained_share_where_no_trade_was_right": float((abst["label"] == "NO_TRADE").mean()) if len(abst) else float("nan")}


def evaluate(records, ds, cfg: dict) -> dict:
    df = decisions_frame(records, ds)
    df = df[df["fwd_return_atr"].notna()]
    m = {}
    m.update(classification_metrics(df))
    m.update(trade_metrics(df, ds.horizon, cfg["data"]["timeframe"], cfg["reward"]["cost_atr"]))
    m["calibration"] = calibration(df)
    m["regimes"] = regime_breakdown(df, cfg["reward"]["cost_atr"])
    m["no_trade"] = no_trade_quality(df, cfg["reward"]["cost_atr"])
    m["label_distribution"] = df["label"].value_counts().to_dict()
    return m
