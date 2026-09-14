"""Metrics for the danger objective ([C]).

The organism emits CALM / ALERT / ESCAPE per bar; the outcome is whether the
largest excursion over the horizon exceeded the danger threshold. We report:

- AUC of the danger share against the danger label (does the readout rank risk?)
- ESCAPE precision / recall / false-alarm rate; ALERT-or-ESCAPE recall
- lead time: for each danger event (a run of DANGER-labelled bars), how many bars
  before its first bar the organism was already in ALERT/ESCAPE (0 if never)
- exposure avoided: mean adverse excursion of a naive always-exposed trend rule on
  bars where the organism was CALM vs where it ESCAPED, and the share of total
  adverse excursion that sits inside ESCAPE bars (how much storm it dodged) vs
  the share of bars spent escaping (the cost in missed exposure).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def danger_frame(records, ds, threshold: float) -> pd.DataFrame:
    idx = pd.to_datetime([r.t for r in records], utc=True)
    df = pd.DataFrame({
        "state": [r.action for r in records], "arousal": [r.arousal for r in records], "gated": [r.gated for r in records],
        "share_danger": [r.pop_share.get("danger", np.nan) for r in records],
        "share_safe": [r.pop_share.get("safe", np.nan) for r in records],
    }, index=idx)
    out = ds.outcomes.reindex(idx)
    df["risk"] = np.maximum(out["mfe_long"].to_numpy(), out["mae_long"].to_numpy())
    df["danger"] = (df["risk"] > threshold).astype(int)
    df["mae_long"] = out["mae_long"].to_numpy(); df["mfe_long"] = out["mfe_long"].to_numpy()
    df["trend"] = ds.z["trend_slope"].reindex(idx).to_numpy()
    df["atr_regime"] = ds.z["atr_regime"].reindex(idx).to_numpy()
    return df[np.isfinite(df["risk"])]


def lead_times(df: pd.DataFrame) -> list[int]:
    """Bars of ALERT/ESCAPE immediately preceding each danger event's first bar."""
    d = df["danger"].to_numpy(); warn = df["state"].isin(["ALERT", "ESCAPE"]).to_numpy()
    leads = []
    for i in range(1, len(d)):
        if d[i] == 1 and d[i - 1] == 0:                      # event start
            k, j = 0, i - 1
            while j >= 0 and warn[j] and d[j] == 0:
                k += 1; j -= 1
            leads.append(k)
    return leads


def evaluate_danger(records, ds, cfg: dict, threshold: float) -> dict:
    df = danger_frame(records, ds, threshold)
    m = {"threshold_atr": float(threshold), "bars": int(len(df)), "danger_base_rate": float(df["danger"].mean())}
    for st in ("CALM", "ALERT", "ESCAPE"):
        m[f"frac_{st}"] = float((df["state"] == st).mean())
    sd = df["share_danger"].fillna(0.0)
    try:
        m["auc_share_danger"] = float(roc_auc_score(df["danger"], sd)) if df["danger"].nunique() == 2 else float("nan")
    except ValueError:
        m["auc_share_danger"] = float("nan")
    esc = df["state"] == "ESCAPE"; warn = df["state"].isin(["ALERT", "ESCAPE"]); dng = df["danger"] == 1
    m["escape_precision"] = float((esc & dng).sum() / esc.sum()) if esc.sum() else float("nan")
    m["escape_recall"] = float((esc & dng).sum() / dng.sum()) if dng.sum() else float("nan")
    m["escape_false_alarm_rate"] = float((esc & ~dng).sum() / (~dng).sum()) if (~dng).sum() else float("nan")
    m["warn_recall"] = float((warn & dng).sum() / dng.sum()) if dng.sum() else float("nan")
    leads = lead_times(df)
    m["danger_events"] = len(leads)
    m["events_warned_before"] = float(np.mean([l > 0 for l in leads])) if leads else float("nan")
    m["mean_lead_bars"] = float(np.mean(leads)) if leads else float("nan")
    # exposure avoided for an always-exposed naive trend rule
    follow = np.sign(np.nan_to_num(df["trend"].to_numpy()))
    adverse = np.where(follow > 0, df["mae_long"], np.where(follow < 0, df["mfe_long"], np.nan))
    adv = pd.Series(adverse, index=df.index)
    m["adverse_when_calm"] = float(adv[df["state"] == "CALM"].mean()) if (df["state"] == "CALM").any() else float("nan")
    m["adverse_when_escape"] = float(adv[esc].mean()) if esc.any() else float("nan")
    tot = np.nansum(adverse)
    m["share_of_adverse_inside_escape"] = float(np.nansum(adverse[esc.to_numpy()]) / tot) if tot > 0 else float("nan")
    m["share_of_bars_escaped"] = float(esc.mean())
    m["dodge_ratio"] = (m["share_of_adverse_inside_escape"] / m["share_of_bars_escaped"]
                        if m["share_of_bars_escaped"] > 0 else float("nan"))       # >1 = escapes concentrate on storms
    d = df.copy()
    d["regime"] = pd.cut(d["atr_regime"], [-np.inf, -0.3, 0.3, np.inf], labels=["low_vol", "mid_vol", "high_vol"])
    m["regimes"] = {str(k): {"bars": int(len(v)), "frac_escape": float((v["state"] == "ESCAPE").mean()),
                             "danger_rate": float(v["danger"].mean())} for k, v in d.groupby("regime", observed=True)}
    m["state_timeline"] = df["state"].map({"CALM": 0, "ALERT": 1, "ESCAPE": 2}).tolist()
    return m
