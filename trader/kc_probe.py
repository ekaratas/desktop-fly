#!/usr/bin/env python3
"""Does the Kenyon-cell code carry information at all?  ([C] diagnostic)

Two linear probes (logistic regression) from recorded KC spike codes, fitted on
the train split and scored on validation/test:

  direction: sign of the forward return            (what LONG/SHORT needs)
  danger:    large adverse excursion in BOTH directions within the horizon,
             i.e. max(MFE, MAE) in the top `danger_q` quantile of train
             (what a looming→escape / NO_TRADE decision needs)

If neither probe beats chance the problem is upstream of the plasticity rule
(features, sensory encoding, KC expansion) or there is no edge at this horizon.
The danger probe also reports whether abstaining on predicted-danger bars
improves a naive trend-following rule — the "learn when not to trade" question.

    python run_experiment.py configs/... --dump-kc --no-baselines --tag probe
    python kc_probe.py runs/<run_dir>
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def load(run_dir, split):
    z = np.load(os.path.join(run_dir, f"kc_{split}.npz"))
    X = (z["codes"] > 0).astype(np.float32)
    ok = np.isfinite(z["fwd_return_atr"])
    d = {k: z[k][ok] for k in z.files if k != "codes"}
    return X[ok], d


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--C", type=float, default=0.05)
    p.add_argument("--cost", type=float, default=None)
    p.add_argument("--danger-q", type=float, default=0.7)
    a = p.parse_args()
    cfg = json.load(open(os.path.join(a.run_dir, "manifest.json")))["config"]
    cost = a.cost if a.cost is not None else cfg["reward"]["cost_atr"]
    Xtr, tr = load(a.run_dir, "train")
    print(f"train: {Xtr.shape[0]} bars, {Xtr.shape[1]} KCs, mean active {Xtr.mean():.3f}")

    dir_clf = LogisticRegression(C=a.C, max_iter=2000).fit(Xtr, (tr["fwd_return_atr"] > 0).astype(int))
    has_exc = "mae_long" in tr
    if has_exc:
        risk_tr = np.maximum(tr["mfe_long"], tr["mae_long"])
        thr = np.quantile(risk_tr, a.danger_q)
        dan_clf = LogisticRegression(C=a.C, max_iter=2000).fit(Xtr, (risk_tr > thr).astype(int))
        print(f"danger threshold: max excursion > {thr:.2f} ATR (top {100*(1-a.danger_q):.0f}% of train)")

    for split in ("validation", "test"):
        try:
            X, d = load(a.run_dir, split)
        except FileNotFoundError:
            continue
        fr = d["fwd_return_atr"]
        pr = dir_clf.predict_proba(X)[:, 1]
        print(f"\n== {split}: {len(fr)} bars")
        print(f"DIRECTION probe: sign accuracy {((pr > 0.5) == (fr > 0)).mean():.3f}  chance {max((fr>0).mean(), (fr<=0).mean()):.3f}  "
              f"AUC {roc_auc_score((fr > 0).astype(int), pr):.3f}")
        print("  conf   freq      n   exp(ATR)  win")
        for th in (0.50, 0.55, 0.60, 0.65):
            take = np.abs(pr - 0.5) >= th - 0.5
            if take.sum() == 0:
                continue
            pnl = np.clip(np.where(pr > 0.5, 1.0, -1.0)[take] * fr[take] - cost, -3, 3)
            print(f"  {th:.2f}   {take.mean():.3f} {len(pnl):6d}   {pnl.mean():+.3f}   {(pnl>0).mean():.2f}")
        if not has_exc:
            continue
        risk = np.maximum(d["mfe_long"], d["mae_long"])
        y = (risk > thr).astype(int)
        pd_ = dan_clf.predict_proba(X)[:, 1]
        print(f"DANGER probe:    AUC {roc_auc_score(y, pd_):.3f} (0.5 = no information)  base rate {y.mean():.3f}")
        # does abstaining on predicted danger help a naive trend rule?
        trend = d["trend"]
        follow = np.sign(np.nan_to_num(trend))
        base = np.clip(follow * fr - cost * (follow != 0), -3, 3)
        print("  trend-following (long if trend_slope>0 else short) with danger filter:")
        print("  danger-cutoff  freq      n   exp(ATR)   maxAdverse(ATR)")
        for q in (1.01, 0.8, 0.6, 0.5, 0.4, 0.3):
            keep = (pd_ < np.quantile(pd_, q)) if q <= 1 else np.ones_like(pd_, bool)
            keep &= follow != 0
            adverse = np.where(follow[keep] > 0, d["mae_long"][keep], d["mfe_long"][keep])
            print(f"  {q if q<=1 else 'none':>12}   {keep.mean():.3f} {keep.sum():6d}   {base[keep].mean():+.3f}   {adverse.mean():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
