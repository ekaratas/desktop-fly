#!/usr/bin/env python3
"""Post-hoc sweep of the readout knobs (margin, no_trade_bias) from recorded decisions.

Valid because with the default `observed_all` dopamine rule learning does not
depend on the chosen action, so the MBON shares recorded per bar are what any
margin/bias would have seen. Tune on validation, then confirm once on test.

    python readout_sweep.py runs/<run_dir> [--split validation] [--cost 0.15]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def load(run_dir: str, split: str):
    rows = [json.loads(l) for l in open(os.path.join(run_dir, f"decisions_{split}.jsonl"))]
    rows = [r for r in rows if r.get("fwd_return_atr") is not None]
    S = np.array([[r["pop_share"]["long"], r["pop_share"]["short"], r["pop_share"]["avoid"]] for r in rows])
    fr = np.array([r["fwd_return_atr"] for r in rows])
    gated = np.array([r["gated"] for r in rows])
    return S, fr, gated


def decide(S: np.ndarray, margin: float, bias: float, gated: np.ndarray) -> np.ndarray:
    sc = S.copy(); sc[:, 2] += bias
    best = sc.argmax(axis=1)
    srt = np.sort(sc, axis=1)
    ambiguous = (srt[:, -1] - srt[:, -2]) < margin
    act = np.where(ambiguous | gated, 2, best)
    return act


def score(act: np.ndarray, fr: np.ndarray, cost: float, r_max: float = 3.0) -> dict:
    d = np.where(act == 0, 1.0, np.where(act == 1, -1.0, 0.0))
    pnl = np.clip(d * fr - cost, -r_max, r_max)
    tr = pnl[d != 0]
    out = {"freq": float((d != 0).mean()), "n": int(len(tr))}
    if len(tr):
        w, l = tr[tr > 0], tr[tr <= 0]
        out.update(exp=float(tr.mean()), win=float((tr > 0).mean()),
                   pf=float(w.sum() / -l.sum()) if len(l) and l.sum() < 0 else float("inf"),
                   long_share=float((act == 0).sum() / max(1, len(tr))))
    return out


def danger_sweep(run_dir: str, split: str, confirm: str) -> int:
    """Sweep alert/escape margins on recorded (safe, danger) shares; report precision/recall/dodge."""
    man = json.load(open(os.path.join(run_dir, "manifest.json")))
    thr = man.get("danger_threshold")
    rows = [json.loads(l) for l in open(os.path.join(run_dir, f"decisions_{split}.jsonl"))]
    rows = [r for r in rows if r.get("reward") is not None]
    z = np.array([r["gap_z"] if r.get("gap_z") is not None else 0.0 for r in rows])
    danger = np.array([r["label"] == "DANGER" for r in rows])
    print(f"{split}: {len(rows)} bars, danger base rate {danger.mean():.3f}, threshold {thr:.2f} ATR")
    print(" escape_z  frac_escape  precision  recall  false_alarm  lift")
    for m in (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5):
        esc = z >= m
        if esc.sum() == 0:
            print(f" {m:12.2f}  0.000          –"); continue
        prec = (esc & danger).sum() / esc.sum(); rec = (esc & danger).sum() / danger.sum(); fa = (esc & ~danger).sum() / (~danger).sum()
        print(f" {m:12.2f}  {esc.mean():.3f}        {prec:.3f}     {rec:.3f}   {fa:.3f}      {prec / danger.mean():.2f}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--split", default="validation")
    p.add_argument("--confirm", default="test", help="split to report the chosen setting on (no tuning)")
    p.add_argument("--cost", type=float, default=None)
    p.add_argument("--min-trades", type=int, default=40)
    a = p.parse_args()
    cfg = json.load(open(os.path.join(a.run_dir, "manifest.json")))["config"]
    if cfg.get("objective") == "danger":
        return danger_sweep(a.run_dir, a.split, a.confirm)
    cost = a.cost if a.cost is not None else cfg["reward"]["cost_atr"]
    S, fr, gated = load(a.run_dir, a.split)
    grid = [(m, b) for m in (0.0, 0.02, 0.03, 0.05, 0.07, 0.10, 0.13, 0.16) for b in (0.0, 0.02, 0.04, 0.06, 0.08)]
    rows = []
    for m, b in grid:
        s = score(decide(S, m, b, gated), fr, cost)
        rows.append((m, b, s))
    print(f"{a.split}: {len(fr)} bars, cost {cost} ATR\n margin  bias   freq     n   exp(ATR)   win    PF   long%")
    for m, b, s in rows:
        if s["n"] == 0:
            print(f" {m:5.2f} {b:5.2f}  {s['freq']:.3f} {s['n']:6d}      –"); continue
        print(f" {m:5.2f} {b:5.2f}  {s['freq']:.3f} {s['n']:6d}  {s['exp']:+.3f}  {s['win']:.2f}  {s['pf']:5.2f}  {s['long_share']:.2f}")
    ok = [(m, b, s) for m, b, s in rows if s["n"] >= a.min_trades]
    if ok:
        m, b, s = max(ok, key=lambda x: x[2]["exp"])
        print(f"\nbest on {a.split} (>= {a.min_trades} trades): margin {m} bias {b} → exp {s['exp']:+.3f} freq {s['freq']:.3f} PF {s['pf']:.2f}")
        try:
            S2, fr2, g2 = load(a.run_dir, a.confirm)
            s2 = score(decide(S2, m, b, g2), fr2, cost)
            print(f"same setting on {a.confirm}: exp {s2.get('exp', float('nan')):+.3f} freq {s2['freq']:.3f} n {s2['n']} PF {s2.get('pf', float('nan')):.2f}")
        except FileNotFoundError:
            pass
        print(f"apply with: --set agent.margin={m} --set agent.no_trade_bias={b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
