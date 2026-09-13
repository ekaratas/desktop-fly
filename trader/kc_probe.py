#!/usr/bin/env python3
"""Does the Kenyon-cell code carry directional information at all?  ([C] diagnostic)

Fits a linear probe (logistic regression) from recorded KC spike codes to the
sign of the forward return, on the train split, and scores it on validation/test.
If the probe finds an edge the organism's plasticity misses, the problem is the
learning rule; if the probe finds nothing, the problem is upstream (features,
sensory encoding, KC expansion) or there is no edge at this horizon.

    python run_experiment.py configs/... --dump-kc --no-baselines --tag probe
    python kc_probe.py runs/<run_dir>
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression


def load(run_dir, split):
    z = np.load(os.path.join(run_dir, f"kc_{split}.npz"))
    X = (z["codes"] > 0).astype(np.float32)            # binary KC activity
    fr = z["fwd_return_atr"]
    ok = np.isfinite(fr)
    return X[ok], fr[ok]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--C", type=float, default=0.05)
    p.add_argument("--cost", type=float, default=None)
    a = p.parse_args()
    cfg = json.load(open(os.path.join(a.run_dir, "manifest.json")))["config"]
    cost = a.cost if a.cost is not None else cfg["reward"]["cost_atr"]
    Xtr, ftr = load(a.run_dir, "train")
    print(f"train: {Xtr.shape[0]} bars, {Xtr.shape[1]} KCs, mean active {Xtr.mean():.3f}")
    # direction probe: does KC code predict sign(fwd return) beyond chance?
    ytr = (ftr > 0).astype(int)
    clf = LogisticRegression(C=a.C, max_iter=2000).fit(Xtr, ytr)
    for split in ("validation", "test"):
        try:
            X, fr = load(a.run_dir, split)
        except FileNotFoundError:
            continue
        pr = clf.predict_proba(X)[:, 1]
        acc = ((pr > 0.5) == (fr > 0)).mean()
        print(f"\n{split}: {len(fr)} bars, sign accuracy {acc:.3f} (chance ≈ {max((fr>0).mean(), (fr<=0).mean()):.3f})")
        print("  conf-threshold   freq   n     exp(ATR)  win")
        for th in (0.50, 0.53, 0.56, 0.60, 0.65):
            take = np.abs(pr - 0.5) >= th - 0.5
            d = np.where(pr > 0.5, 1.0, -1.0)[take]
            pnl = np.clip(d * fr[take] - cost, -3, 3)
            if len(pnl) == 0:
                print(f"  {th:.2f}            0.000      0        –"); continue
            print(f"  {th:.2f}            {take.mean():.3f} {len(pnl):6d}   {pnl.mean():+.3f}   {(pnl>0).mean():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
