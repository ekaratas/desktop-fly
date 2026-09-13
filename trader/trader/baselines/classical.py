"""Classical baselines ([C]) on the *same* causal features, labels and time splits.

They answer "what does a conventional learner extract from these stimuli?" so the
organism's behavior can be compared, not just its PnL.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..labels.excursion import ACTION_NAMES, LONG, NO_TRADE, SHORT


def make_baselines(seed: int) -> dict:
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")),
        "hist_gbm": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0,
                                                   class_weight="balanced", random_state=seed),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=300,
                                                             early_stopping=True, random_state=seed)),
    }


def baseline_records(ds, train_mask: pd.Series, eval_mask: pd.Series, seed: int, min_prob: float = 0.45) -> dict[str, list]:
    """Fit each baseline on train bars, decide on eval bars; returns DecisionRecord-like objects."""
    from ..explog.explain import DecisionRecord

    X = ds.z.to_numpy(dtype=np.float64)
    y = ds.outcomes["label"].to_numpy()
    tr = (train_mask & ds.valid).to_numpy()
    ev = (eval_mask & ds.valid).to_numpy()
    out = {}
    for name, model in make_baselines(seed).items():
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[ev])
        classes = list(model.classes_)
        recs = []
        for i, t in enumerate(np.nonzero(ev)[0]):
            p = {ACTION_NAMES[c]: float(proba[i, classes.index(c)]) if c in classes else 0.0 for c in (LONG, SHORT, NO_TRADE)}
            best = max(p, key=p.get)
            action = best if (best == "NO_TRADE" or p[best] >= min_prob) else "NO_TRADE"
            recs.append(DecisionRecord(
                t=str(ds.bars.index[t]), split="baseline", arousal=float(ds.arousal.iloc[t]), channels={}, gated=False,
                kc_active_frac=0.0, pop_drive={}, pop_share={"long": p["LONG"], "short": p["SHORT"], "avoid": p["NO_TRADE"]},
                mbon_spikes=[], action=action, label=ACTION_NAMES[int(y[t])],
                fwd_return_atr=float(ds.outcomes["fwd_return_atr"].iloc[t])))
        out[name] = recs
    return out
