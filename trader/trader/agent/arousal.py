"""Arousal ([B]): "is something worth attending to happening?"

An EMA of volatility expansion, volume surprise and |acceleration| (all causal
z-scores). It gates how strongly stimuli are presented and, below a threshold,
skips the decision entirely (the organism rests: NO_TRADE). It never chooses a
direction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AROUSAL_INPUTS = ["vol_expansion", "volume_z", "acceleration", "range_shock", "breakout"]


def arousal_series(z: pd.DataFrame, tau_bars: int = 6) -> pd.Series:
    x = z[AROUSAL_INPUTS].abs().mean(axis=1)
    x = x.ewm(halflife=tau_bars, adjust=False).mean()
    return (np.tanh(x / 1.5)).fillna(0.0)          # 0..1


def sensory_gain(arousal: float, lo: float = 0.6, hi: float = 1.4) -> float:
    return float(lo + (hi - lo) * arousal)
