"""Causal normalization ([C]). Rolling statistics use only past rows (shifted by one)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def causal_zscore(f: pd.DataFrame, window: int = 500, clip: float = 4.0, min_periods: int = 100) -> pd.DataFrame:
    mu = f.rolling(window, min_periods=min_periods).mean().shift(1)
    sd = f.rolling(window, min_periods=min_periods).std().shift(1)
    z = (f - mu) / sd.replace(0, np.nan)
    return z.clip(-clip, clip)
