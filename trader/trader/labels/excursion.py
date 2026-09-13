"""Outcome labels from future price action ([C]; modular, volatility-normalized).

Labels are derived automatically from bars t+1..t+N and MUST only be used as
targets/rewards, never as inputs. Excursions are measured in ATR(t) units.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_VERSION = "l1"
LONG, SHORT, NO_TRADE = 0, 1, 2
ACTION_NAMES = {LONG: "LONG", SHORT: "SHORT", NO_TRADE: "NO_TRADE"}


def future_excursions(df: pd.DataFrame, atr_abs: pd.Series, horizon: int) -> pd.DataFrame:
    """MFE/MAE for a long entered at close(t), over bars t+1..t+horizon, in ATR(t) units.

    Uses a reversed rolling max/min so no Python loop is needed; the last `horizon`
    rows are NaN (their future is unknown) and are dropped from training.
    """
    c = df["close"]
    fut_high = df["high"][::-1].rolling(horizon, min_periods=horizon).max()[::-1].shift(-1)
    fut_low = df["low"][::-1].rolling(horizon, min_periods=horizon).min()[::-1].shift(-1)
    fut_close = c.shift(-horizon)
    out = pd.DataFrame(index=df.index)
    out["mfe_long"] = (fut_high - c) / atr_abs
    out["mae_long"] = (c - fut_low) / atr_abs
    out["mfe_short"] = out["mae_long"]
    out["mae_short"] = out["mfe_long"]
    out["fwd_return_atr"] = (fut_close - c) / atr_abs           # signed, ATR units
    out["fwd_return"] = np.log(fut_close / c)
    return out


def opportunity_labels(exc: pd.DataFrame, edge_threshold_atr: float = 1.0, ratio_threshold: float = 1.8) -> pd.Series:
    """LONG if upside excursion dominates, SHORT if downside dominates, else NO_TRADE."""
    up, dn = exc["mfe_long"], exc["mae_long"]
    long_ok = (up >= edge_threshold_atr) & (up >= ratio_threshold * dn.clip(lower=0.1))
    short_ok = (dn >= edge_threshold_atr) & (dn >= ratio_threshold * up.clip(lower=0.1))
    lab = pd.Series(NO_TRADE, index=exc.index, dtype=int)
    lab[long_ok & ~short_ok] = LONG
    lab[short_ok & ~long_ok] = SHORT
    lab[exc["mfe_long"].isna()] = -1                              # unknown future
    return lab
