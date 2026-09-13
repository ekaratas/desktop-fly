"""Raw market signals ([C] trading engineering; computed strictly causally).

Every column at row t uses only bars <= t. Nothing here looks forward; labels
live in trader/labels and are kept in separate columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_VERSION = "f2"


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def compute_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]
    logc = np.log(c)
    ret1 = logc.diff()
    a = atr(df, 14)
    atr_rel = a / c                                     # ATR as fraction of price
    f = pd.DataFrame(index=df.index)
    # --- motion (visual-motion-like) ---
    f["ret1"] = ret1
    f["ret4"] = logc.diff(4)
    f["ret12"] = logc.diff(12)
    f["velocity"] = ret1.ewm(span=6, adjust=False).mean() / atr_rel
    f["acceleration"] = f["velocity"].diff(3)
    f["trend_slope"] = (logc - logc.rolling(24).mean()) / atr_rel
    hi48, lo48 = h.rolling(48).max(), l.rolling(48).min()
    f["breakout"] = np.where(c > hi48.shift(1), (c - hi48.shift(1)) / a,
                             np.where(c < lo48.shift(1), (c - lo48.shift(1)) / a, 0.0))
    # --- attraction (odor-like) ---
    f["volume_z"] = (np.log(v + 1) - np.log(v + 1).rolling(48).mean()) / np.log(v + 1).rolling(48).std()
    f["volume_accel"] = np.log(v + 1).diff().ewm(span=3, adjust=False).mean()
    tb = df["taker_buy_volume"]
    f["taker_imbalance"] = ((2 * tb - v) / v).ewm(span=6, adjust=False).mean()
    typical = (h + l + c) / 3
    vwap24 = (typical * v).rolling(24).sum() / v.rolling(24).sum()
    f["vwap_dist"] = (c - vwap24) / a
    f["continuation"] = np.sign(f["ret4"]) * np.sign(f["ret12"]) * (f["ret4"].abs() / atr_rel)
    # --- threat (looming-like) ---
    rv6 = ret1.rolling(6).std()
    rv48 = ret1.rolling(48).std()
    f["vol_expansion"] = np.log(rv6 / rv48)
    f["range_shock"] = ((h - l) / a).clip(0, 10)
    f["reversal"] = -(ret1 * f["ret4"].shift(1)) / atr_rel**2          # move against the recent drift
    upper_wick = (h - np.maximum(o, c)) / (h - l).replace(0, np.nan)
    lower_wick = (np.minimum(o, c) - l) / (h - l).replace(0, np.nan)
    f["rejection"] = (upper_wick - lower_wick).fillna(0.0)
    f["drawdown_from_high"] = (c / h.rolling(48).max() - 1) / atr_rel
    f["runup_from_low"] = (c / l.rolling(48).min() - 1) / atr_rel
    # --- context (environmental state) ---
    f["atr_regime"] = np.log(atr_rel / atr_rel.rolling(240).mean())
    f["compression"] = np.log((h.rolling(12).max() - l.rolling(12).min()) / (a * 12))
    f["htf_direction"] = (logc - logc.rolling(168).mean()) / atr_rel        # ~1 week on 1h
    f["activity"] = np.log(df["count"] + 1) - np.log(df["count"] + 1).rolling(168).mean()
    f["body_ratio"] = ((c - o).abs() / (h - l).replace(0, np.nan)).fillna(0.0)
    f["atr_rel"] = atr_rel
    # --- extended senses (funding / premium / open interest / positioning); NaN when the
    #     source table is absent — encoded as silent PNs, never invalidating the bar ---
    if "funding_rate" in df:
        fr_ = df["funding_rate"]
        f["funding"] = fr_ * 1e3
        f["funding_trend"] = (fr_.rolling(9).mean() - fr_.rolling(90).mean()) * 1e3     # 3 vs 30 days of 8h events
    if "premium_index" in df:
        f["premium"] = df["premium_index"] * 1e3
        f["premium_accel"] = f["premium"].diff(3)
    if "open_interest" in df:
        loi = np.log(df["open_interest"].where(df["open_interest"] > 0))
        f["oi_change_1"] = loi.diff()
        f["oi_change_24"] = loi.diff(24)
        f["oi_vs_price"] = loi.diff(6) * np.sign(logc.diff(6))          # OI building with vs against the move
    if "top_ls_positions" in df:
        f["top_ls"] = np.log(df["top_ls_positions"].where(df["top_ls_positions"] > 0))
    if "global_ls_accounts" in df:
        f["crowd_ls"] = np.log(df["global_ls_accounts"].where(df["global_ls_accounts"] > 0))
    if "taker_ls_ratio" in df:
        f["taker_ls"] = np.log(df["taker_ls_ratio"].where(df["taker_ls_ratio"] > 0)).ewm(span=6, adjust=False).mean()
    return f.replace([np.inf, -np.inf], np.nan)


CORE_CHANNELS = {  # [B] bio-inspired grouping: which fly sense each signal group stands in for
    "motion":     ["ret1", "ret4", "ret12", "velocity", "acceleration", "trend_slope", "breakout"],
    "attraction": ["volume_z", "volume_accel", "taker_imbalance", "vwap_dist", "continuation"],
    "threat":     ["vol_expansion", "range_shock", "reversal", "rejection", "drawdown_from_high", "runup_from_low"],
    "context":    ["atr_regime", "compression", "htf_direction", "activity", "body_ratio"],
}
EXTENDED_CHANNELS = {  # [B] extra senses from derivatives-market tables; present only when the data is
    "positioning": ["oi_change_1", "oi_change_24", "oi_vs_price", "top_ls", "crowd_ls", "taker_ls"],   # crowd / pheromone-like
    "pressure":    ["funding", "funding_trend", "premium", "premium_accel"],                             # thermal-like cost gradient
}
CHANNELS = {**CORE_CHANNELS, **EXTENDED_CHANNELS}
CORE_FEATURES = [c for cols in CORE_CHANNELS.values() for c in cols]
EXTENDED_FEATURES = [c for cols in EXTENDED_CHANNELS.values() for c in cols]
ALL_FEATURES = CORE_FEATURES + EXTENDED_FEATURES


def available_features(f) -> list[str]:
    """ALL_FEATURES restricted to columns present in the computed frame."""
    return [c for c in ALL_FEATURES if c in f.columns]
