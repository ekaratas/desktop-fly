"""Synthetic OHLCV fixture ([C]) for offline tests and sandboxes without archive access.

A regime-switching random walk with volatility clustering, volume that co-moves
with absolute returns and occasional momentum bursts, so that features, labels and
the full pipeline can be exercised end to end. It carries no information about
real markets; results on it are only a software check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_klines(n_bars: int = 20000, seed: int = 0, start: str = "2020-01-01",
                          timeframe: str = "1h", price0: float = 10000.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # regime: 0 chop, 1 trend up, 2 trend down; sticky Markov chain
    P = np.array([[0.985, 0.0075, 0.0075], [0.02, 0.975, 0.005], [0.02, 0.005, 0.975]])
    regime = np.zeros(n_bars, dtype=int)
    for i in range(1, n_bars):
        regime[i] = rng.choice(3, p=P[regime[i - 1]])
    drift = np.array([0.0, 0.0006, -0.0006])[regime]
    # GARCH-like volatility clustering
    sigma = np.empty(n_bars)
    sigma[0] = 0.006
    eps = rng.standard_normal(n_bars)
    for i in range(1, n_bars):
        sigma[i] = np.sqrt(1e-6 + 0.85 * sigma[i - 1] ** 2 + 0.12 * (sigma[i - 1] * eps[i - 1]) ** 2)
    r = drift + sigma * eps
    close = price0 * np.exp(np.cumsum(r))
    open_ = np.concatenate([[price0], close[:-1]])
    wick = np.abs(rng.standard_normal(n_bars)) * sigma * close * 0.8
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - np.abs(rng.standard_normal(n_bars)) * sigma * close * 0.8
    base_vol = 5000.0
    volume = base_vol * (1 + 40 * np.abs(r)) * np.exp(0.3 * rng.standard_normal(n_bars))
    taker_buy = volume * np.clip(0.5 + 6 * r + 0.05 * rng.standard_normal(n_bars), 0.05, 0.95)
    idx = pd.date_range(start, periods=n_bars, freq=pd.Timedelta(_tf_to_timedelta(timeframe)), tz="UTC")
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
        "quote_volume": volume * close, "count": np.maximum(1, (volume / 2).astype(int)),
        "taker_buy_volume": taker_buy, "taker_buy_quote_volume": taker_buy * close,
    }, index=idx)
    df.index.name = "open_time"
    return df


def _tf_to_timedelta(tf: str) -> str:
    unit = tf[-1]
    n = int(tf[:-1])
    return {"m": f"{n}min", "h": f"{n}h", "d": f"{n}D"}[unit]


def make_synthetic_extras(bars: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Fake funding / premium / OI / positioning columns (software check only)."""
    rng = np.random.default_rng(seed + 99)
    n = len(bars)
    out = bars.copy()
    r = np.log(out["close"]).diff().fillna(0.0).to_numpy()
    out["premium_index"] = 0.0002 * np.tanh(np.convolve(r, np.ones(8) / 8, mode="same") * 50) + 0.00005 * rng.standard_normal(n)
    out["funding_rate"] = pd.Series(out["premium_index"]).rolling(8).mean().fillna(0.0001).to_numpy()
    out["open_interest"] = 50000 * np.exp(np.cumsum(0.002 * rng.standard_normal(n)))
    out["open_interest_value"] = out["open_interest"] * out["close"]
    out["top_ls_positions"] = np.exp(0.1 * rng.standard_normal(n))
    out["global_ls_accounts"] = np.exp(0.2 * rng.standard_normal(n))
    out["taker_ls_ratio"] = np.exp(0.15 * rng.standard_normal(n) + 5 * r)
    return out
