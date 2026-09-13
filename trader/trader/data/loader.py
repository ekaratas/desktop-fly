"""Unified market-data entry point ([C])."""
from __future__ import annotations

import hashlib

import pandas as pd

from .binance_archive import ArchiveSpec, load_klines
from .synthetic import make_synthetic_klines


def load_market_data(cfg: dict, synthetic: bool = False, synthetic_bars: int = 30000) -> pd.DataFrame:
    d = cfg["data"]
    if synthetic:
        return make_synthetic_klines(synthetic_bars, seed=cfg.get("seed", 0), start=d["start"] + "-01",
                                     timeframe=d["timeframe"])
    spec = ArchiveSpec(d["symbol"], d["timeframe"], d.get("market", "um"))
    return load_klines(spec, d["start"], d["end"], d.get("cache_dir", "cache"), download=True)


def data_fingerprint(df: pd.DataFrame) -> str:
    """Stable hash of the bars used, stored with every run for reproducibility."""
    h = hashlib.sha256()
    h.update(str(df.index[0]).encode()); h.update(str(df.index[-1]).encode()); h.update(str(len(df)).encode())
    h.update(pd.util.hash_pandas_object(df[["open", "high", "low", "close", "volume"]].round(8)).values.tobytes())
    return h.hexdigest()[:16]
