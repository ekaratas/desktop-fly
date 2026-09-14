"""Unified market-data entry point ([C])."""
from __future__ import annotations

import hashlib

import pandas as pd

from .binance_archive import ArchiveSpec, load_klines
from .binance_extra import attach_extras, load_funding, load_metrics, load_premium_index
from .synthetic import make_synthetic_extras, make_synthetic_klines


def load_market_data(cfg: dict, synthetic: bool = False, synthetic_bars: int = 30000) -> pd.DataFrame:
    d = cfg["data"]
    extras = d.get("extras", False)          # False | True | list of {"funding","premium","metrics"}
    wanted = {"funding", "premium", "metrics"} if extras is True else set(extras or [])
    if synthetic:
        bars = make_synthetic_klines(synthetic_bars, seed=cfg.get("seed", 0), start=d["start"] + "-01", timeframe=d["timeframe"])
        return make_synthetic_extras(bars, seed=cfg.get("seed", 0)) if wanted else bars
    spec = ArchiveSpec(d["symbol"], d["timeframe"], d.get("market", "um"))
    cache = d.get("cache_dir", "cache")
    bars = load_klines(spec, d["start"], d["end"], cache, download=True)
    if not wanted:
        return bars
    funding = load_funding(d["symbol"], d["start"], d["end"], cache, spec.market) if "funding" in wanted else None
    premium = load_premium_index(d["symbol"], d["timeframe"], d["start"], d["end"], cache, spec.market) if "premium" in wanted else None
    metrics = load_metrics(d["symbol"], d["start"], d["end"], cache, spec.market) if "metrics" in wanted else None
    bars = attach_extras(bars, d["timeframe"], funding, premium, metrics)
    for name, tbl in (("funding", funding), ("premium", premium), ("metrics", metrics)):
        if name in wanted:
            print(f"[extras] {name}: {'absent' if tbl is None else f'{len(tbl)} rows'}")
    return bars


def data_fingerprint(df: pd.DataFrame) -> str:
    """Stable hash of the bars used, stored with every run for reproducibility."""
    h = hashlib.sha256()
    h.update(str(df.index[0]).encode()); h.update(str(df.index[-1]).encode()); h.update(str(len(df)).encode())
    h.update(pd.util.hash_pandas_object(df[["open", "high", "low", "close", "volume"]].round(8)).values.tobytes())
    return h.hexdigest()[:16]
