"""Binance public data archive downloader ([C] trading engineering).

Reads monthly kline zips from https://data.binance.vision (USDⓈ-M futures by
default), verifies the published SHA-256 CHECKSUM, caches the raw zip and returns
one clean UTC-indexed DataFrame. No API key, no private endpoint, no trading.

Layout: data/futures/um/monthly/klines/<SYMBOL>/<TF>/<SYMBOL>-<TF>-<YYYY-MM>.zip
Newer archive files carry a header row and some carry microsecond timestamps;
both variants are handled.
"""
from __future__ import annotations

import hashlib
import io
import os
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date

import pandas as pd

BASE = "https://data.binance.vision/data"
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
KEEP = ["open", "high", "low", "close", "volume", "quote_volume", "count",
        "taker_buy_volume", "taker_buy_quote_volume"]


@dataclass(frozen=True)
class ArchiveSpec:
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    market: str = "um"  # um = USDⓈ-M futures, cm = COIN-M, spot = spot

    def url(self, month: str) -> str:
        if self.market == "spot":
            root = f"{BASE}/spot/monthly/klines"
        else:
            root = f"{BASE}/futures/{self.market}/monthly/klines"
        return f"{root}/{self.symbol}/{self.timeframe}/{self.symbol}-{self.timeframe}-{month}.zip"


def month_range(start: str, end: str) -> list[str]:
    """Inclusive list of 'YYYY-MM' strings."""
    y0, m0 = (int(x) for x in start.split("-")[:2])
    y1, m1 = (int(x) for x in end.split("-")[:2])
    out = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def _fetch(url: str, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "trader-fly/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def download_month(spec: ArchiveSpec, month: str, cache_dir: str, verify: bool = True) -> str | None:
    """Download one monthly zip into cache_dir; returns local path or None if absent (404)."""
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, os.path.basename(spec.url(month)))
    if os.path.exists(local):
        return local
    try:
        blob = _fetch(spec.url(month))
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        if e.code == 404:
            return None
        raise
    if verify:
        checksum_line = _fetch(spec.url(month) + ".CHECKSUM").decode().split()[0]
        digest = hashlib.sha256(blob).hexdigest()
        if digest != checksum_line:
            raise RuntimeError(f"checksum mismatch for {month}: {digest} != {checksum_line}")
    with open(local, "wb") as f:
        f.write(blob)
    return local


def parse_kline_zip(path_or_bytes) -> pd.DataFrame:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        zf = zipfile.ZipFile(io.BytesIO(path_or_bytes))
    else:
        zf = zipfile.ZipFile(path_or_bytes)
    with zf as z:
        name = z.namelist()[0]
        raw = z.read(name)
    has_header = raw[:20].lower().startswith(b"open_time")
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None, names=None if has_header else KLINE_COLUMNS)
    df.columns = KLINE_COLUMNS[: len(df.columns)]
    return normalize_klines(df)


def normalize_klines(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.to_numeric(df["open_time"], errors="coerce")
    unit = "us" if t.max() > 1e14 else "ms"
    idx = pd.to_datetime(t, unit=unit, utc=True).astype("datetime64[ns, UTC]")
    out = df[KEEP].apply(pd.to_numeric, errors="coerce")
    out.index = idx
    out.index.name = "open_time"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def load_klines(spec: ArchiveSpec, start: str, end: str, cache_dir: str, download: bool = True) -> pd.DataFrame:
    """Load [start, end] months from cache (downloading missing months when allowed)."""
    frames = []
    for month in month_range(start, end):
        local = os.path.join(cache_dir, os.path.basename(spec.url(month)))
        if not os.path.exists(local):
            if not download:
                continue
            got = download_month(spec, month, cache_dir)
            if got is None:
                print(f"[archive] {month}: not published, skipped", file=sys.stderr)
                continue
        frames.append(parse_kline_zip(local))
    if not frames:
        raise FileNotFoundError(f"no kline data for {spec.symbol} {spec.timeframe} in {cache_dir}")
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="last")].sort_index()


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Download Binance monthly klines into a local cache.")
    p.add_argument("symbol")
    p.add_argument("timeframe")
    p.add_argument("start", help="YYYY-MM")
    p.add_argument("end", nargs="?", default=date.today().strftime("%Y-%m"), help="YYYY-MM")
    p.add_argument("--market", default="um")
    p.add_argument("--cache", default="cache")
    a = p.parse_args(argv)
    spec = ArchiveSpec(a.symbol, a.timeframe, a.market)
    df = load_klines(spec, a.start, a.end, a.cache)
    print(f"{spec.symbol} {spec.timeframe}: {len(df)} bars {df.index[0]} → {df.index[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
