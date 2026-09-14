"""Extra Binance futures archive tables ([C]): funding, premium index, open-interest metrics.

All from https://data.binance.vision/data/futures/um/… and merged **as-of** onto the
kline index using each bar's close time, so a bar only sees values published at or
before its own close (no look-ahead).

  fundingRate        monthly  calc_time, funding_interval_hours, last_funding_rate  (8h events)
  premiumIndexKlines monthly  kline layout, per timeframe (premium = perp − index, funding driver)
  metrics            daily    5-min: sum_open_interest, sum_open_interest_value,
                              count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio,
                              count_long_short_ratio, sum_taker_long_short_vol_ratio
                              (published from late 2021 onward; earlier months are simply absent)

Each source is optional: a 404 is logged and skipped, and downstream features fall
back to NaN → silent projection neurons, without invalidating the bar.
"""
from __future__ import annotations

import io
import os
import sys
import zipfile
from datetime import date, timedelta

import pandas as pd

from .binance_archive import BASE, KLINE_COLUMNS, download_month, month_range, _fetch


def _read_zip_csv(path: str, names: list[str] | None = None) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0])
    first = raw.split(b"\n", 1)[0].decode(errors="ignore")
    has_header = any(c.isalpha() for c in first.split(",")[0])
    if has_header:
        return pd.read_csv(io.BytesIO(raw))
    return pd.read_csv(io.BytesIO(raw), header=None, names=names)


def _to_utc(series: pd.Series) -> pd.Series:
    """Parse ms/µs epoch or ISO strings to tz-aware UTC at nanosecond resolution.

    pandas ≥ 2 keeps the resolution it parsed (ms vs µs) and merge_asof refuses to
    join keys of different resolutions, so everything is cast to [ns]."""
    t = pd.to_numeric(series, errors="coerce")
    if t.notna().mean() > 0.9:
        unit = "us" if t.max() > 1e14 else "ms"
        out = pd.to_datetime(t, unit=unit, utc=True)
    else:
        out = pd.to_datetime(series, utc=True, errors="coerce")
    return out.astype("datetime64[ns, UTC]")


class _MonthlySpec:
    """Minimal spec for download_month: url(month) from a format string with {m}."""

    def __init__(self, fmt: str):
        self.fmt = fmt

    def url(self, month: str) -> str:
        return self.fmt.format(m=month)


def _fetch_months(spec: _MonthlySpec, start: str, end: str, cache_dir: str, tag: str) -> list[str]:
    paths = []
    for m in month_range(start, end):
        try:
            p = download_month(spec, m, cache_dir)  # type: ignore[arg-type]
        except Exception as e:  # network/other: report and continue
            print(f"[{tag}] {m}: {e}", file=sys.stderr); continue
        if p is None:
            print(f"[{tag}] {m}: not published, skipped", file=sys.stderr); continue
        paths.append(p)
    return paths


def load_funding(symbol: str, start: str, end: str, cache_dir: str, market="um") -> pd.DataFrame | None:
    spec = _MonthlySpec(f"{BASE}/futures/{market}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{{m}}.zip")
    paths = _fetch_months(spec, start, end, os.path.join(cache_dir, "funding"), "funding")
    if not paths:
        return None
    df = pd.concat(_read_zip_csv(p, ["calc_time", "funding_interval_hours", "last_funding_rate"]) for p in paths)
    df.columns = [c.strip().lower() for c in df.columns]
    df["time"] = _to_utc(df["calc_time"])
    out = df[["time", "last_funding_rate"]].rename(columns={"last_funding_rate": "funding_rate"})
    out["funding_rate"] = pd.to_numeric(out["funding_rate"], errors="coerce")
    return out.dropna().sort_values("time").drop_duplicates("time")


def load_premium_index(symbol: str, timeframe: str, start: str, end: str, cache_dir: str, market="um") -> pd.DataFrame | None:
    spec = _MonthlySpec(f"{BASE}/futures/{market}/monthly/premiumIndexKlines/{symbol}/{timeframe}/{symbol}-{timeframe}-{{m}}.zip")
    paths = _fetch_months(spec, start, end, os.path.join(cache_dir, "premium", timeframe), "premium")
    if not paths:
        return None
    df = pd.concat(_read_zip_csv(p, KLINE_COLUMNS) for p in paths)
    df.columns = [c.strip().lower() for c in df.columns][: len(df.columns)]
    df = df.rename(columns={df.columns[0]: "open_time", df.columns[4]: "premium_close"})
    out = pd.DataFrame({"open_time": _to_utc(df["open_time"]), "premium_index": pd.to_numeric(df["premium_close"], errors="coerce")})
    return out.dropna().sort_values("open_time").drop_duplicates("open_time")


def load_metrics(symbol: str, start: str, end: str, cache_dir: str, market="um", earliest: str = "2021-12-01") -> pd.DataFrame | None:
    """Daily 5-minute metrics files. Starts at max(start, earliest) to avoid ~700 needless 404s."""
    d0 = max(date.fromisoformat(start[:7] + "-01"), date.fromisoformat(earliest))
    y1, m1 = (int(x) for x in end.split("-")[:2])
    d1 = min(date.today() - timedelta(days=1), (date(y1 + (m1 == 12), (m1 % 12) + 1, 1) - timedelta(days=1)))
    cdir = os.path.join(cache_dir, "metrics"); os.makedirs(cdir, exist_ok=True)
    frames, missing = [], 0
    d = d0
    while d <= d1:
        key = d.isoformat()
        local = os.path.join(cdir, f"{symbol}-metrics-{key}.zip")
        if not os.path.exists(local):
            url = f"{BASE}/futures/{market}/daily/metrics/{symbol}/{symbol}-metrics-{key}.zip"
            try:
                blob = _fetch(url)
                with open(local, "wb") as f:
                    f.write(blob)
            except Exception as e:
                code = getattr(e, "code", None)
                missing += 1
                if code != 404:
                    print(f"[metrics] {key}: {e}", file=sys.stderr)
                d += timedelta(days=1); continue
        try:
            frames.append(_read_zip_csv(local))
        except Exception as e:
            print(f"[metrics] {key}: unreadable ({e})", file=sys.stderr)
        d += timedelta(days=1)
    if missing:
        print(f"[metrics] {missing} daily files not published, skipped", file=sys.stderr)
    if not frames:
        return None
    df = pd.concat(frames)
    df.columns = [c.strip().lower() for c in df.columns]
    df["time"] = _to_utc(df["create_time"])
    keep = {"sum_open_interest": "open_interest", "sum_open_interest_value": "open_interest_value",
            "count_toptrader_long_short_ratio": "top_ls_accounts", "sum_toptrader_long_short_ratio": "top_ls_positions",
            "count_long_short_ratio": "global_ls_accounts", "sum_taker_long_short_vol_ratio": "taker_ls_ratio"}
    cols = {k: v for k, v in keep.items() if k in df.columns}
    out = df[["time", *cols]].rename(columns=cols)
    for c in cols.values():
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.sort_values("time").drop_duplicates("time")


def attach_extras(bars: pd.DataFrame, timeframe: str, funding=None, premium=None, metrics=None) -> pd.DataFrame:
    """As-of merge onto bars using each bar's close time (open_time + timeframe)."""
    from .synthetic import _tf_to_timedelta

    out = bars.copy()
    out.index = pd.DatetimeIndex(out.index).astype("datetime64[ns, UTC]")
    out.index.name = "open_time"
    close_time = out.index + pd.Timedelta(_tf_to_timedelta(timeframe))
    key = pd.DataFrame({"close_time": close_time}, index=out.index)
    if funding is not None and len(funding):
        f = funding.rename(columns={"time": "close_time"}).sort_values("close_time")
        f["close_time"] = f["close_time"].astype("datetime64[ns, UTC]")
        m = pd.merge_asof(key.reset_index(), f, on="close_time", direction="backward").set_index("open_time")
        out["funding_rate"] = m["funding_rate"].to_numpy()
    if premium is not None and len(premium):
        p = premium.set_index(premium["open_time"].astype("datetime64[ns, UTC]"))["premium_index"]
        out["premium_index"] = p.reindex(out.index).to_numpy()          # same bar grid, no look-ahead
    if metrics is not None and len(metrics):
        mm = metrics.rename(columns={"time": "close_time"}).sort_values("close_time")
        mm["close_time"] = mm["close_time"].astype("datetime64[ns, UTC]")
        tol = max(pd.Timedelta("2h"), pd.Timedelta(_tf_to_timedelta(timeframe)) // 2)
        m = pd.merge_asof(key.reset_index(), mm, on="close_time", direction="backward",
                          tolerance=tol).set_index("open_time")
        for c in mm.columns:
            if c != "close_time":
                out[c] = m[c].to_numpy()
    return out


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Download funding / premium index / metrics into the cache.")
    p.add_argument("symbol"); p.add_argument("timeframe"); p.add_argument("start"); p.add_argument("end", nargs="?", default=date.today().strftime("%Y-%m"))
    p.add_argument("--cache", default="cache"); p.add_argument("--no-metrics", action="store_true")
    a = p.parse_args(argv)
    f = load_funding(a.symbol, a.start, a.end, a.cache); print("funding:", None if f is None else f.shape)
    pr = load_premium_index(a.symbol, a.timeframe, a.start, a.end, a.cache); print("premium:", None if pr is None else pr.shape)
    if not a.no_metrics:
        m = load_metrics(a.symbol, a.start, a.end, a.cache); print("metrics:", None if m is None else (m.shape, m["time"].min(), m["time"].max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
