import io
import zipfile

import pandas as pd

from trader.data.binance_archive import ArchiveSpec, KLINE_COLUMNS, month_range, parse_kline_zip


def _zip(rows: str, name="BTCUSDT-1h-2023-01.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, rows)
    return buf.getvalue()


def test_url_and_month_range():
    s = ArchiveSpec("BTCUSDT", "1h", "um")
    assert s.url("2023-01") == "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2023-01.zip"
    assert month_range("2022-11", "2023-02") == ["2022-11", "2022-12", "2023-01", "2023-02"]


def test_parse_with_and_without_header_and_microseconds():
    row = "1672531200000,16500,16600,16400,16550,100,1672534799999,1650000,500,60,990000,0\n"
    a = parse_kline_zip(_zip(row))
    b = parse_kline_zip(_zip(",".join(KLINE_COLUMNS) + "\n" + row))
    pd.testing.assert_frame_equal(a, b)
    assert a.index[0] == pd.Timestamp("2023-01-01", tz="UTC") and a["close"].iloc[0] == 16550
    us = row.replace("1672531200000", "1672531200000000")
    c = parse_kline_zip(_zip(us))
    assert c.index[0] == pd.Timestamp("2023-01-01", tz="UTC")


def test_asof_merge_handles_mixed_timestamp_resolutions():
    import numpy as np
    from trader.data.binance_extra import attach_extras
    from trader.data.synthetic import make_synthetic_klines

    bars = make_synthetic_klines(200)
    bars.index = bars.index.astype("datetime64[us, UTC]")                      # kline side in µs
    ft = pd.to_datetime(np.arange(0, 200, 8) * 3600_000 + bars.index[0].value // 10**6, unit="ms", utc=True)
    funding = pd.DataFrame({"time": ft.astype("datetime64[ms, UTC]"), "funding_rate": np.arange(len(ft), dtype=float)})
    metrics = pd.DataFrame({"time": ft.astype("datetime64[ms, UTC]"), "open_interest": np.ones(len(ft))})
    out = attach_extras(bars, "1h", funding=funding, metrics=metrics)
    assert out["funding_rate"].notna().sum() > 0 and out["open_interest"].notna().sum() > 0
