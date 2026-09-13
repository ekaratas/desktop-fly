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
