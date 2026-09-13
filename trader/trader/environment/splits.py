"""Time-ordered splits ([C]). Never random: every test bar is later than every train bar."""
from __future__ import annotations

import numpy as np
import pandas as pd


def time_split(index: pd.DatetimeIndex, cfg_splits: dict) -> dict[str, pd.Series]:
    """Boolean masks per split from [start, end] date strings in the config."""
    masks = {}
    for name, (a, b) in cfg_splits.items():
        a_ts = pd.Timestamp(a, tz="UTC")
        b_ts = pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1)
        masks[name] = pd.Series((index >= a_ts) & (index < b_ts), index=index)
    return masks


def walk_forward_folds(index: pd.DatetimeIndex, train_years: int = 3, test_months: int = 12, start_year: int | None = None):
    """Yield (train_mask, test_mask) pairs: expanding-origin yearly folds."""
    years = sorted(set(index.year))
    start_year = start_year or years[0]
    for y in range(start_year + train_years, years[-1] + 1):
        tr = pd.Series((index.year >= start_year) & (index.year < y), index=index)
        te = pd.Series((index.year == y), index=index)
        if te.sum() > 0 and tr.sum() > 0:
            yield y, tr, te


def purge_boundary(mask: pd.Series, horizon: int) -> pd.Series:
    """Drop the last `horizon` bars of a training mask so no label overlaps the next split."""
    m = mask.copy()
    idx = m[m].index
    if len(idx) > horizon:
        m.loc[idx[-horizon:]] = False
    return m


def proportional_split(index: pd.DatetimeIndex, fractions=(0.6, 0.2, 0.2)) -> dict[str, pd.Series]:
    """Fallback when the configured date windows do not cover the data (synthetic/short runs):
    contiguous train/validation/test blocks in time order."""
    n = len(index)
    a = int(n * fractions[0]); b = a + int(n * fractions[1])
    pos = np.arange(n)
    return {"train": pd.Series(pos < a, index=index), "validation": pd.Series((pos >= a) & (pos < b), index=index),
            "test": pd.Series(pos >= b, index=index)}
