"""Historical replay ([C]): the frozen dataset every module consumes.

`Dataset` bundles causal inputs (z-scores, PN rates, arousal) and *separate* future
outcome columns (labels, forward return). The replay iterator yields bars in time
order; the outcome for bar t is only released `horizon` bars later, exactly as a
live stream would reveal it.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from ..agent.arousal import arousal_series
from ..features.normalize import causal_zscore
from ..features.raw import ALL_FEATURES, atr, compute_raw_features
from ..features.sensory import encode_rates
from ..labels.excursion import future_excursions, opportunity_labels


@dataclass
class Dataset:
    bars: pd.DataFrame
    z: pd.DataFrame               # causal normalized features (inputs)
    pn_rates: np.ndarray          # (T, n_pn) Hz
    arousal: pd.Series
    outcomes: pd.DataFrame        # future-derived columns: mfe/mae, fwd_return_atr, label
    horizon: int
    valid: pd.Series              # rows with complete inputs AND known outcome
    atr_rel: pd.Series            # ATR / price at t (for cost accounting)

    @property
    def n_pn(self) -> int:
        return self.pn_rates.shape[1]


def build_dataset(bars: pd.DataFrame, cfg: dict) -> Dataset:
    raw = compute_raw_features(bars)
    z = causal_zscore(raw[ALL_FEATURES], window=cfg["features"]["norm_window"])
    rates = encode_rates(z, max_rate_hz=cfg["sensory"]["max_rate_hz"])
    ar = arousal_series(z, tau_bars=cfg["arousal"]["tau_bars"])
    hz = cfg["labels"]["horizon_bars"]
    exc = future_excursions(bars, atr(bars), hz)
    exc["label"] = opportunity_labels(exc, cfg["labels"]["edge_threshold_atr"], cfg["labels"]["ratio_threshold"])
    inputs_ok = z.notna().all(axis=1)
    valid = inputs_ok & (exc["label"] >= 0)
    return Dataset(bars, z, rates, ar, exc, hz, valid, raw["atr_rel"])


def replay(ds: Dataset, mask: pd.Series) -> Iterator[tuple[int, list[int]]]:
    """Yield (t, [bars whose outcome just became known]) in time order.

    Outcome of bar i is released at t = i + horizon. Bars at the end of the mask
    whose outcome falls outside are flushed at the end (their future is known in
    the dataset; training only ever uses them after the delay).
    """
    pending: deque[int] = deque()
    for t in np.nonzero(mask.to_numpy())[0]:
        matured = []
        while pending and pending[0] + ds.horizon <= t:
            matured.append(pending.popleft())
        pending.append(int(t))
        yield int(t), matured
    if pending:
        yield -1, list(pending)
