"""Sensory encoding ([B] bio-inspired mapping, designed by us).

Each normalized feature drives a pair of projection-neuron (PN) rates: an ON cell
firing for positive deviations and an OFF cell for negative ones (rectified,
saturating). This mimics opponent coding in fly sensory pathways but is *not*
a measured Drosophila circuit; the feature→channel assignment is a design table
(trader.features.raw.CHANNELS) and can be edited freely.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .raw import ALL_FEATURES, CHANNELS

SENSORY_VERSION = "s1"


def pn_names() -> list[str]:
    return [f"{c}_{s}" for c in ALL_FEATURES for s in ("on", "off")]


def encode_rates(z: pd.DataFrame, max_rate_hz: float = 200.0, gain: float = 1.0) -> np.ndarray:
    """(T, 2*F) PN firing rates in Hz from clipped z-scores. NaN → 0 Hz (silent)."""
    x = z[ALL_FEATURES].to_numpy(dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0)
    on = np.tanh(gain * np.maximum(x, 0.0) / 2.0)
    off = np.tanh(gain * np.maximum(-x, 0.0) / 2.0)
    rates = np.empty((x.shape[0], 2 * x.shape[1]))
    rates[:, 0::2] = on
    rates[:, 1::2] = off
    return rates * max_rate_hz


def channel_summary(z_row: pd.Series) -> dict[str, float]:
    """Mean signed activation per channel for the explainability log."""
    return {ch: float(np.nanmean(z_row[cols].to_numpy(dtype=float))) if np.isfinite(z_row[cols].to_numpy(dtype=float)).any() else 0.0
            for ch, cols in CHANNELS.items()}
