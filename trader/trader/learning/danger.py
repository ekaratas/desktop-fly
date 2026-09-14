"""Danger objective: the looming → escape analog ([A] circuit role, [B] mapping, [C] target).

Measured biology ([A]): the fly's LC4/LPLC2 → giant-fiber pathway turns an
expanding (looming) stimulus into an escape; in the mushroom body, PPL1
punishment DANs depress KC→approach-MBON synapses so that a stimulus paired
with harm loses its appeal.

Our mapping ([B]): two MBON populations, `safe` (keep foraging / stay exposed)
and `danger` (retreat). The outcome of bar t is the largest excursion in
*either* direction over the horizon, in ATR units — how violently price moved
regardless of direction. If it exceeded the danger threshold the `safe`
population is punished (it promoted staying exposed into a storm); if the
horizon stayed calm the `danger` population is punished (a false alarm).

Behavioral readout: the gap share(danger) − share(safe) is standardized by a slow
causal running mean/std of its own history (`GapNormalizer`, [B]: adaptive
threshold, like habituation) and compared with `alert_z` / `escape_z`. Trading meaning ([C]): ESCAPE = do not
hold or open positions; the metric is how much adverse excursion that avoids.
"""
from __future__ import annotations

import numpy as np

DANGER_POPULATIONS = ("safe", "danger")
CALM, ALERT, ESCAPE = 0, 1, 2
STATE_NAMES = {CALM: "CALM", ALERT: "ALERT", ESCAPE: "ESCAPE"}


def risk_outcome(mfe_long: float, mae_long: float) -> float:
    """Largest excursion in either direction over the horizon, in ATR(t) units."""
    return float(max(mfe_long, mae_long))


def dan_drive_danger(risk: float, threshold: float, false_alarm_scale: float = 1.0) -> np.ndarray:
    """(2,) depression drive for (safe, danger).

    risk > threshold → punish `safe` by the excess; else punish `danger` by the shortfall
    (scaled, since calm bars are the majority and the two DANs should fire about
    equally often in expectation)."""
    if not np.isfinite(risk):
        return np.zeros(2)
    d = np.zeros(2)
    if risk > threshold:
        d[0] = risk - threshold
    else:
        d[1] = false_alarm_scale * (threshold - risk)
    return d


def select_state(readout: np.ndarray, alert_margin: float = 0.02, escape_margin: float = 0.06) -> tuple[int, np.ndarray]:
    total = readout.sum()
    share = readout / total if total > 0 else np.array([0.5, 0.5])
    gap = share[1] - share[0]
    if gap >= escape_margin:
        return ESCAPE, share
    if gap >= alert_margin:
        return ALERT, share
    return CALM, share


class GapNormalizer:
    """Causal running standardization of the danger−safe gap ([B], adaptive threshold).

    z_t = (gap_t − mean_{<t}) / std_{<t}; statistics are updated *after* use, so the
    state at t depends only on past gaps. Warm-up: until `min_n` samples, z = 0 (CALM)."""

    def __init__(self, tau_bars: float = 500.0, min_n: int = 50):
        self.alpha = 1.0 / max(1.0, tau_bars)
        self.min_n = min_n
        self.mean, self.var, self.n = 0.0, 0.0, 0

    def __call__(self, gap: float) -> float:
        z = 0.0
        if self.n >= self.min_n and self.var > 0:
            z = (gap - self.mean) / (self.var ** 0.5)
        if self.n == 0:
            self.mean = gap
        else:
            delta = gap - self.mean
            self.mean += self.alpha * delta
            self.var = (1 - self.alpha) * (self.var + self.alpha * delta * delta)
        self.n += 1
        return float(z)


def state_from_z(z: float, alert_z: float = 0.75, escape_z: float = 1.5) -> int:
    if z >= escape_z:
        return ESCAPE
    if z >= alert_z:
        return ALERT
    return CALM
