"""Vectorized leaky integrate-and-fire population ([B] model; constants from Sim.swift).

1 ms step, membrane tau 20 ms (decay exp(-1/20) = 0.9512), threshold 1.0,
refractory 2 ms — the same operating point as DesktopFly's FlyWire simulation so
the two simulations share one neuron model. Float64 throughout.
"""
from __future__ import annotations

import numpy as np

DECAY = float(np.exp(-1.0 / 20.0))
THRESHOLD = 1.0
REFRACTORY_MS = 2


class LIFPopulation:
    def __init__(self, n: int, threshold: float | np.ndarray = THRESHOLD, decay: float = DECAY):
        self.n = n
        self.v = np.zeros(n)
        self.refr = np.zeros(n, dtype=np.int64)
        self.threshold = np.broadcast_to(np.asarray(threshold, dtype=np.float64), (n,)).copy()
        self.decay = decay

    def reset(self) -> None:
        self.v[:] = 0.0
        self.refr[:] = 0

    def step(self, current: np.ndarray) -> np.ndarray:
        """Advance 1 ms with input `current`; returns boolean spike vector."""
        active = self.refr <= 0
        self.v = self.v * self.decay + np.where(active, current, 0.0)
        spikes = active & (self.v >= self.threshold)
        self.v[spikes] = 0.0
        self.refr[spikes] = REFRACTORY_MS
        self.refr[~spikes & ~active] -= 1
        return spikes
