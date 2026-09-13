"""Dopamine-gated depression of KC→MBON synapses ([A] rule form, [B] parameters).

dw[kc, mbon in pop p] = -eta * trace[kc] * dan[p];  w clipped to [w_min, w_max];
a slow recovery toward w_max stands in for forgetting/homeostasis ([B]).
"""
from __future__ import annotations

import numpy as np

from ..connectome.mb_topology import MBTopology


class DopaminePlasticity:
    def __init__(self, topo: MBTopology, eta: float = 0.02, w_min: float = 0.05, w_max: float = 1.0, recovery: float = 5e-4):
        self.topo, self.eta, self.w_min, self.w_max, self.recovery = topo, eta, w_min, w_max, recovery
        self.updates = 0

    def apply(self, kc_trace: np.ndarray, dan: np.ndarray) -> float:
        """Apply one delayed outcome. Returns total |dw| for logging."""
        W = self.topo.kc_mbon
        if self.recovery > 0:
            W += self.recovery * (self.w_max - W)
        if not np.any(dan > 0) or not np.any(kc_trace > 0):
            return 0.0
        active = np.nonzero(kc_trace > 1e-6)[0]
        tr = kc_trace[active] / max(1.0, kc_trace[active].max())     # normalize trace to [0,1]
        dan_per_mbon = dan[self.topo.mbon_pop]                        # (n_mbon,)
        dw = -self.eta * np.outer(tr, dan_per_mbon)
        W[active] = np.clip(W[active] + dw, self.w_min, self.w_max)
        self.updates += 1
        return float(np.abs(dw).sum())
