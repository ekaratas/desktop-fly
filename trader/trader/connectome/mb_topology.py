"""Reduced mushroom-body (MB) topology generator.

Category tags:
  [A] numbers taken from the measured Drosophila MB (hemibrain / FlyWire literature):
      ~2,000 Kenyon cells (KC) per hemisphere, 5–7 projection-neuron (PN) claws per KC,
      ~5 % of KCs respond to a given odor, a single APL neuron feeds back inhibition
      onto all KCs, KC→MBON synapses start strong and are *depressed* by dopamine.
  [B] our mapping: 46 market PNs instead of ~150 olfactory glomeruli, three MBON
      populations (approach-LONG, approach-SHORT, avoid) instead of ~34 MBON types
      in 15 compartments, one DAN drive per population.
  [C] the sizes are scaled down for speed (n_kc default 1200).

No edge here is read from a connectome file. `trader/connectome/loader.py`
documents how a real KC/MBON/DAN extract from FlyWire would replace this generator.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TOPOLOGY_VERSION = "mb1"
POPULATIONS = ("long", "short", "avoid")


@dataclass
class MBTopology:
    n_pn: int
    n_kc: int
    n_mbon: int
    pn_kc: np.ndarray          # (n_kc, n_pn) binary claw matrix          [A stats, B wiring]
    kc_mbon: np.ndarray        # (n_kc, n_mbon) plastic weights, start at w_max  [A rule]
    mbon_pop: np.ndarray       # (n_mbon,) population index 0=long 1=short 2=avoid  [B]
    claws_per_kc: int
    kc_sparsity_target: float
    meta: dict = field(default_factory=dict)

    def pop_mask(self, pop: int) -> np.ndarray:
        return self.mbon_pop == pop


def build_mb_topology(n_pn: int, n_kc: int = 1200, claws_per_kc: int = 6, kc_sparsity: float = 0.05,
                      n_mbon_per_pop: int = 4, w_max: float = 1.0, seed: int = 0) -> MBTopology:
    rng = np.random.default_rng(seed)
    pn_kc = np.zeros((n_kc, n_pn), dtype=np.float64)
    for k in range(n_kc):
        pn_kc[k, rng.choice(n_pn, size=claws_per_kc, replace=False)] = 1.0   # random claws [A stats]
    n_mbon = n_mbon_per_pop * len(POPULATIONS)
    mbon_pop = np.repeat(np.arange(len(POPULATIONS)), n_mbon_per_pop)
    kc_mbon = np.full((n_kc, n_mbon), w_max, dtype=np.float64)              # naive fly: all strong [A]
    return MBTopology(n_pn, n_kc, n_mbon, pn_kc, kc_mbon, mbon_pop, claws_per_kc, kc_sparsity,
                      meta={"version": TOPOLOGY_VERSION, "seed": seed, "category": "B (A-derived statistics)"})
