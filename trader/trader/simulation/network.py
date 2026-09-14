"""One market bar → one stimulus episode through the reduced MB ([B]).

PN rates (Hz) are turned into Poisson spikes for `episode_ms` model milliseconds;
KCs integrate their claws with global APL feedback inhibition; MBONs receive the
plastic KC→MBON drive. The KC eligibility trace (exponentially decaying spike
history, tau `eligibility_tau_ms`) is returned so a reward arriving bars later
can still modify the synapses that were active — a modeled stand-in for the
KC cAMP/calcium coincidence trace ([B], biologically motivated).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..connectome.mb_topology import MBTopology, POPULATIONS
from .lif import LIFPopulation


@dataclass
class EpisodeResult:
    kc_trace: np.ndarray        # (n_kc,) eligibility trace at end of episode
    kc_spikes: np.ndarray       # (n_kc,) spike counts
    kc_active_frac: float
    mbon_drive: np.ndarray      # (n_mbon,) summed weighted KC input (decision variable)
    mbon_spikes: np.ndarray     # (n_mbon,) spike counts (logged)
    pop_drive: np.ndarray       # (n_pop,) mean drive per population
    pn_spikes: int


class MBNetwork:
    def __init__(self, topo: MBTopology, episode_ms: int = 100, eligibility_tau_ms: float = 60.0,
                 pn_kc_weight: float = 0.09, apl_gain: float = 0.06, mbon_gain: float | None = None, seed: int = 0):
        self.topo = topo
        self.episode_ms = episode_ms
        self.trace_decay = float(np.exp(-1.0 / eligibility_tau_ms))
        self.pn_kc_weight = pn_kc_weight              # [B] calibrated, see calibrate_sparsity
        self.apl_gain = apl_gain                      # [B] global inhibition strength
        expected_active = max(1.0, topo.n_kc * topo.kc_sparsity_target)
        self.mbon_gain = mbon_gain if mbon_gain is not None else 2.0 / expected_active
        self.kc = LIFPopulation(topo.n_kc)
        self.mbon = LIFPopulation(topo.n_mbon)
        self.rng = np.random.default_rng(seed)

    def run_episode(self, pn_rates_hz: np.ndarray, gain: float = 1.0) -> EpisodeResult:
        topo = self.topo
        self.kc.reset(); self.mbon.reset()
        p_spike = np.clip(pn_rates_hz * gain / 1000.0, 0.0, 1.0)     # per-ms Poisson probability
        trace = np.zeros(topo.n_kc)
        kc_counts = np.zeros(topo.n_kc)
        mbon_counts = np.zeros(topo.n_mbon)
        mbon_drive = np.zeros(topo.n_mbon)
        apl = 0.0
        pn_total = 0
        W = topo.kc_mbon * self.mbon_gain
        pn_spikes = self.rng.random((self.episode_ms, topo.n_pn)) < p_spike     # Poisson PN spikes
        for pn_spk in pn_spikes:
            pn_total += int(pn_spk.sum())
            kc_in = self.pn_kc_weight * (topo.pn_kc @ pn_spk.astype(np.float64)) - apl
            kc_spk = self.kc.step(kc_in)
            n_spk = float(kc_spk.sum())
            apl = apl * 0.8 + self.apl_gain * n_spk                    # APL: fast global feedback [A role, B params]
            kc_counts += kc_spk
            trace = trace * self.trace_decay + kc_spk
            drive = kc_spk.astype(np.float64) @ W                      # (n_mbon,)
            mbon_drive += drive
            mbon_counts += self.mbon.step(drive)
        pop_drive = np.array([mbon_drive[topo.pop_mask(p)].mean() for p in range(topo.n_pop)])
        return EpisodeResult(trace, kc_counts, float((kc_counts > 0).mean()), mbon_drive, mbon_counts, pop_drive, pn_total)


def calibrate_sparsity(net: MBNetwork, pn_rate_samples: np.ndarray, target: float | None = None, iters: int = 12) -> float:
    """Tune pn_kc_weight by bisection so ~target fraction of KCs spike per episode.

    Uses only stimulus samples (no labels) from the training period, so it cannot
    leak outcome information. Returns the chosen weight.
    """
    target = target if target is not None else net.topo.kc_sparsity_target
    lo, hi = 0.005, 0.6
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        net.pn_kc_weight = mid
        frac = np.mean([net.run_episode(r).kc_active_frac for r in pn_rate_samples])
        if frac < target:
            lo = mid
        else:
            hi = mid
    net.pn_kc_weight = 0.5 * (lo + hi)
    return net.pn_kc_weight
