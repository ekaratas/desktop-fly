import numpy as np

from trader.connectome.mb_topology import build_mb_topology
from trader.learning.dopamine import dan_drive
from trader.learning.plasticity import DopaminePlasticity
from trader.labels.excursion import LONG, NO_TRADE, SHORT
from trader.simulation.lif import LIFPopulation
from trader.simulation.network import MBNetwork, calibrate_sparsity


def test_lif_spikes_and_refractory():
    p = LIFPopulation(1)
    spikes = [p.step(np.array([0.3]))[0] for _ in range(50)]
    assert any(spikes)
    idx = [i for i, s in enumerate(spikes) if s]
    assert all(b - a >= 3 for a, b in zip(idx, idx[1:]))           # refractory 2 ms → gap >= 3 steps
    q = LIFPopulation(1)
    assert not any(q.step(np.array([0.02]))[0] for _ in range(200))  # sub-threshold: 20 ms tau × 0.02 < 1


def test_topology_statistics():
    topo = build_mb_topology(46, 500, 6, 0.05, 4, seed=1)
    assert topo.pn_kc.shape == (500, 46) and (topo.pn_kc.sum(axis=1) == 6).all()
    assert topo.kc_mbon.shape == (500, 12) and (topo.kc_mbon == 1.0).all()
    assert list(np.bincount(topo.mbon_pop)) == [4, 4, 4]


def test_sparsity_calibration_and_determinism():
    topo = build_mb_topology(46, 600, 6, 0.05, 4, seed=2)
    net = MBNetwork(topo, episode_ms=60, seed=5)
    rng = np.random.default_rng(1)
    samples = np.abs(rng.standard_normal((12, 46))) * 90
    calibrate_sparsity(net, samples, iters=10)
    frac = np.mean([net.run_episode(s).kc_active_frac for s in samples])
    assert 0.02 < frac < 0.10
    a = MBNetwork(topo, episode_ms=60, seed=9).run_episode(samples[0])
    b = MBNetwork(topo, episode_ms=60, seed=9).run_episode(samples[0])
    np.testing.assert_array_equal(a.kc_spikes, b.kc_spikes)


def test_dopamine_rule_symmetry():
    d_up = dan_drive(NO_TRADE, 0.0, np.array([1.0, -1.3, 0.0]))
    d_dn = dan_drive(NO_TRADE, 0.0, np.array([-1.3, 1.0, 0.0]))
    assert d_up[0] == 0 and d_up[1] == 1.3 and np.isclose(d_up[2], 0.6)
    assert d_dn[1] == 0 and d_dn[0] == 1.3 and np.isclose(d_dn[2], 0.6)
    both_lose = dan_drive(LONG, -0.2, np.array([-0.2, -0.1, 0.0]))
    assert both_lose[2] == 0.0 and both_lose[0] > 0 and both_lose[1] > 0


def test_plasticity_only_depresses_active_kcs_in_target_population():
    topo = build_mb_topology(10, 50, 3, 0.1, 2, seed=0)
    pl = DopaminePlasticity(topo, eta=0.1, w_min=0.05, w_max=1.0, recovery=0.0)
    trace = np.zeros(50); trace[[3, 7]] = 1.0
    pl.apply(trace, np.array([0.0, 2.0, 0.0]))                       # punish SHORT population
    W = topo.kc_mbon
    assert np.allclose(W[[3, 7]][:, topo.mbon_pop == 1], 0.8)
    assert np.allclose(W[[3, 7]][:, topo.mbon_pop != 1], 1.0)
    mask = np.ones(50, bool); mask[[3, 7]] = False
    assert np.allclose(W[mask], 1.0)
    for _ in range(100):
        pl.apply(trace, np.array([0.0, 2.0, 0.0]))
    assert W.min() >= 0.05                                           # clipped at w_min


def test_toy_conditioning_learns_stimulus_action_map():
    """Two PN patterns: A is rewarded for LONG, B for SHORT. After delayed dopamine, choices follow."""
    from trader.agent.action import select_action
    topo = build_mb_topology(20, 400, 4, 0.08, 3, seed=0)
    net = MBNetwork(topo, episode_ms=60, seed=1)
    A = np.zeros(20); A[:8] = 180.0
    B = np.zeros(20); B[10:18] = 180.0
    calibrate_sparsity(net, np.array([A, B]), target=0.08, iters=10)
    pl = DopaminePlasticity(topo, eta=0.05, w_min=0.05, w_max=1.0, recovery=0.0)
    for _ in range(40):
        for stim, cf in ((A, np.array([1.0, -1.3, 0.0])), (B, np.array([-1.3, 1.0, 0.0]))):
            res = net.run_episode(stim)
            act, _ = select_action(res.pop_drive, 0.0, 0.03)
            pl.apply(res.kc_trace, dan_drive(act, 0.0, cf))
    a_act = [select_action(net.run_episode(A).pop_drive, 0.0, 0.03)[0] for _ in range(5)]
    b_act = [select_action(net.run_episode(B).pop_drive, 0.0, 0.03)[0] for _ in range(5)]
    assert a_act.count(LONG) >= 4 and b_act.count(SHORT) >= 4
