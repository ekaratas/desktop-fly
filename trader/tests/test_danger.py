import json

import numpy as np

from trader.agent.organism import TradingOrganism
from trader.baselines.classical import baseline_records
from trader.environment.replay import build_dataset
from trader.evaluation.metrics import evaluate
from trader.learning.danger import ALERT, CALM, ESCAPE, dan_drive_danger, select_state


def test_danger_dopamine_rule():
    d = dan_drive_danger(3.5, 2.0)
    assert d[0] == 1.5 and d[1] == 0.0                       # storm: punish `safe`
    d = dan_drive_danger(1.0, 2.0, false_alarm_scale=0.5)
    assert d[0] == 0.0 and d[1] == 0.5                       # calm: punish `danger`
    assert not dan_drive_danger(float("nan"), 2.0).any()


def test_state_selection_thresholds():
    assert select_state(np.array([1.0, 1.0]), 0.02, 0.06)[0] == CALM
    assert select_state(np.array([0.97, 1.03]), 0.02, 0.06)[0] == ALERT
    assert select_state(np.array([0.9, 1.1]), 0.02, 0.06)[0] == ESCAPE


def test_toy_danger_conditioning_learns_to_escape():
    """Stimulus A is always followed by a storm, B by calm: after training A → ESCAPE, B → CALM."""
    from trader.connectome.mb_topology import build_mb_topology
    from trader.learning.danger import DANGER_POPULATIONS
    from trader.learning.plasticity import DopaminePlasticity
    from trader.simulation.network import MBNetwork, calibrate_sparsity

    topo = build_mb_topology(20, 400, 4, 0.08, 3, seed=0, populations=DANGER_POPULATIONS)
    net = MBNetwork(topo, episode_ms=60, seed=1)
    A = np.zeros(20); A[:8] = 180.0
    B = np.zeros(20); B[10:18] = 180.0
    calibrate_sparsity(net, np.array([A, B]), target=0.08, iters=10)
    pl = DopaminePlasticity(topo, eta=0.05, w_min=0.05, w_max=1.0, recovery=0.0, synaptic_scaling=False)
    for _ in range(40):
        pl.apply(net.run_episode(A).kc_trace, dan_drive_danger(4.0, 2.0))
        pl.apply(net.run_episode(B).kc_trace, dan_drive_danger(0.8, 2.0))
    a = [select_state(net.run_episode(A).pop_drive, 0.02, 0.06)[0] for _ in range(5)]
    b = [select_state(net.run_episode(B).pop_drive, 0.02, 0.06)[0] for _ in range(5)]
    assert a.count(ESCAPE) >= 4 and b.count(CALM) >= 4


def test_danger_pipeline_end_to_end(bars, cfg):
    cfg = json.loads(json.dumps(cfg))
    cfg["objective"] = "danger"; cfg["danger"] = {"quantile": 0.7}
    cfg["network"].update(n_kc=300, episode_ms=40)
    ds = build_dataset(bars, cfg)
    tr = ds.valid.copy(); tr.iloc[:] = False; tr.iloc[600:1400] = True
    te = ds.valid.copy(); te.iloc[:] = False; te.iloc[1500:1900] = True
    org = TradingOrganism(cfg, ds.n_pn)
    org.calibrate(ds, tr, n_samples=8)
    assert org.danger_threshold is not None and org.danger_threshold > 0
    recs = org.run_split(ds, tr, "train", learn=True, epochs=1)
    assert {r.action for r in recs} <= {"CALM", "ALERT", "ESCAPE"} and {r.label for r in recs} <= {"CALM", "DANGER"}
    ev = org.run_split(ds, te, "test", learn=False)
    m = evaluate(ev, ds, cfg, org.danger_threshold)
    assert 0 <= m["danger_base_rate"] <= 1 and m["bars"] > 0 and "dodge_ratio" in m
    for name, b in baseline_records(ds, tr, te, 1, objective="danger", danger_threshold=org.danger_threshold).items():
        mb = evaluate(b, ds, cfg, org.danger_threshold)
        assert np.isfinite(mb["auc_share_danger"])


def test_gap_normalizer_is_causal_and_standardizes():
    from trader.learning.danger import GapNormalizer, state_from_z
    rng = np.random.default_rng(0)
    g = GapNormalizer(tau_bars=200, min_n=20)
    zs = [g(x) for x in rng.normal(0.3, 0.05, 2000)]
    assert all(z == 0.0 for z in zs[:20])                    # warm-up
    tail = np.array(zs[500:])
    assert abs(tail.mean()) < 0.15 and 0.7 < tail.std() < 1.3
    g2 = GapNormalizer(tau_bars=200, min_n=20)
    xs = rng.normal(0.3, 0.05, 100).tolist()
    a = [g2(x) for x in xs]
    g3 = GapNormalizer(tau_bars=200, min_n=20)
    b = [g3(x) for x in xs[:60] + [9.0] * 40]                 # future changed
    assert a[:60] == b[:60]
    assert state_from_z(2.0) == ESCAPE and state_from_z(1.0) == ALERT and state_from_z(0.0) == CALM


def test_state_bridge_mirror_and_replay(tmp_path):
    import json as _json
    import os
    from trader.ui.replay import main as replay_main
    from trader.ui.state import creature_behavior, write_state_atomic

    assert creature_behavior("ESCAPE", 0.9, False, None) == "escape"
    assert creature_behavior("ALERT", 0.5, False, None) == "alert"
    assert creature_behavior("CALM", 0.1, True, None) == "sleep"
    assert creature_behavior("CALM", 0.5, False, None) == "explore"
    bridge = tmp_path / "market_state.json"
    write_state_atomic(str(bridge), {"behavior": "alert", "arousal": 0.4, "updated": 1.0})
    assert _json.load(open(bridge))["behavior"] == "alert"
    run = tmp_path / "run"; run.mkdir()
    with open(run / "decisions_test.jsonl", "w") as f:
        for act in ("CALM", "ALERT", "ESCAPE"):
            f.write(_json.dumps({"t": "2025-01-01 00:00:00+00:00", "split": "test", "action": act, "label": "CALM",
                                 "arousal": 0.6, "gated": False, "reward": 0.0}) + "\n")
    assert replay_main([str(run), "--rate", "1000", "--path", str(bridge)]) == 0
    last = _json.load(open(bridge))
    assert last["behavior"] == "escape" and last["replay"] is True and last["updated"] > 1e9
