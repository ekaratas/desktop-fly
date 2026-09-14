import numpy as np
import pandas as pd

from trader.agent.organism import TradingOrganism
from trader.baselines.classical import baseline_records
from trader.environment.replay import build_dataset, replay
from trader.environment.splits import purge_boundary, time_split, walk_forward_folds
from trader.evaluation.metrics import evaluate


def test_time_split_never_overlaps_and_purge(bars, cfg):
    idx = bars.index
    splits = {"train": ["2020-01-01", "2020-03-31"], "test": ["2020-04-01", "2020-12-31"]}
    m = time_split(idx, splits)
    assert not (m["train"] & m["test"]).any()
    assert idx[m["train"]].max() < idx[m["test"]].min()
    p = purge_boundary(m["train"], 12)
    assert p.sum() == m["train"].sum() - 12
    folds = list(walk_forward_folds(pd.date_range("2019-01-01", "2022-12-31", freq="1D", tz="UTC"), train_years=2))
    assert [y for y, _, _ in folds] == [2021, 2022]
    for _, tr, te in folds:
        assert tr[tr].index.max() < te[te].index.min()


def test_replay_releases_outcomes_only_after_horizon(bars, cfg):
    ds = build_dataset(bars, cfg)
    mask = ds.valid.copy(); mask.iloc[:] = False; mask.iloc[700:900] = True
    seen = set()
    for t, matured in replay(ds, mask):
        for i in matured:
            assert t < 0 or i + ds.horizon <= t
            seen.add(i)
        if t >= 0:
            assert t not in seen
    assert seen == set(range(700, 900))


def test_small_end_to_end_with_baselines(bars, cfg):
    cfg = dict(cfg); cfg["network"] = dict(cfg["network"], n_kc=300, episode_ms=40)
    ds = build_dataset(bars, cfg)
    tr = ds.valid.copy(); tr.iloc[:] = False; tr.iloc[600:1400] = True
    te = ds.valid.copy(); te.iloc[:] = False; te.iloc[1500:1900] = True
    org = TradingOrganism(cfg, ds.n_pn)
    org.calibrate(ds, tr, n_samples=8)
    recs = org.run_split(ds, tr, "train", learn=True, epochs=1)
    assert len(recs) == int((tr & ds.valid).sum()) and org.plasticity.updates > 0
    ev = org.run_split(ds, te, "test", learn=False)
    m = evaluate(ev, ds, cfg)
    assert set(m["label_distribution"]) <= {"LONG", "SHORT", "NO_TRADE"}
    assert 0.0 <= m["trade_frequency"] <= 1.0 and np.isfinite(m["total_log_return"])
    for name, b in baseline_records(ds, tr, te, seed=1).items():
        mb = evaluate(b, ds, cfg)
        assert 0.0 <= mb["trade_frequency"] <= 1.0
