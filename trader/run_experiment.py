#!/usr/bin/env python3
"""Single-command reproducible experiment.

    python run_experiment.py configs/btcusdt_1h_mvp.json            # real archive data (downloads to cache/)
    python run_experiment.py configs/btcusdt_1h_mvp.json --synthetic # offline software check
    python run_experiment.py ... --max-bars 20000 --tag quick        # shorter run

Pipeline: load bars → causal features → PN rates → time splits → calibrate KC sparsity
on train stimuli → train organism (delayed dopamine) → evaluate on train/validation/test
→ same splits for classical baselines → metrics.json, decisions.jsonl, report.html,
state.json (dashboard / desktop-creature bridge).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trader.agent.organism import TradingOrganism                      # noqa: E402
from trader.baselines.classical import baseline_records                # noqa: E402
from trader.data.loader import data_fingerprint, load_market_data      # noqa: E402
from trader.environment.replay import build_dataset                    # noqa: E402
from trader.environment.splits import proportional_split, purge_boundary, time_split       # noqa: E402
from trader.evaluation.metrics import decisions_frame, evaluate        # noqa: E402
from trader.evaluation.report import write_report                      # noqa: E402
from trader.explog.experiment import RunDir                            # noqa: E402
from trader.ui.state import StateWriter                                # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--synthetic", action="store_true", help="use the synthetic fixture instead of Binance data")
    p.add_argument("--synthetic-bars", type=int, default=30000)
    p.add_argument("--max-bars", type=int, default=None, help="truncate the dataset to the first N bars")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--tag", default="")
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--dump-kc", action="store_true", help="save KC spike codes per split (kc_<split>.npz) for kc_probe.py")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, dotted path, e.g. --set learning.avoid_scale=0.7")
    a = p.parse_args()

    with open(a.config) as f:
        cfg = json.load(f)
    for kv in a.set:
        key, val = kv.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = json.loads(val)
    if a.epochs is not None:
        cfg["learning"]["epochs"] = a.epochs
    np.random.seed(cfg["seed"])

    bars = load_market_data(cfg, synthetic=a.synthetic, synthetic_bars=a.synthetic_bars)
    if a.max_bars:
        bars = bars.iloc[: a.max_bars]
    run = RunDir(cfg, tag=(a.tag or ("synthetic" if a.synthetic else "")))
    print(f"[run] {run.path}\n[data] {len(bars)} bars {bars.index[0]} → {bars.index[-1]} synthetic={a.synthetic}")

    ds = build_dataset(bars, cfg)
    masks = time_split(ds.bars.index, cfg["splits"])
    if any(int((m & ds.valid).sum()) == 0 for m in masks.values()):
        print("[split] configured date windows do not cover this data → proportional 60/20/20 time split")
        masks = proportional_split(ds.bars.index)
    hz = ds.horizon
    train_mask = purge_boundary(masks["train"], hz)
    for k, m in masks.items():
        print(f"[split] {k:<10} {int((m & ds.valid).sum()):>7} valid bars")
    if int((train_mask & ds.valid).sum()) == 0:
        print("no training bars in the configured train window for this data; adjust splits", file=sys.stderr)
        return 2

    org = TradingOrganism(cfg, ds.n_pn)
    w = org.calibrate(ds, train_mask)
    print(f"[net] n_pn={ds.n_pn} n_kc={org.topo.n_kc} n_mbon={org.topo.n_mbon} pn_kc_weight={w:.4f} (KC sparsity target {org.topo.kc_sparsity_target})")

    state = StateWriter(run.file("state.json"), run.file("decisions.jsonl"), bars=ds.bars)
    every = 500

    def on_decision(rec, organism):
        state.update(rec, organism, quiet=a.quiet)
        if not a.quiet and state.count % every == 0:
            print(f"  [{rec.split}] {rec.t} {rec.action:<8} arousal {rec.arousal:.2f} shares "
                  + " ".join(f"{k}={v:.2f}" for k, v in rec.pop_share.items()))

    def dump_kc(split):
        if not a.dump_kc or not org.kc_dump:
            return
        ts = np.array([t for t, _ in org.kc_dump]); codes = np.stack([c for _, c in org.kc_dump])
        np.savez_compressed(run.file(f"kc_{split}.npz"), t=ts, codes=codes,
                            label=ds.outcomes["label"].to_numpy()[ts], fwd_return_atr=ds.outcomes["fwd_return_atr"].to_numpy()[ts],
                            mfe_long=ds.outcomes["mfe_long"].to_numpy()[ts], mae_long=ds.outcomes["mae_long"].to_numpy()[ts],
                            trend=ds.z["trend_slope"].to_numpy()[ts])
        org.kc_dump.clear()

    results = {}
    if a.dump_kc:
        org.kc_dump = []
    train_recs = org.run_split(ds, train_mask, "train", learn=True, epochs=cfg["learning"]["epochs"], on_decision=on_decision)
    if a.dump_kc and cfg["learning"]["epochs"] > 1:      # keep only the last epoch's codes
        n = len(train_recs); org.kc_dump = org.kc_dump[-n:]
    dump_kc("train")
    thr = org.danger_threshold
    if org.objective == "danger":
        print(f"[danger] threshold {thr:.2f} ATR (train quantile {org.danger_quantile})")
    results["train"] = {"organism": evaluate(train_recs, ds, cfg, thr)}
    frozen = {}
    for split in ("validation", "test"):
        if int((masks[split] & ds.valid).sum()) == 0:
            continue
        recs = org.run_split(ds, masks[split], split, learn=False, on_decision=on_decision)
        dump_kc(split)
        results[split] = {"organism": evaluate(recs, ds, cfg, thr)}
        frozen[split] = recs
        for r in recs:
            run.append_jsonl(f"decisions_{split}.jsonl", r.to_json())

    if not a.no_baselines:
        for split in [s for s in ("validation", "test") if s in results]:
            for name, recs in baseline_records(ds, train_mask, masks[split], cfg["seed"], objective=org.objective,
                                               danger_threshold=thr).items():
                results[split][name] = evaluate(recs, ds, cfg, thr)

    W = org.topo.kc_mbon
    weights_by_pop = {pop: W[:, org.topo.mbon_pop == i].ravel() for i, pop in enumerate(org.populations)}
    np.save(run.file("kc_mbon_weights.npy"), W)
    run.write_json("metrics.json", results)
    test_split = "test" if "test" in frozen else ("validation" if "validation" in frozen else None)
    if org.objective == "danger":
        from trader.evaluation.danger_metrics import danger_frame
        from trader.evaluation.report import write_danger_report
        dec_df = danger_frame(frozen[test_split] if test_split else train_recs, ds, thr)
        write_danger_report(run.file("report.html"), results, dec_df, ds.bars, weights_by_pop, {**run.manifest,
                            "data_range": f"{bars.index[0]} → {bars.index[-1]}", "data_fingerprint": data_fingerprint(bars)})
    else:
        dec_df = decisions_frame(frozen[test_split], ds) if test_split else decisions_frame(train_recs, ds)
        write_report(run.file("report.html"), results, dec_df, ds.bars, weights_by_pop, {**run.manifest,
                     "data_range": f"{bars.index[0]} → {bars.index[-1]}", "data_fingerprint": data_fingerprint(bars)})
    run.finish({"data_range": f"{bars.index[0]} → {bars.index[-1]}", "data_fingerprint": data_fingerprint(bars),
                "n_bars": len(bars), "pn_kc_weight": w, "plasticity_updates": org.plasticity.updates,
                "objective": org.objective, "danger_threshold": thr,
                "summary": {s: {m: {k: v for k, v in mm.items() if k in ("accuracy", "trade_frequency", "expectancy_atr",
                                                                         "profit_factor", "sharpe", "total_log_return", "max_drawdown")}
                                for m, mm in models.items()} for s, models in results.items()}})

    print("\n=== summary ===")
    for split, models in results.items():
        for name, m in models.items():
            if org.objective == "danger":
                print(f"{split:<10} {name:<10} AUC {m.get('auc_share_danger', float('nan')):.3f} "
                      f"escape {m['frac_ESCAPE']:.2f} alert {m['frac_ALERT']:.2f} prec {m.get('escape_precision', float('nan')):.2f} "
                      f"recall {m.get('escape_recall', float('nan')):.2f} FA {m.get('escape_false_alarm_rate', float('nan')):.2f} "
                      f"warned-before {m.get('events_warned_before', float('nan')):.2f} lead {m.get('mean_lead_bars', float('nan')):.1f} "
                      f"dodge {m.get('dodge_ratio', float('nan')):.2f} advCalm {m.get('adverse_when_calm', float('nan')):.2f} advEsc {m.get('adverse_when_escape', float('nan')):.2f}")
                continue
            print(f"{split:<10} {name:<10} freq {m['trade_frequency']:.2f} acc {m['accuracy']:.2f} "
                  f"precL {m.get('precision_LONG', float('nan')):.2f} precS {m.get('precision_SHORT', float('nan')):.2f} "
                  f"exp {m.get('expectancy_atr', float('nan')):+.3f} PF {m.get('profit_factor', float('nan')):.2f} "
                  f"sharpe {m.get('sharpe', float('nan')):+.2f} ret {m['total_log_return']:+.3f} maxDD {m['max_drawdown']:+.3f}")
    print(f"report: {run.file('report.html')}\ndashboard: python -m trader.ui.dashboard {run.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
