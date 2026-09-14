# Trader Fly 🪰📈 — a Drosophila-inspired trading organism (research prototype)

Market data → sensory encoding → reduced mushroom-body spiking network →
dopamine-gated learning → **LONG / SHORT / NO_TRADE**. Trained on historical
Binance Futures klines, evaluated strictly on later, unseen periods.
**No exchange keys, no orders, no live trading.** See [ARCHITECTURE.md](ARCHITECTURE.md)
for the design and the measured / bio-inspired / engineering ([A]/[B]/[C]) boundary.

## Quickstart

```sh
cd trader
python -m pip install -e ".[dev]"

# 1. offline software check (no network needed)
python run_experiment.py configs/btcusdt_1h_mvp.json --synthetic

# 2. real data: downloads monthly zips (~30 MB total for 1h) into cache/, SHA-256 verified
python -m trader.data.binance_archive BTCUSDT 1h 2020-01 2026-08
python run_experiment.py configs/btcusdt_1h_mvp.json

# 3. watch it think (during or after a run)
python -m trader.ui.dashboard runs/<run_dir>      # http://127.0.0.1:8765

python -m pytest -q                               # tests incl. the no-look-ahead check
```

Every run writes `runs/<timestamp>_<experiment>/` with `manifest.json` (config,
seed, data fingerprint, versions), `metrics.json` (organism + baselines per
split), `decisions_<split>.jsonl` (explainable per-bar records), `report.html`
(plotly), `kc_mbon_weights.npy` and `state.json` (dashboard / creature bridge).

Useful flags: `--max-bars N`, `--epochs N`, `--no-baselines`, `--quiet`,
`--set learning.avoid_scale=0.7` (override any config key).

### Danger objective

Milestone 1 found no directional edge in these senses (see [FINDINGS.md](FINDINGS.md)),
so the organism can also be run on the fly's best-characterized behavior instead:
looming → escape. It learns CALM / ALERT / ESCAPE from whether the horizon brought
a large excursion in either direction, and is judged on exposure avoided.

```sh
python run_experiment.py configs/btcusdt_1h_danger.json --epochs 1
python readout_sweep.py runs/<run_dir>          # sweeps escape_z post hoc
```

## What a decision looks like

```
[2025-03-04 13:00:00+00:00] AROUSAL: 0.71
  MOTION      +1.12
  ATTRACTION  +0.64
  THREAT      -0.30
  CONTEXT     +0.08
  LONG POP    0.41
  SHORT POP   0.27
  AVOID POP   0.32
  DECISION: LONG   (label LONG, reward 0.83)
```

## Layout

```
trader/data         Binance archive downloader + cache, synthetic fixture      [C]
trader/features     causal raw signals, causal z-scores, PN sensory encoding   [C]/[B]
trader/connectome   MB topology generator (measured statistics), loader stub   [A stats, B]
trader/simulation   LIF neurons (Sim.swift constants), one-bar episode         [B]
trader/learning     reward, dopamine drive, KC→MBON depression                 [C]/[B]/[A form]
trader/labels       ATR-normalized MFE/MAE opportunity labels                  [C]
trader/environment  time-ordered replay with delayed outcomes, time splits     [C]
trader/agent        arousal gate, action selection, the organism loop          [B]/[C]
trader/evaluation   metrics, HTML report                                       [C]
trader/baselines    logistic / gradient boosting / MLP on the same data        [C]
trader/explog       decision records, run manifests                            [C]
trader/ui           state bridge, live dashboard                               [C]
```

## Honest expectations

The research question is whether a circuit shaped like the fly's learning center
can pick up structure in a world it never evolved for — and *when it learns to do
nothing*. Judge it on precision, expectancy after cost, drawdown, calibration and
no-trade quality against the baselines, never on accuracy alone. Results on the
synthetic fixture mean nothing about markets; they only prove the software runs.
