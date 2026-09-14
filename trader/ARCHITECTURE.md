# Trader Fly — technical architecture (milestone 1)

A research prototype that treats Binance Futures market data as the sensory world
of a Drosophila-inspired organism. The organism perceives market stimuli, runs them
through a reduced mushroom-body (MB) spiking circuit, receives dopamine-like
reward/punishment from *future* price action and learns to emit
**LONG / SHORT / NO_TRADE**. It never places orders.

```
MARKET DATA ──► SENSORY ENCODING ──► MB SPIKING NETWORK ──► ACTION SELECTION ──► LONG/SHORT/NO_TRADE
 (klines)        (PN rates)           (KC sparse code,        (MBON populations,
                                       APL, MBON)              arousal gate)
                                          ▲                          │
                                          │ dopamine-gated           │ horizon N bars later
                                          │ KC→MBON depression       ▼
                                      DAN DRIVE ◄──────────── FUTURE OUTCOME (MFE/MAE, return)
```

## 1. The three categories

Every module, parameter and claim is tagged in code comments and docstrings:

| tag | meaning | examples in this prototype |
|---|---|---|
| **[A] measured** | numbers or wiring read from the real Drosophila connectome/literature | ~2,000 KCs/hemisphere, 5–7 PN claws per KC, ~5 % KC sparsity, one APL, KC→MBON synapses start strong and are depressed by dopamine, PPL1 = punishment / PAM = reward compartments, LIF operating point shared with `Sim.swift` |
| **[B] bio-inspired mapping (ours)** | biologically motivated but designed by us | feature→sense channel table, ON/OFF opponent PN coding, three MBON populations (long/short/avoid), one DAN per population, eligibility trace, arousal gate, sizes scaled down |
| **[C] trading engineering** | pure engineering, no biological claim | features, causal normalization, labels, reward with cost, splits, metrics, baselines, dashboard, the "observed_all" dopamine rule justification |

**Milestone 1 contains no [A] synapses.** The MB topology is *generated* from [A]
statistics (`trader/connectome/mb_topology.py`, category recorded in every run
manifest as `"B (statistics only; no measured edges)"`). The path to real wiring
is documented in `trader/connectome/loader.py`: extend the root `etl.py`
`CORE_TYPES` with KC / MBON / PAM / PPL1 / APL types from FlyWire v783 and load
the extracted compartment graph into the same `MBTopology` structure.

Nothing here is a claim of biologically calibrated learning. The organism is a
model whose *form* follows the fly; its parameters are fitted to make software
run, not to reproduce measured fly physiology.

## 2. Data layer `[C]`

- `trader/data/binance_archive.py` — monthly kline zips from
  `https://data.binance.vision/data/futures/um/monthly/klines/<SYMBOL>/<TF>/`,
  SHA-256 verified against the published `.CHECKSUM`, cached under `cache/`.
  Handles header/no-header files and ms/µs timestamps. Returns a UTC-indexed frame
  with open/high/low/close/volume/quote_volume/count/taker_buy_volume.
- `trader/data/synthetic.py` — regime-switching random walk with volatility
  clustering; **only** for software checks (no market information).
- Later additions (same interface): mark price, funding, open interest, order-flow.

## 3. Features and sensory encoding

**Raw signals `[C]`** (`trader/features/raw.py`, version `f1`): returns at 1/4/12
bars, ATR-normalized velocity/acceleration/trend slope, breakout magnitude,
volume z-score and acceleration, taker buy/sell imbalance, VWAP distance,
continuation, volatility expansion, range shock, reversal, wick rejection,
drawdown from 48-bar high / run-up from low, ATR regime, compression, weekly
direction, activity, body ratio. All computed with bars ≤ t only.

**Causal normalization `[C]`** (`normalize.py`): rolling z-score whose mean/std
are shifted by one bar (the current row never enters its own statistics), clipped ±4.

**Sensory channels `[B]`** (`raw.CHANNELS`, editable table):

| fly sense (stand-in) | channel | signals |
|---|---|---|
| visual motion | `motion` | ret1, ret4, ret12, velocity, acceleration, trend_slope, breakout |
| odor / attraction | `attraction` | volume_z, volume_accel, taker_imbalance, vwap_dist, continuation |
| looming / threat | `threat` | vol_expansion, range_shock, reversal, rejection, drawdown_from_high, runup_from_low |
| environment | `context` | atr_regime, compression, htf_direction, activity, body_ratio |

**PN encoding `[B]`** (`sensory.py`, version `s1`): each z-scored signal drives an
ON and an OFF projection neuron, rate = 200 Hz · tanh(|z|/2) on the matching sign.
23 signals → 46 PNs. Poisson spikes are drawn from these rates during an episode.

## 4. Mushroom-body network `[A stats, B wiring, C sizes]`

`trader/connectome/mb_topology.py`, `trader/simulation/`:

- **KC layer**: 1,200 Kenyon cells, each with 6 random PN claws (binary). The
  PN→KC weight is calibrated by bisection on *training-period stimuli only* so that
  ≈5 % of KCs spike per episode (`calibrate_sparsity`).
- **APL**: one global inhibitory feedback proportional to the KC spike count of the
  previous millisecond (fast decay) — role from biology, parameters ours.
- **MBONs**: 3 populations × 4 cells (long, short, avoid). Plastic KC→MBON weights
  start at `w_max` (a naive fly approaches everything equally).
- **LIF neuron** (`simulation/lif.py`): 1 ms step, τ = 20 ms, threshold 1.0,
  refractory 2 ms — the constants DesktopFly's FlyWire simulation uses, so both
  simulations share a neuron model. Float64.
- **Episode** = one market bar presented for `episode_ms` (100 ms model time). Output:
  KC spike counts, the KC **eligibility trace** (exponential decay, τ = 60 ms), and
  MBON drive per population (the decision variable; MBON spikes are logged).

## 5. Arousal `[B]`

`trader/agent/arousal.py`: EMA (half-life 6 bars) of the mean |z| of
volatility expansion, volume surprise, acceleration, range shock and breakout,
squashed to 0..1. It scales stimulus gain (0.6–1.4) and, below
`arousal.gate_threshold`, skips the episode: the organism rests and emits
NO_TRADE. It never selects a direction.

## 6. Learning `[A rule form, B mapping, C signal]`

- **Reward `[C]`** (`learning/reward.py`): `r = clip(dir · fwd_return_ATR − cost_ATR, ±r_max)`;
  NO_TRADE earns 0. `cost_atr = 0.15` ATR per round trip stands for fees + slippage
  and makes tiny edges net-negative, so abstaining can win.
- **Dopamine drive `[B]`** (`learning/dopamine.py`): each population has a
  punishment-coding DAN. Default rule `observed_all`: because the market reveals
  what every action would have earned, DAN_long = (−r_long)⁺, DAN_short = (−r_short)⁺,
  DAN_avoid = avoid_scale · max(r_long, r_short)⁺ (missed edge). Symmetric in
  LONG/SHORT. The alternative `chosen_only` (only the taken action is evaluated,
  reward depresses competitors) is implemented but showed rich-get-richer collapse
  in tests.
- **Plasticity `[A form]`** (`learning/plasticity.py`):
  `Δw[kc, mbon∈pop] = −η · trace[kc] · DAN[pop]`, clipped to `[w_min, w_max]`, plus a
  slow recovery toward `w_max` (forgetting/homeostasis, `[B]`).
- **Synaptic scaling `[B]`**: after every update each MBON population's weights are
  rescaled to a fixed mean (`scaling_target`). Depression then only redistributes
  weight across KCs, i.e. learns *which stimulus patterns* favor which action.
  Without it the population means drift with the market's base rates and the
  long/short/avoid balance becomes a global offset controlled by `avoid_scale`
  (first real-data run: 50 % trade frequency at 0.6, 0 % at 0.45). With it the
  trade frequency is set by the readout knobs only, which `readout_sweep.py`
  can tune post hoc from recorded decisions.
- **Delay**: the outcome of bar t is released at t + horizon (12 bars). Until then
  the decision's KC trace waits in a queue; weights used for decisions at t+1…t+11
  cannot contain information about t's outcome. Enforced by `environment/replay.py`.

`agent.margin` and `agent.no_trade_bias` set how eager the organism is to trade;
with the `observed_all` rule learning does not depend on the chosen action, so
`python readout_sweep.py runs/<run>` evaluates them on the recorded validation
decisions without retraining. Confirm once on test; never tune on test.

## 7. Action selection `[B readout, C bias]`

`trader/agent/organism.py` first applies **MBON gain control `[B]`**: each
population's drive is divided by a slow causal EMA (τ = 500 bars) of its own past
drive. Long-run shares are therefore 1/3 each whatever the global weight levels
(synaptic scaling alone left an activity-weighted offset, so `avoid_scale` still
moved trade frequency from 1 % to 45 % on the synthetic fixture); decisions come
from stimulus-specific deviations. `trader/agent/action.py` then normalizes the
gain-controlled drive to shares; add
`no_trade_bias` to `avoid`; argmax; if the winning direction beats the runner-up
by less than `margin`, abstain. With all weights equal the naive organism
abstains — it must *learn* to trade.

## 8. Labels `[C]`

`trader/labels/excursion.py` (version `l1`): for bars t+1…t+N, MFE/MAE of a long
in ATR(t) units (SHORT mirrored). LONG_OPPORTUNITY if MFE ≥ 1 ATR and MFE ≥ 1.8·MAE;
SHORT symmetric; else NO_TRADE. Labels are only used for evaluation and as
baseline targets — the organism learns from the reward signal, not from labels.

## 9. Evaluation `[C]`

- Time splits from the config (train 2020–2023, validation 2024, test 2025→); the
  last `horizon` bars of train are purged so no label overlaps validation. Yearly
  walk-forward folds are available (`splits.walk_forward_folds`). If the configured
  windows don't cover the data (synthetic runs) a 60/20/20 proportional time split
  is used and printed.
- Metrics (`evaluation/metrics.py`): per-class precision/recall, accuracy, trade
  frequency, expectancy (ATR), win rate, profit factor, average favorable/adverse
  excursion, sequential non-overlapping equity curve (log return, cost in ATR·price
  units), max drawdown, exposure, Sharpe/Sortino (annualized from trade count),
  confidence calibration buckets, ATR-regime breakdown and a **no-trade quality**
  block: how much edge the abstained bars carried and how often abstaining was right.
- Baselines (`baselines/classical.py`): logistic regression, HistGradientBoosting,
  MLP on identical features/labels/splits (a probability floor of 0.45 lets them abstain).

## 10. Explainability and reproducibility `[C]`

Every decision is a `DecisionRecord` (time, arousal, channel activations, gated
flag, KC active fraction, MBON drive/shares/spikes, action, label, matured reward,
DAN drive). Test/validation decisions go to `decisions_<split>.jsonl`; a rolling
`state.json` feeds the dashboard and, later, the desktop creature.
`manifest.json` records config, seed, git commit, data range and fingerprint,
feature/sensory/label/topology versions, library versions, calibrated PN→KC weight,
number of plasticity updates and a metrics summary.

## 11. Dashboard and creature bridge `[C]`

`python -m trader.ui.dashboard runs/<run>` serves a single page (stdlib HTTP,
no framework) that polls `state.json`: recent candles with LONG/SHORT marks and
arousal bars, matured PnL curve, decision, creature behavior
(sleep/explore/alert/hunt_long/hunt_short/reward/aversive), sensory channels, MBON
shares and mean weights. The same file is the contract for a DesktopFly hook:
the Swift/Electron app only needs to read `behavior`, `arousal`, `decision`.

## 12. Danger objective (`objective: "danger"`)

Added after milestone 1 showed no directional edge (see `FINDINGS.md`). The same
sensory → KC machinery, two MBON populations `safe` / `danger` and one outcome:
the largest excursion in *either* direction over the horizon, in ATR units.

- **Outcome & DAN `[A role, B mapping]`**: a storm (risk > threshold, threshold = the
  training-period quantile `danger.quantile`, computed at calibration from train bars
  only) punishes `safe`; a calm horizon punishes `danger` (false alarm), scaled by
  `false_alarm_scale`. Modeled on PPL1 punishment and the looming → escape circuit.
- **Readout `[B]`**: gain-controlled shares as before; the gap share(danger) − share(safe)
  is standardized by a causal running mean/std (`GapNormalizer`, adaptive threshold /
  habituation) and thresholded at `alert_z` → ALERT, `escape_z` → ESCAPE, else CALM.
  Both knobs are readout-only and sweepable post hoc (`readout_sweep.py`).
- **Metrics `[C]`** (`evaluation/danger_metrics.py`): AUC of the danger share, ESCAPE
  precision / recall / false-alarm rate, lead time before danger events, and
  *dodge ratio* = share of a naive exposed rule's adverse excursion that falls inside
  ESCAPE bars ÷ share of bars escaped (> 1: escapes concentrate on storms).
- **Creature `[C]`**: CALM → sleep/explore by arousal, ALERT → alert, ESCAPE → escape.
  Trading meaning: ESCAPE = flat; the objective is exposure avoided, not PnL.

## 13. Assumptions and known limitations

- One symbol, one timeframe, klines only (no OI/funding/mark yet).
- Fixed horizon, fixed cost, no position sizing, entry at bar close.
- The MB here is ~1/2 of one hemisphere's KC count and 3 MBON populations instead of
  ~34 types; APL and DAN dynamics are caricatures.
- Poisson PN spiking makes decisions stochastic; seeds fix them per run.
- Learning from *observed counterfactual outcomes* is a market-specific liberty a
  foraging fly does not have — tagged [C] and switchable.
- Expectation: on real data the organism will most likely not beat gradient boosting.
  The research value is in *what* it learns (which stimuli it attends to, when it
  abstains), not in PnL.
