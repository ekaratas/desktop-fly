# Findings log

Running record of what the experiments established. Negative results are kept:
they bound what the organism can be expected to learn and where the next effort
should go. All runs: BTCUSDT, USDⓈ-M futures, Binance public archive, cost 0.15 ATR
per round trip, splits train 2020–2023 / validation 2024 / test 2025-01→2026-08.

## 2026-09-13 — OHLCV-only stimuli carry no exploitable direction (1h and 4h)

**Setup.** 23 causal OHLCV-derived signals → 46 ON/OFF projection neurons → 1,200 KCs
(6 claws, ~5 % active) → long/short/avoid MBON populations; `observed_all` dopamine
rule; horizon 12 bars (1h) / 6 bars (4h).

**Organism.** After fixing three readout/plasticity issues (avoid_scale offset →
synaptic scaling → MBON gain control; effective memory of only a few months at
eta 0.03 → eta 0.001, recovery 0) the readout became well behaved: trade frequency
moves smoothly with `margin`/`no_trade_bias`. But at every frequency the win rate
stayed at 0.47–0.50 and expectancy after cost negative.

**Linear probe on the KC code** (`kc_probe.py`, logistic regression, 34.6k train bars):
direction AUC 0.517 (validation) / 0.503 (test) at 1h, 0.496 / 0.499 at 4h. Chance.
So the plasticity rule is not the bottleneck: no linear readout of this KC code
predicts the sign of the forward return.

**Classical baselines on the raw features** agree. At 1h all three lose on test. At 4h
an MLP looked positive on one seed (validation +0.25, test +0.10 ATR); across five
seeds test expectancy was positive in 1/5 while validation was positive in 4/5 —
validation-year overfit, not signal.

**Danger is weakly predictable.** A probe for "large adverse excursion in both
directions" (top 30 %) reaches AUC 0.577 (1h) and 0.57 (4h), consistent across
validation and test. The KC code carries volatility-clustering information. Used as
a filter on a naive trend rule it lowered the mean adverse excursion (1.88 → 1.62 ATR)
but did not turn the rule's expectancy positive: a filter cannot rescue a base rule
with no edge.

**Conclusion.** With OHLCV-derived stimuli at 1h/4h horizons and realistic cost, there
is no directional edge for the organism (or anything else) to learn. Its low trade
frequency is correct behavior but reflects absence of signal, not learned caution.
The research question therefore moves to (a) richer senses — funding, premium
index, open interest, positioning ratios — and (b) whether the weak but real
danger signal can be turned into a learned NO_TRADE behavior that pays.

**Next.** `data.extras = true` attaches funding / premium / metrics tables (as-of
merged on bar close). Re-run the probe before re-tuning anything: if direction AUC
stays at 0.5 with the new senses, direction is dropped as an objective.

## 2026-09-14 — funding / premium / OI / positioning do not add direction at 4h

`data.extras = true` (66 PNs). Direction probe on the KC code: AUC 0.508 (validation)
/ 0.497 (test). Classical baselines got *worse* with the extra channels (all three
negative on test), consistent with added noise rather than added information at a
24 h horizon. Danger probe unchanged at AUC 0.574 / 0.563.

Per the pre-registered criterion, **direction is dropped as an objective at 1h and 4h**.
Remaining avenues: (1) daily timeframe, where funding/positioning effects are
reported to live (`configs/btcusdt_1d_mvp.json`, 5-day horizon); (2) turning the
consistent but weak danger signal into a learned NO_TRADE behavior, judged by
abstention on dangerous bars rather than by PnL.
