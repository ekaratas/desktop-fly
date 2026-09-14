"""The trading organism: sensory → MB network → action, with delayed dopamine learning.

Everything here is glue ([C]) around the tagged components. `run_split` replays a
time mask; when `learn=True` matured outcomes update KC→MBON weights through the
DAN drive. Decisions at bar t never see weights updated by outcomes of bars
whose horizon has not elapsed yet (enforced by the replay's release schedule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..agent.action import select_action
from ..agent.arousal import sensory_gain
from ..connectome.mb_topology import POPULATIONS, build_mb_topology
from ..environment.replay import Dataset, replay
from ..explog.explain import DecisionRecord
from ..features.sensory import channel_summary
from ..labels.excursion import ACTION_NAMES, NO_TRADE
from ..learning.danger import (DANGER_POPULATIONS, ESCAPE, STATE_NAMES, GapNormalizer, dan_drive_danger,
                               risk_outcome, select_state, state_from_z)
from ..learning.dopamine import dan_drive
from ..learning.plasticity import DopaminePlasticity
from ..learning.reward import counterfactual_rewards, realized_reward
from ..simulation.network import MBNetwork, calibrate_sparsity


class TradingOrganism:
    def __init__(self, cfg: dict, n_pn: int):
        self.cfg = cfg
        self.objective = cfg.get("objective", "direction")          # "direction" | "danger"
        self.populations = DANGER_POPULATIONS if self.objective == "danger" else POPULATIONS
        D = cfg.get("danger", {})
        self.danger_quantile = D.get("quantile", 0.7)                 # train-period quantile → threshold
        self.danger_threshold: float | None = D.get("threshold_atr")  # or a fixed ATR threshold
        self.false_alarm_scale = D.get("false_alarm_scale", 1.0)
        self.alert_z = D.get("alert_z", 0.75)
        self.escape_z = D.get("escape_z", 1.5)
        self.gap_norm = GapNormalizer(D.get("gap_tau_bars", 500), D.get("gap_min_n", 50))
        n = cfg["network"]
        self.topo = build_mb_topology(n_pn, n["n_kc"], n["claws_per_kc"], n["kc_sparsity"], n["n_mbon_per_pop"],
                                      w_max=cfg["learning"]["w_max"], seed=cfg["seed"], populations=self.populations)
        self.net = MBNetwork(self.topo, n["episode_ms"], n["eligibility_tau_ms"], seed=cfg["seed"] + 1)
        L = cfg["learning"]
        self.plasticity = DopaminePlasticity(self.topo, L["eta"], L["w_min"], L["w_max"], L["recovery"],
                                             synaptic_scaling=L.get("synaptic_scaling", True),
                                             scaling_target=L.get("scaling_target", 0.5))
        self.rule = L.get("rule", "observed_all")
        self.avoid_scale = L.get("avoid_scale", 0.6)
        self.calibrated_weight: float | None = None
        # [B] MBON homeostatic gain control: each population's drive is read relative to
        # a slow causal EMA of its own past drive, so long-run shares are 1/3 each and
        # decisions depend on stimulus-specific deviations, not on global weight levels.
        A = cfg["agent"]
        self.normalize_drive = A.get("normalize_drive", True)
        self.baseline_alpha = 1.0 / max(1.0, A.get("baseline_tau_bars", 500))
        self.drive_baseline = np.ones(len(self.populations))
        self.baseline_n = 0
        self.kc_dump: list | None = None      # set to [] to record (t, kc_spike_counts) per decision

    def calibrate(self, ds: Dataset, mask: pd.Series, n_samples: int = 24) -> float:
        idx = np.nonzero((mask & ds.valid).to_numpy())[0]
        rng = np.random.default_rng(self.cfg["seed"])
        pick = rng.choice(idx, size=min(n_samples, len(idx)), replace=False)
        self.calibrated_weight = calibrate_sparsity(self.net, ds.pn_rates[pick])
        if self.objective == "danger" and self.danger_threshold is None:
            # threshold from the TRAINING period's outcome distribution only (no test leakage)
            risk = np.maximum(ds.outcomes["mfe_long"], ds.outcomes["mae_long"]).to_numpy()[idx]
            self.danger_threshold = float(np.nanquantile(risk, self.danger_quantile))
        return self.calibrated_weight

    def decide(self, ds: Dataset, t: int, split: str) -> tuple[int, dict, DecisionRecord]:
        ar = float(ds.arousal.iloc[t])
        z_row = ds.z.iloc[t]
        gated = ar < self.cfg["arousal"]["gate_threshold"]
        n_pop = len(self.populations)
        gap_z = None
        if gated:
            res = None
            if self.objective == "danger":
                action, share = 0, np.array([1.0, 0.0])               # resting = CALM
            else:
                action, share = NO_TRADE, np.array([0.0, 0.0, 1.0])
            pop_drive = np.zeros(n_pop)
        else:
            res = self.net.run_episode(ds.pn_rates[t], gain=sensory_gain(ar))
            pop_drive = res.pop_drive
            readout = pop_drive
            if self.normalize_drive:
                if self.baseline_n == 0:
                    self.drive_baseline = np.maximum(pop_drive, 1e-9).copy()
                readout = pop_drive / np.maximum(self.drive_baseline, 1e-9)
                self.drive_baseline += self.baseline_alpha * (pop_drive - self.drive_baseline)   # update after use
                self.baseline_n += 1
            if self.objective == "danger":
                _, share = select_state(readout, 0.0, np.inf)
                gap_z = self.gap_norm(float(share[1] - share[0]))
                action = state_from_z(gap_z, self.alert_z, self.escape_z)
            else:
                action, share = select_action(readout, self.cfg["agent"]["no_trade_bias"], self.cfg["agent"].get("margin", 0.02))
        if self.objective == "danger":
            action_name = STATE_NAMES[action]
            thr = self.danger_threshold if self.danger_threshold is not None else np.inf
            risk = risk_outcome(float(ds.outcomes["mfe_long"].iloc[t]), float(ds.outcomes["mae_long"].iloc[t]))
            label_name = "DANGER" if risk > thr else "CALM"
        else:
            action_name = ACTION_NAMES[action]
            label_name = ACTION_NAMES.get(int(ds.outcomes["label"].iloc[t]), "?")
        rec = DecisionRecord(
            t=str(ds.bars.index[t]), split=split, arousal=ar, channels=channel_summary(z_row), gated=gated,
            kc_active_frac=res.kc_active_frac if res else 0.0,
            pop_drive={p: float(v) for p, v in zip(self.populations, pop_drive)},
            pop_share={p: float(v) for p, v in zip(self.populations, share)},
            mbon_spikes=res.mbon_spikes.tolist() if res else [], action=action_name, label=label_name, gap_z=gap_z,
        )
        state = {"trace": res.kc_trace if res else None, "action": action}
        if res is not None and self.kc_dump is not None:
            self.kc_dump.append((t, res.kc_spikes.astype(np.uint8)))
        return action, state, rec

    def learn_from(self, ds: Dataset, t: int, state: dict, rec: DecisionRecord) -> None:
        if self.objective == "danger":
            risk = risk_outcome(float(ds.outcomes["mfe_long"].iloc[t]), float(ds.outcomes["mae_long"].iloc[t]))
            rec.reward = float(-(risk - self.danger_threshold)) if state["action"] == 0 else float(risk - self.danger_threshold)
            rec.fwd_return_atr = float(ds.outcomes["fwd_return_atr"].iloc[t])
            if state["trace"] is None:
                return
            d = dan_drive_danger(risk, self.danger_threshold, self.false_alarm_scale)
            rec.dan = {p: float(v) for p, v in zip(self.populations, d)}
            self.plasticity.apply(state["trace"], d)
            return
        R = self.cfg["reward"]
        fr = float(ds.outcomes["fwd_return_atr"].iloc[t])
        r = realized_reward(state["action"], fr, R["cost_atr"], R["r_max"])
        cf = counterfactual_rewards(fr, R["cost_atr"], R["r_max"])
        rec.reward, rec.fwd_return_atr = r, fr
        if state["trace"] is None:
            return                                          # rested: no KC activity, nothing to modify
        d = dan_drive(state["action"], r, cf, self.rule, self.avoid_scale)
        rec.dan = {p: float(v) for p, v in zip(self.populations, d)}
        self.plasticity.apply(state["trace"], d)

    def run_split(self, ds: Dataset, mask: pd.Series, split: str, learn: bool, epochs: int = 1,
                  on_decision=None) -> list[DecisionRecord]:
        records: list[DecisionRecord] = []
        m = mask & ds.valid
        for ep in range(epochs if learn else 1):
            ep_records: dict[int, tuple[dict, DecisionRecord]] = {}
            for t, matured in replay(ds, m):
                for i in matured:
                    st, rc = ep_records.pop(i)
                    R = self.cfg["reward"]
                    if learn:
                        self.learn_from(ds, i, st, rc)
                    elif self.objective == "danger":
                        risk = risk_outcome(float(ds.outcomes["mfe_long"].iloc[i]), float(ds.outcomes["mae_long"].iloc[i]))
                        rc.reward = float(-(risk - self.danger_threshold)) if st["action"] == 0 else float(risk - self.danger_threshold)
                        rc.fwd_return_atr = float(ds.outcomes["fwd_return_atr"].iloc[i])
                    else:
                        fr = float(ds.outcomes["fwd_return_atr"].iloc[i])
                        rc.reward = realized_reward(st["action"], fr, R["cost_atr"], R["r_max"]); rc.fwd_return_atr = fr
                if t < 0:
                    break
                action, st, rc = self.decide(ds, t, split)
                ep_records[t] = (st, rc)
                if ep == (epochs if learn else 1) - 1:
                    records.append(rc)
                if on_decision is not None:
                    on_decision(rc, self)
        return records
