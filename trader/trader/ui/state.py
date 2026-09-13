"""State bridge ([C]): the JSON the dashboard and a future desktop creature read.

`state.json` is rewritten atomically after every decision; it carries the latest
decision, population activities, arousal, the last N candles with actions, a
running cost-adjusted PnL in ATR units and a suggested creature behavior
(sleep / explore / alert / hunt-long / hunt-short / rest / reward / aversive).
Nothing in here feeds back into the model.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque

import numpy as np
import pandas as pd


def creature_behavior(action: str, arousal: float, gated: bool, last_reward: float | None) -> str:
    if last_reward is not None and last_reward > 0.5:
        return "reward"
    if last_reward is not None and last_reward < -0.5:
        return "aversive"
    if action == "LONG":
        return "hunt_long"
    if action == "SHORT":
        return "hunt_short"
    if gated or arousal < 0.25:
        return "sleep"
    if arousal < 0.6:
        return "explore"
    return "alert"


class StateWriter:
    def __init__(self, state_path: str, decisions_path: str, bars: pd.DataFrame, keep_bars: int = 160):
        self.state_path, self.decisions_path = state_path, decisions_path
        self.bars = bars
        self.keep = keep_bars
        self.count = 0
        self.recent: deque[dict] = deque(maxlen=keep_bars)
        self.pnl_atr = 0.0
        self.pnl_curve: deque[float] = deque(maxlen=600)
        self.last_reward: float | None = None
        self.stats = {"LONG": 0, "SHORT": 0, "NO_TRADE": 0}
        self._pending = {}

    def update(self, rec, organism, quiet: bool = False) -> None:
        self.count += 1
        self.stats[rec.action] = self.stats.get(rec.action, 0) + 1
        ts = pd.Timestamp(rec.t)
        try:
            b = self.bars.loc[ts]
            candle = {"t": rec.t, "o": float(b["open"]), "h": float(b["high"]), "l": float(b["low"]), "c": float(b["close"]),
                      "v": float(b["volume"]), "action": rec.action, "arousal": rec.arousal}
        except KeyError:
            candle = {"t": rec.t, "action": rec.action, "arousal": rec.arousal}
        self.recent.append(candle)
        # rewards mature later; pick up any that were filled in since
        if rec.reward is not None:
            self._account(rec)
        else:
            self._pending[rec.t] = rec
        for k in list(self._pending):
            r = self._pending[k]
            if r.reward is not None:
                self._account(r); del self._pending[k]
        W = organism.topo.kc_mbon
        pops = {p: float(W[:, organism.topo.mbon_pop == i].mean()) for i, p in enumerate(("long", "short", "avoid"))}
        state = {
            "updated": time.time(), "count": self.count, "split": rec.split, "t": rec.t,
            "decision": rec.action, "label": rec.label, "arousal": rec.arousal, "gated": rec.gated,
            "channels": rec.channels, "pop_share": rec.pop_share, "pop_drive": rec.pop_drive,
            "kc_active_frac": rec.kc_active_frac, "mbon_spikes": rec.mbon_spikes,
            "weights_mean": pops, "plasticity_updates": organism.plasticity.updates,
            "last_reward": self.last_reward, "pnl_atr": self.pnl_atr, "pnl_curve": list(self.pnl_curve),
            "action_counts": self.stats, "candles": list(self.recent),
            "behavior": creature_behavior(rec.action, rec.arousal, rec.gated, self.last_reward),
        }
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, default=_default)
        os.replace(tmp, self.state_path)

    def _account(self, rec) -> None:
        if rec.action != "NO_TRADE":
            self.pnl_atr += rec.reward
            self.last_reward = rec.reward
        else:
            self.last_reward = 0.0
        self.pnl_curve.append(self.pnl_atr)
        with open(self.decisions_path, "a") as f:
            f.write(rec.to_json() + "\n")


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and o != o:
        return None
    return str(o)
