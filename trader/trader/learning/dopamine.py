"""Outcome → per-population dopamine (depression) drive.

Measured biology ([A]): dopaminergic neurons (PPL1 for punishment, PAM for reward)
innervate MB compartments and *depress* the KC→MBON synapses of co-active KCs in
their compartment. Each compartment therefore learns "how bad was the outcome of
the behavior this MBON promotes, given this stimulus".

Our mapping ([B]): three MBON populations promote LONG, SHORT and NO_TRADE (avoid).
Each has its own punishment-coding DAN. Two rules are available:

  "observed_all" (default, [C] justified by the market): the future price reveals
      what *every* action would have earned, so each population's DAN fires in
      proportion to how much its action would have lost:
          dan_long  = max(0, -r_long)
          dan_short = max(0, -r_short)
          dan_avoid = max(0, max(r_long, r_short))   # abstaining missed a real edge
      This is symmetric between LONG and SHORT and has no rich-get-richer feedback.
  "chosen_only" ([B], closer to a foraging fly that only tastes what it chose):
      punishment depresses the chosen population when r < 0; reward depresses the
      competing populations when r > 0. Unstable in practice (see ARCHITECTURE.md).
"""
from __future__ import annotations

import numpy as np

from ..labels.excursion import LONG, NO_TRADE, SHORT

POP_OF_ACTION = {LONG: 0, SHORT: 1, NO_TRADE: 2}


def dan_drive(action: int, reward: float, counterfactual: np.ndarray, rule: str = "observed_all",
              avoid_scale: float = 0.6) -> np.ndarray:
    """(3,) non-negative depression drive per MBON population (long, short, avoid)."""
    d = np.zeros(3)
    r_long, r_short = float(counterfactual[0]), float(counterfactual[1])
    if rule == "observed_all":
        d[0] = max(0.0, -r_long)
        d[1] = max(0.0, -r_short)
        d[2] = avoid_scale * max(0.0, r_long, r_short)
        return d
    if rule == "chosen_only":
        chosen = POP_OF_ACTION[action]
        if action != NO_TRADE:
            if reward < 0:
                d[chosen] = -reward
            elif reward > 0:
                d[[p for p in range(3) if p != chosen]] = reward
        return d
    raise ValueError(f"unknown dopamine rule {rule!r}")
