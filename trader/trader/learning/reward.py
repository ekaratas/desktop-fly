"""Reward / punishment signal ([C] trading engineering).

r = clip(direction * forward_return_ATR - cost_ATR, -r_max, r_max) for LONG/SHORT;
NO_TRADE earns 0. The cost term makes tiny edges net-negative so that abstaining
can win. The scalar is later split into PAM-like (reward) and PPL1-like
(punishment) dopamine drive in trader.learning.dopamine ([B]).
"""
from __future__ import annotations

import numpy as np

from ..labels.excursion import LONG, NO_TRADE, SHORT


def realized_reward(action: int, fwd_return_atr: float, cost_atr: float = 0.15, r_max: float = 3.0) -> float:
    if action == NO_TRADE or not np.isfinite(fwd_return_atr):
        return 0.0
    direction = 1.0 if action == LONG else -1.0
    return float(np.clip(direction * fwd_return_atr - cost_atr, -r_max, r_max))


def counterfactual_rewards(fwd_return_atr: float, cost_atr: float, r_max: float) -> np.ndarray:
    """Rewards each of the three actions would have earned (for metrics/regret only)."""
    return np.array([realized_reward(a, fwd_return_atr, cost_atr, r_max) for a in (LONG, SHORT, NO_TRADE)])
