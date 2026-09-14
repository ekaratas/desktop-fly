"""Action selection ([B] readout, [C] bias): MBON population drive → LONG / SHORT / NO_TRADE.

Drive is normalized across the three populations; `avoid` gets an additive
no-trade bias so ties fall to abstaining. Low arousal short-circuits to NO_TRADE.
"""
from __future__ import annotations

import numpy as np

from ..labels.excursion import LONG, NO_TRADE, SHORT

POP_TO_ACTION = {0: LONG, 1: SHORT, 2: NO_TRADE}


def select_action(pop_drive: np.ndarray, no_trade_bias: float = 0.05, margin: float = 0.02) -> tuple[int, np.ndarray]:
    total = pop_drive.sum()
    p = pop_drive / total if total > 0 else np.array([1 / 3, 1 / 3, 1 / 3])
    scores = p.copy()
    scores[2] += no_trade_bias
    best = int(np.argmax(scores))
    if best != 2:
        others = np.delete(scores, best)
        if scores[best] - others.max() < margin:       # ambiguous → abstain
            best = 2
    return POP_TO_ACTION[best], p
