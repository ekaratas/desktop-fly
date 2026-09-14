import numpy as np

from trader.features.raw import atr
from trader.labels.excursion import LONG, NO_TRADE, SHORT, future_excursions, opportunity_labels
from trader.learning.reward import realized_reward


def test_excursions_match_explicit_loop(bars):
    hz = 12
    a = atr(bars)
    exc = future_excursions(bars, a, hz)
    for i in (500, 1234, 3000):
        c = bars["close"].iloc[i]
        hi = bars["high"].iloc[i + 1: i + 1 + hz].max(); lo = bars["low"].iloc[i + 1: i + 1 + hz].min()
        assert np.isclose(exc["mfe_long"].iloc[i], (hi - c) / a.iloc[i])
        assert np.isclose(exc["mae_long"].iloc[i], (c - lo) / a.iloc[i])
        assert np.isclose(exc["fwd_return_atr"].iloc[i], (bars["close"].iloc[i + hz] - c) / a.iloc[i])
    assert exc["mfe_long"].iloc[-hz:].isna().all()


def test_labels_are_symmetric_and_cover_all_classes(bars):
    exc = future_excursions(bars, atr(bars), 12)
    lab = opportunity_labels(exc, 1.0, 1.8)
    assert set(lab.unique()) >= {LONG, SHORT, NO_TRADE}
    mirrored = exc.rename(columns={"mfe_long": "mae_long", "mae_long": "mfe_long"})
    lab_m = opportunity_labels(mirrored, 1.0, 1.8)
    swap = lab.replace({LONG: SHORT, SHORT: LONG})
    assert (lab_m[lab >= 0] == swap[lab >= 0]).all()


def test_reward_costs_and_abstain():
    assert realized_reward(NO_TRADE, 5.0) == 0.0
    assert realized_reward(LONG, 1.0, cost_atr=0.15) == 0.85
    assert realized_reward(SHORT, 1.0, cost_atr=0.15) == -1.15
    assert realized_reward(LONG, 100.0, r_max=3.0) == 3.0
    assert realized_reward(LONG, float("nan")) == 0.0
