"""Perturbing bars after t must not change any input at t (features, z-scores, PN rates, arousal)."""
import numpy as np

from trader.agent.arousal import arousal_series
from trader.features.normalize import causal_zscore
from trader.features.raw import available_features, compute_raw_features
from trader.features.sensory import encode_rates


def _inputs(bars, cfg):
    raw = compute_raw_features(bars)
    z = causal_zscore(raw[available_features(raw)], window=cfg["features"]["norm_window"])
    return raw, z, encode_rates(z), arousal_series(z, cfg["arousal"]["tau_bars"])


def test_future_perturbation_leaves_present_unchanged(bars, cfg):
    t = 2500
    raw_a, z_a, r_a, ar_a = _inputs(bars, cfg)
    pert = bars.copy().astype(float)
    rng = np.random.default_rng(0)
    for col in ("open", "high", "low", "close", "volume", "taker_buy_volume", "count"):
        pert.iloc[t + 1:, pert.columns.get_loc(col)] *= np.exp(rng.standard_normal(len(pert) - t - 1) * 0.3)
    pert["high"] = pert[["open", "high", "close"]].max(axis=1); pert["low"] = pert[["open", "low", "close"]].min(axis=1)
    raw_b, z_b, r_b, ar_b = _inputs(pert, cfg)
    np.testing.assert_allclose(raw_a.iloc[: t + 1].to_numpy(), raw_b.iloc[: t + 1].to_numpy(), equal_nan=True)
    np.testing.assert_allclose(z_a.iloc[: t + 1].to_numpy(), z_b.iloc[: t + 1].to_numpy(), equal_nan=True)
    np.testing.assert_allclose(r_a[: t + 1], r_b[: t + 1])
    np.testing.assert_allclose(ar_a.iloc[: t + 1].to_numpy(), ar_b.iloc[: t + 1].to_numpy())
    # and the future did change (the test is not vacuous)
    assert not np.allclose(np.nan_to_num(raw_a.iloc[t + 20:].to_numpy()), np.nan_to_num(raw_b.iloc[t + 20:].to_numpy()))


def test_normalization_excludes_current_row(bars, cfg):
    raw = compute_raw_features(bars)
    f = raw[["ret1"]].copy()
    f.iloc[3000, 0] = 50.0                                        # huge outlier at t
    z = causal_zscore(f, window=100)
    # the row itself is clipped by the outlier but its mean/std came from the past only:
    z_ref = causal_zscore(raw[["ret1"]], window=100)
    assert np.isclose(z.iloc[2999, 0], z_ref.iloc[2999, 0])       # row before is untouched
    assert z.iloc[3000, 0] == 4.0                                 # clipped, computed with past stats


def test_extended_features_are_causal_and_optional(bars, cfg):
    from trader.data.synthetic import make_synthetic_extras
    from trader.features.raw import CORE_FEATURES, EXTENDED_FEATURES, available_features

    plain = compute_raw_features(bars)
    assert available_features(plain) == CORE_FEATURES                     # no extras → core only
    ext = make_synthetic_extras(bars, seed=1)
    full = compute_raw_features(ext)
    assert set(EXTENDED_FEATURES) <= set(available_features(full))
    t = 2500
    pert = ext.copy().astype(float)
    for col in ("funding_rate", "premium_index", "open_interest", "taker_ls_ratio"):
        pert.iloc[t + 1:, pert.columns.get_loc(col)] *= 1.5
    full_b = compute_raw_features(pert)
    np.testing.assert_allclose(full.iloc[: t + 1][EXTENDED_FEATURES].to_numpy(), full_b.iloc[: t + 1][EXTENDED_FEATURES].to_numpy(), equal_nan=True)


def test_asof_merge_never_uses_future_rows(bars):
    import pandas as pd
    from trader.data.binance_extra import attach_extras

    times = bars.index[::8] + pd.Timedelta("30min")                        # events between bar boundaries
    funding = pd.DataFrame({"time": times, "funding_rate": np.arange(len(times), dtype=float)})
    out = attach_extras(bars, "1h", funding=funding)
    for i in (100, 1001, 2222):
        close_t = bars.index[i] + pd.Timedelta("1h")
        expected = funding[funding["time"] <= close_t]["funding_rate"].iloc[-1]
        assert out["funding_rate"].iloc[i] == expected
