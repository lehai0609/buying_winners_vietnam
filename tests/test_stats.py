import numpy as np
import pandas as pd
import pytest

from src.bwv import stats

RNG = np.random.RandomState(12345)


def make_monthly_index(start="2010-01-31", periods=120):
    return pd.date_range(start=start, periods=periods, freq="ME")


def test_alpha_newey_west_recovers_alpha():
    # Build synthetic data: benchmark ~ N(0.005, 0.02), strategy = 0.5*benchmark + alpha + AR(1) noise
    periods = 120
    alpha_true = 0.002  # monthly alpha
    beta_true = 0.5
    eps = np.zeros(periods)
    phi = 0.2
    sigma = 0.01
    # AR(1) noise
    for t in range(1, periods):
        eps[t] = phi * eps[t - 1] + RNG.normal(scale=sigma)
    bench = pd.Series(RNG.normal(loc=0.005, scale=0.02, size=periods), index=make_monthly_index(periods=periods))
    strat = beta_true * bench + alpha_true + eps
    res = stats.alpha_newey_west(strat, bench, lags=6, rf=0.0, periods_per_year=12)
    assert "alpha_monthly" in res
    # alpha estimate should be close to true up to reasonable tolerance
    assert abs(res["alpha_monthly"] - alpha_true) < 6e-4
    assert res["n_obs"] == periods


def test_bootstrap_cis_contains_point_estimate():
    periods = 200
    alpha_true = 0.0015
    beta_true = 0.4
    # simple iid noise here
    bench = pd.Series(RNG.normal(loc=0.004, scale=0.015, size=periods), index=make_monthly_index(periods=periods))
    strat = beta_true * bench + alpha_true + RNG.normal(scale=0.01, size=periods)
    # compute point estimate
    pt = stats.alpha_newey_west(strat, bench, lags=6, rf=0.0, periods_per_year=12)
    alpha_ann_pt = pt["alpha_annual"]
    boot = stats.bootstrap_cis(strat, benchmark=bench, n=300, block_size=6, alpha=0.05, seed=42, lags=6, rf=0.0, periods_per_year=12)
    assert "alpha_ann_ci" in boot
    lower, upper = boot["alpha_ann_ci"]
    assert lower <= alpha_ann_pt <= upper


def test_subperiod_stats_structure():
    periods = 60
    bench = pd.Series(RNG.normal(loc=0.006, scale=0.02, size=periods), index=make_monthly_index(periods=periods))
    strat = 0.6 * bench + 0.001 + RNG.normal(scale=0.01, size=periods)
    # define two subperiods
    idx = make_monthly_index(periods=periods)
    p1 = (idx[0], idx[29])
    p2 = (idx[30], idx[-1])
    out = stats.subperiod_stats(strat, bench, [p1, p2], lags=4, rf=0.0, periods_per_year=12)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == 2
    assert "alpha_monthly" in out.columns
