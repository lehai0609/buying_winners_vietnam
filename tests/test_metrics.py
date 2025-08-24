import numpy as np
import pandas as pd
import pytest

from src.bwv import metrics


def make_monthly_index(start="2010-01-31", periods=120):
    return pd.date_range(start=start, periods=periods, freq="ME")


def test_cagr_constant_monthly():
    # 120 months of 1% monthly returns -> expected CAGR = (1.01^12)-1
    returns = pd.Series([0.01] * 120, index=make_monthly_index(periods=120))
    c = metrics.cagr(returns, periods_per_year=12)
    expected = (1.01 ** 12) - 1.0
    assert pytest.approx(expected, rel=1e-9) == c


def test_drawdown_known_series():
    # construct returns where peak = 1.2 then drop to 0.9 -> drawdown = 0.25
    returns = pd.Series([0.2, 0.0, 0.0, -0.25, 0.05, 0.05], index=make_monthly_index(periods=6))
    dd = metrics.drawdown_stats(returns)
    assert pytest.approx(0.25, rel=1e-12) == dd["mdd"]
    assert dd["mdd_start"] is not None
    assert dd["mdd_end"] is not None


def test_turnover_stats_simple():
    # two-period weight transition: A,B initially 0.5/0.5 -> then 0.0/1.0
    idx = make_monthly_index(periods=2)
    w0 = {"A": 0.5, "B": 0.5}
    w1 = {"A": 0.0, "B": 1.0}
    weights = pd.DataFrame([w0, w1], index=idx)
    t = metrics.turnover_stats(weights)
    # turnover should be 0.5 * (|0.0-0.5| + |1.0-0.5|) = 0.5
    assert pytest.approx(0.5, rel=1e-12) == t["mean_turnover"]
    assert t["turnover_series"].iloc[0] == pytest.approx(0.5, rel=1e-12)


def test_perf_summary_with_benchmark_alpha():
    # strategy = benchmark + 0.001 monthly alpha
    rng = np.random.RandomState(42)
    bench = pd.Series(rng.normal(loc=0.005, scale=0.01, size=60), index=make_monthly_index(periods=60))
    strat = bench + 0.001
    res = metrics.perf_summary(strat, benchmark=bench, rf=0.0, periods_per_year=12)
    # alpha_monthly should be approximately 0.001 (within tolerance)
    assert "alpha_monthly" in res
    assert abs(res["alpha_monthly"] - 0.001) < 1e-6
    # alpha annualized should be > 0
    assert res["alpha_ann"] > 0
