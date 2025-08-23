import pandas as pd
import numpy as np

from bwv import portfolio as port


def _make_top_mask(dates, tickers, truth_map):
    """
    Build a top_mask DataFrame (object-bool dtype) indexed by dates with given tickers.
    truth_map: dict date -> list of tickers included (or set)
    """
    df = pd.DataFrame(False, index=pd.DatetimeIndex(dates), columns=tickers, dtype=object)
    for d, members in truth_map.items():
        for t in members:
            df.at[pd.Timestamp(d), t] = True
    return df


def test_build_cohorts_freezes_and_sorts():
    dates = pd.to_datetime(["2025-01-31", "2025-02-28"])
    tickers = ["B", "A", "C"]
    truth = {
        "2025-01-31": ["B", "A"],
        "2025-02-28": ["C"],
    }
    top_mask = _make_top_mask(dates, tickers, truth)
    cohorts = port.build_cohorts(top_mask, K=3)
    # keys present
    assert pd.Timestamp("2025-01-31") in cohorts
    assert pd.Timestamp("2025-02-28") in cohorts
    # members sorted ascending
    assert list(cohorts[pd.Timestamp("2025-01-31")]) == ["A", "B"]
    assert list(cohorts[pd.Timestamp("2025-02-28")]) == ["C"]


def test_target_weights_overlap_and_sums():
    # Create 3 formation months with distinct cohorts, each with 2 tickers
    dates = pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31"])
    tickers = ["A1", "A2", "B1", "B2", "C1", "C2"]
    truth = {
        "2025-01-31": ["A1", "A2"],
        "2025-02-28": ["B1", "B2"],
        "2025-03-31": ["C1", "C2"],
    }
    top_mask = _make_top_mask(dates, tickers, truth)
    cohorts = port.build_cohorts(top_mask, K=3)
    # produce weights for the same formation months
    weights = port.target_weights_from_cohorts(cohorts, dates, K=3, constraints=None, universe_mask=None)
    # At the last formation date (2025-03-31), active cohorts are Jan, Feb, Mar.
    # Each cohort weight = 1/3, each member within cohort equal-weight -> 1/3 * 1/2 = 1/6
    last = pd.Timestamp("2025-03-31")
    expected_per = 1.0 / 6.0
    # verify per-name weight approximately expected
    vals = weights.loc[last, ["A1", "A2", "B1", "B2", "C1", "C2"]].to_numpy()
    assert np.allclose(vals, expected_per, atol=1e-12)
    vals = weights.loc[last, ["A1", "A2", "B1", "B2", "C1", "C2"]].to_numpy()
    assert np.allclose(vals, expected_per, atol=1e-12)
    # row sum should be ~1.0
    assert np.isclose(weights.loc[last].sum(), 1.0, atol=1e-12)


def test_universe_mask_excludes_and_leaves_cash_buffer():
    # Single formation month with cohort of two tickers, but universe excludes one at rebalance
    dates = pd.to_datetime(["2025-04-30"])
    tickers = ["X", "Y"]
    truth = {"2025-04-30": ["X", "Y"]}
    top_mask = _make_top_mask(dates, tickers, truth)
    cohorts = port.build_cohorts(top_mask, K=1)
    # Universe mask excludes 'Y' at formation date
    u_mask = pd.DataFrame(True, index=dates, columns=tickers, dtype=object)
    u_mask.at[pd.Timestamp("2025-04-30"), "Y"] = False
    weights = port.target_weights_from_cohorts(cohorts, dates, K=1, constraints=None, universe_mask=u_mask)
    # With K=1, cohort weight = 1.0; only X is eligible -> equal-weight within cohort => X gets 1.0
    assert weights.loc[pd.Timestamp("2025-04-30"), "X"] == 1.0
    assert weights.loc[pd.Timestamp("2025-04-30"), "Y"] == 0.0


def test_per_name_cap_and_renormalization_behavior():
    # One formation month, single cohort with 2 members; set per-name cap below raw weight
    dates = pd.to_datetime(["2025-05-31"])
    tickers = ["M1", "M2"]
    truth = {"2025-05-31": ["M1", "M2"]}
    top_mask = _make_top_mask(dates, tickers, truth)
    cohorts = port.build_cohorts(top_mask, K=3)  # K=3 so cohort weight = 1/3
    # per-name raw = 1/3 * 1/2 = 1/6 ~ 0.1666667
    constraints = {"per_name_cap": 0.10, "renorm_within_cohort": True}
    weights = port.target_weights_from_cohorts(cohorts, dates, K=3, constraints=constraints, universe_mask=None)
    # Both members should be capped to 0.10 each and residual left as cash (no uncapped to absorb)
    assert np.isclose(weights.loc[pd.Timestamp("2025-05-31"), "M1"], 0.10, atol=1e-12)
    assert np.isclose(weights.loc[pd.Timestamp("2025-05-31"), "M2"], 0.10, atol=1e-12)
    # Sum should be 0.20 which is less than cohort weight 1/3 (~0.3333)
    assert weights.loc[pd.Timestamp("2025-05-31")].sum() < 1.0


def test_generate_trades_timing_and_deltas():
    # trading dates are daily; choose formation_date that is a trading date
    trading_dates = pd.date_range("2025-06-01", periods=10, freq="D")
    formation_date = trading_dates[2]  # third trading day
    prev = pd.Series({"A": 0.10, "B": 0.20})
    targ = pd.Series({"A": 0.15, "B": 0.05, "C": 0.10})
    trades = port.generate_trades(prev, targ, formation_date, trading_dates, t_plus=1, settlement_days=2)
    # exec_date should be trading_dates[pos + 1], settle = pos + 1 + 2
    pos = list(trading_dates).index(formation_date)
    assert trades["exec_date"].iloc[0] == trading_dates[pos + 1]
    assert trades["settle_date"].iloc[0] == trading_dates[pos + 3]
    # deltas:
    assert np.isclose(trades.loc[(formation_date, "A"), "delta_w"], 0.05)
    assert np.isclose(trades.loc[(formation_date, "B"), "delta_w"], -0.15)
    assert np.isclose(trades.loc[(formation_date, "C"), "delta_w"], 0.10)
    # actions
    assert trades.loc[(formation_date, "A"), "action"] == "BUY"
    assert trades.loc[(formation_date, "B"), "action"] == "SELL"
    assert trades.loc[(formation_date, "C"), "action"] == "BUY"
