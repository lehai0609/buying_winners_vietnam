import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from src.bwv import cv
from src.bwv import backtest as bt


def _make_synthetic_ohlcv(start="2018-01-01", end="2021-12-31", tickers=None):
    """
    Create a simple synthetic OHLCV MultiIndex DataFrame for testing.
    Close prices follow an exponential trend per ticker to create deterministic momentum.
    Open == close (no intraday move) to simplify execution logic in backtest.
    Volume is constant positive to avoid halts.
    """
    if tickers is None:
        tickers = ["AAA", "BBB"]
    dates = pd.bdate_range(start=start, end=end)
    rows = []
    for tk_idx, tk in enumerate(tickers):
        # different daily drift per ticker to create separable momentum
        drift = 0.0005 + 0.0005 * tk_idx  # AAA slightly higher than BBB
        price = 100.0 * (1.0 + drift) ** np.arange(len(dates))
        for d, p in zip(dates, price):
            rows.append({"date": d, "ticker": tk, "open": float(p), "high": float(p), "low": float(p), "close": float(p), "volume": 1000})
    df = pd.DataFrame(rows).set_index(["date", "ticker"])
    return df.sort_index()


def _make_universe_monthly(ohlcv_df):
    # simple universe: all tickers eligible at each month-end
    dates = pd.DatetimeIndex(sorted(set(ohlcv_df.index.get_level_values("date"))))
    month_ends = pd.DatetimeIndex(pd.Series(dates).groupby(dates.to_period("M")).max().values)
    tickers = sorted(set(ohlcv_df.index.get_level_values("ticker")))
    df = pd.DataFrame(True, index=month_ends, columns=tickers, dtype=object)
    return df


def test_grid_search_determinism():
    ohlcv = _make_synthetic_ohlcv(start="2019-01-01", end="2020-12-31", tickers=["AAA", "BBB"])
    universe = _make_universe_monthly(ohlcv)
    J_list = [3, 6]
    K_list = [1, 3]

    # Run grid search twice and assert identical results (deterministic)
    res1 = cv.grid_search(ohlcv, universe, J_list, K_list, config={"skip_days": 5})
    res2 = cv.grid_search(ohlcv, universe, J_list, K_list, config={"skip_days": 5})

    # Use pandas assert_frame_equal which treats NaN==NaN
    # reset index to ignore potential index name differences
    assert_frame_equal(res1.reset_index(drop=True), res2.reset_index(drop=True), check_dtype=False)


def test_cost_sensitivity_monotonic():
    ohlcv = _make_synthetic_ohlcv(start="2019-01-01", end="2020-12-31", tickers=["AAA", "BBB"])
    universe = _make_universe_monthly(ohlcv)
    # pick a representative (J,K)
    J = 6
    K = 3
    cost_list = [0.0, 100.0]  # 0 bps vs 100 bps
    caps = [0.05]

    df = cv.cost_sensitivity(ohlcv, universe, J, K, cost_list, caps, config={"skip_days": 5})
    # Ensure we have rows for both costs
    assert set(df["transaction_cost_bps"].tolist()) == set(cost_list)
    # Sharpe at higher costs should not be higher than at lower costs (monotone non-increasing)
    df_sorted = df.sort_values("transaction_cost_bps")
    sharpes = df_sorted["Sharpe"].tolist()
    # Non-numeric Sharpe (nan) tolerated; only check when both are finite
    if all(np.isfinite(sharpes)):
        assert sharpes[0] >= sharpes[1]


def test_walk_forward_basic_smoke():
    # Create slightly longer series to allow at least one fold with train=12, valid=6
    ohlcv = _make_synthetic_ohlcv(start="2016-01-01", end="2020-12-31", tickers=["AAA", "BBB", "CCC"])
    universe = _make_universe_monthly(ohlcv)
    J_list = [3, 6]
    K_list = [1, 3]
    cfg = {"cv": {"train_months": 24, "valid_months": 6, "step": 6}, "skip_days": 5}

    wf = cv.walk_forward(ohlcv, universe, J_list, K_list, config=cfg)
    # basic schema checks
    assert "fold_id" in wf.columns
    assert "J" in wf.columns and "K" in wf.columns
    # At least one fold row produced
    assert len(wf) > 0
    # For each fold there should be rows equal to number of grid points
    grid_size = len(J_list) * len(K_list)
    # group by fold and check counts
    counts = wf.groupby("fold_id").size().unique()
    assert all(count == grid_size for count in counts)
