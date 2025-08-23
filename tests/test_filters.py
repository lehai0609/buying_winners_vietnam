import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import pytest

from bwv import filters


def _make_panel(dates, tickers, close_vals, volume_vals):
    """
    Helper to build minimal OHLCV DataFrame with MultiIndex (date, ticker).
    close_vals and volume_vals can be scalars or dicts keyed by ticker -> list.
    Returns DataFrame with columns open, high, low, close, volume.
    """
    rows = []
    for i, d in enumerate(dates):
        for t in tickers:
            if isinstance(close_vals, dict):
                c = close_vals[t][i]
            else:
                c = close_vals
            if isinstance(volume_vals, dict):
                v = volume_vals[t][i]
            else:
                v = volume_vals
            rows.append({"date": d, "ticker": t, "open": c, "high": c, "low": c, "close": c, "volume": v})
    df = pd.DataFrame(rows)
    df = df.set_index(["date", "ticker"]).sort_index()
    return df


def test_min_history_no_lookahead():
    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    tickers = ["AAA"]
    # close values irrelevant; provide 10 non-null closes
    df = _make_panel(dates, tickers, close_vals=1.0, volume_vals=100)
    # min_days=5 -> eligibility should become True on day 6 (since shift(1) excludes current day)
    mh = filters.min_history(df, min_days=5)
    # index aligned to daily dates
    assert mh.shape[0] == len(dates)
    # day 5 (index 2025-01-05) should be False
    assert mh.loc[pd.Timestamp("2025-01-05"), "AAA"] is False
    # day 6 should be True (5 prior trading observations)
    assert mh.loc[pd.Timestamp("2025-01-06"), "AAA"] is True


def test_price_floor_masks_correctly():
    dates = pd.date_range("2025-02-01", periods=2, freq="D")
    tickers = ["AAA", "BBB"]
    close = {"AAA": [1500.0, 1500.0], "BBB": [900.0, 1100.0]}
    vol = {"AAA": [100, 100], "BBB": [100, 100]}
    df = _make_panel(dates, tickers, close_vals=close, volume_vals=vol)
    pf = filters.price_floor(df, min_price=1000.0)
    # AAA always True
    assert all(pf["AAA"])
    # BBB first day False, second day True
    assert pf.loc[pd.Timestamp("2025-02-01"), "BBB"] is False
    assert pf.loc[pd.Timestamp("2025-02-02"), "BBB"] is True


def test_adv_filter_shifted_window():
    dates = pd.date_range("2025-03-01", periods=5, freq="D")
    tickers = ["AAA"]
    # close 1.0 so trading_value == volume
    volumes = {"AAA": [100, 200, 300, 400, 500]}
    df = _make_panel(dates, tickers, close_vals=1.0, volume_vals=volumes)
    # window=2, min_adv=150 -> rolling means: [100,150,250,350,450], shifted -> eligibility at day3 onward (2025-03-03 True)
    adv_mask = filters.adv_filter(df, min_adv_vnd=150.0, window=2)
    assert adv_mask.loc[pd.Timestamp("2025-03-02"), "AAA"] is False
    assert adv_mask.loc[pd.Timestamp("2025-03-03"), "AAA"] is True


def test_non_trading_days_filter_counts_and_shifts():
    dates = pd.date_range("2025-04-01", periods=5, freq="D")
    tickers = ["AAA"]
    # volumes with zeros on day2 and day3
    volumes = {"AAA": [10, 0, 0, 10, 10]}
    df = _make_panel(dates, tickers, close_vals=100.0, volume_vals=volumes)
    # Use window_days=3 and max_ntd=1 -> for date=2025-04-04 (day4), previous 3-day window (day1..day3) has 2 non-trading -> not eligible
    ntd_mask = filters.non_trading_days_filter(df, max_ntd=1, window_days=3)
    assert ntd_mask.loc[pd.Timestamp("2025-04-04"), "AAA"] is False
    # For day5, previous window day2..day4 has 2 non-trading -> still False
    assert ntd_mask.loc[pd.Timestamp("2025-04-05"), "AAA"] is False


def test_compose_universe_combines_rules_monthly():
    # Build a 10-day panel for two tickers where only AAA meets all criteria by the final month-end
    dates = pd.date_range("2025-05-01", periods=10, freq="D")
    tickers = ["AAA", "BBB"]
    # AAA: sufficient history, price >=1000, decent volume
    close_A = [1000.0] * 10
    vol_A = [1000] * 10
    # BBB: low price sometimes and low ADV
    close_B = [900.0, 900.0, 900.0, 1200.0, 1200.0, 1200.0, 1200.0, 1200.0, 1200.0, 1200.0]
    vol_B = [10] * 10
    close = {"AAA": close_A, "BBB": close_B}
    vol = {"AAA": vol_A, "BBB": vol_B}
    df = _make_panel(dates, tickers, close_vals=close, volume_vals=vol)
    # set small windows so the test is deterministic within 10 days
    uni = filters.compose_universe(
        df,
        monthly=True,
        min_history_days=3,
        min_price=1000.0,
        min_adv_vnd=500000.0,
        adv_window=3,
        max_ntd=2,
        ntd_window=3,
    )
    # uni index should contain the last trading day in May -> 2025-05-10
    assert pd.Timestamp("2025-05-10") in uni.index
    # AAA should be eligible on month-end
    assert uni.loc[pd.Timestamp("2025-05-10"), "AAA"] is True
    # BBB should be False because price floor and adv fail
    assert uni.loc[pd.Timestamp("2025-05-10"), "BBB"] is False
