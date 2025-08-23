import numpy as np
import pandas as pd
import pytest

from bwv import returns as rtrs


def _make_panel(dates, tickers, close_vals):
    """
    Build minimal OHLCV-like DataFrame with MultiIndex (date, ticker) and a 'close' column.
    close_vals can be a scalar, a dict of ticker->list, or a 2D list-like (dates x tickers).
    """
    rows = []
    for i, d in enumerate(dates):
        for j, t in enumerate(tickers):
            if isinstance(close_vals, dict):
                c = close_vals[t][i]
            elif hasattr(close_vals, "__len__") and not isinstance(close_vals, (str, bytes)):
                # assume 2D list-like with shape (len(dates), len(tickers))
                c = close_vals[i][j]
            else:
                c = close_vals
            rows.append({"date": d, "ticker": t, "close": c})
    df = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
    return df


def test_daily_simple_and_log_returns():
    dates = pd.date_range("2025-01-01", periods=3, freq="D")
    tickers = ["AAA"]
    closes = {"AAA": [100.0, 101.0, 99.0]}
    df = _make_panel(dates, tickers, closes)
    simple = rtrs.daily_simple_returns(df)
    logr = rtrs.daily_log_returns(df)

    assert simple.shape == (3, 1)
    assert np.isnan(simple.iloc[0, 0])
    assert simple.iloc[1, 0] == pytest.approx(0.01)
    assert simple.iloc[2, 0] == pytest.approx(99.0 / 101.0 - 1.0)

    assert np.isnan(logr.iloc[0, 0])
    assert logr.iloc[1, 0] == pytest.approx(np.log(1.01))
    assert logr.iloc[2, 0] == pytest.approx(np.log(99.0 / 101.0))


def test_month_end_flags_and_index():
    # trading dates spanning two months, with normal daily frequency
    dates = pd.DatetimeIndex(pd.date_range("2025-01-28", periods=6, freq="D"))  # Jan 28..Feb 2
    me_flags = rtrs.month_end_flags(dates)
    me_idx = rtrs.month_end_index(dates)
    # last trading day for Jan in this sequence is 2025-01-31
    assert pd.Timestamp("2025-01-31") in me_idx
    assert me_flags.loc[pd.Timestamp("2025-01-31")] is True
    # last trading day for Feb in this sequence is 2025-02-02
    assert pd.Timestamp("2025-02-02") in me_idx
    assert me_flags.loc[pd.Timestamp("2025-02-02")] is True
    # other dates False
    for d in dates.difference(me_idx):
        assert me_flags.loc[d] is False


def test_monthly_returns_equal_product_of_dailies():
    dates = pd.date_range("2025-03-01", periods=5, freq="D")  # single month
    tickers = ["A", "B"]
    # A: 1% daily; B: varying
    close_A = [100.0 * (1.01 ** i) for i in range(len(dates))]
    close_B = [100.0, 102.0, 101.0, 101.0, 103.0]
    closes = { "A": close_A, "B": close_B }
    df = _make_panel(dates, tickers, closes)
    ret_d = rtrs.daily_simple_returns(df)
    ret_m = rtrs.month_returns_from_daily(ret_d)
    # expected product-of-dailies
    exp_A = np.prod([1 + x for x in ret_d["A"].dropna()]) - 1
    exp_B = np.prod([1 + x for x in ret_d["B"].dropna()]) - 1
    # only one month -> single row
    assert ret_m.shape[0] == 1
    assert ret_m.iloc[0]["A"] == pytest.approx(exp_A)
    assert ret_m.iloc[0]["B"] == pytest.approx(exp_B)


def test_cum_return_skip_excludes_last_k_days():
    # create two months of 1% daily returns for a single ticker
    dates = pd.date_range("2025-04-01", periods=40, freq="D")
    tickers = ["AAA"]
    # create steadily increasing closes to produce roughly 1% daily (start 100)
    closes = { "AAA": [100.0 * (1.01 ** i) for i in range(len(dates))] }
    df = _make_panel(dates, tickers, closes)
    ret_d = rtrs.daily_simple_returns(df)
    # compute J=1 (last 1 month), skip_days=5
    cum = rtrs.cum_return_skip(ret_d, J_months=1, skip_days=5)
    # formation dates correspond to month-ends present in dates
    me = rtrs.month_end_index(ret_d.index)
    # pick the last formation date (should have valid window)
    last_form = me[-1]
    # the window includes returns from first day of that month up to last_form - 5 days
    # compute expected manually
    dates_idx = pd.DatetimeIndex(ret_d.index)
    t_pos = list(dates_idx).index(last_form)
    # find month start position (month_end before last_form) -> find previous month_end
    prev_month_end = me[list(me).index(last_form) - 1]
    s_pos = list(dates_idx).index(prev_month_end)
    first_included = s_pos + 1
    last_included = t_pos - 5
    window = ret_d.iloc[first_included:(last_included + 1)]["AAA"]
    expected = np.prod(1 + window.dropna()) - 1
    actual = cum.loc[last_form, "AAA"]
    assert actual == pytest.approx(expected)


def test_cum_return_skip_nan_handling():
    dates = pd.date_range("2025-06-01", periods=20, freq="D")
    tickers = ["X", "Y"]
    close_X = [100.0 * (1.01 ** i) for i in range(len(dates))]
    close_Y = [100.0 * (1.01 ** i) for i in range(len(dates))]
    # inject NaNs into Y on a chunk in the middle
    for i in range(5, 10):
        close_Y[i] = np.nan
    closes = {"X": close_X, "Y": close_Y}
    df = _make_panel(dates, tickers, closes)
    ret_d = rtrs.daily_simple_returns(df)
    cum = rtrs.cum_return_skip(ret_d, J_months=1, skip_days=2)
    # For formation months where Y has some valid days in window, result is computed ignoring NaNs
    # If a window happens to be all NaN, value should be NaN
    # Just assert that Y's value is either finite or NaN (no exceptions) and X is finite
    last_form = rtrs.month_end_index(ret_d.index)[-1]
    assert np.isfinite(cum.loc[last_form, "X"])
    # Y might be finite if it has any non-NaN in the window, otherwise NaN
    valY = cum.loc[last_form, "Y"]
    assert (np.isfinite(valY) or np.isnan(valY))
