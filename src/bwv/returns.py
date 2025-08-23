"""Returns and calendar utilities for momentum pipeline (M3).

Functions:
- daily_simple_returns(df) -> wide DataFrame (index=date, columns=ticker) of simple returns
- daily_log_returns(df) -> wide DataFrame of log returns
- month_end_index(dates) -> DatetimeIndex of last trading day per calendar month (from provided dates)
- month_end_flags(dates) -> Series[bool] marking month-ends in provided dates
- month_returns_from_daily(ret_d) -> wide DataFrame of monthly returns indexed by month-end dates
- cum_return_skip(ret_d, J_months, skip_days=5) -> wide DataFrame of J-month cumulative returns excluding last skip_days

Notes:
- Inputs expect an OHLCV DataFrame indexed by (date, ticker) with a 'close' column, or a wide daily returns DataFrame (index=date, columns=ticker).
- All aggregation is performed using the provided trading dates (no external calendar); windows are defined on trading-day positions to guarantee no look-ahead.
"""
from __future__ import annotations

from typing import Iterable, List, Optional
import numpy as np
import pandas as pd


def _to_wide_close(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert MultiIndex (date, ticker) DataFrame to wide DataFrame index=date, columns=ticker for 'close' values.
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise AssertionError("Expected MultiIndex [date, ticker] in input DataFrame")
    tmp = df.reset_index()
    wide = tmp.pivot(index="date", columns="ticker", values="close")
    wide = wide.sort_index().sort_index(axis=1)
    return wide


def daily_simple_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple daily returns per ticker as a wide DataFrame (index=date, columns=ticker).

    ret_t = close_t / close_{t-1} - 1
    """
    close = _to_wide_close(df)
    # pct_change computes within each column automatically
    ret = close.pct_change(fill_method=None)
    # keep dtype float64 and preserve index/columns ordering
    ret = ret.astype(float)
    return ret


def daily_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log daily returns (ln close_t - ln close_{t-1}) as wide DataFrame.
    """
    close = _to_wide_close(df)
    # use np.log for numeric stability; result will be NaN where close or prior close is NaN/<=0
    with np.errstate(divide="ignore", invalid="ignore"):
        log_close = np.log(close.astype(float))
    logret = log_close.diff()
    return logret.astype(float)


def month_end_index(dates: Iterable[pd.Timestamp]) -> pd.DatetimeIndex:
    """
    Given an iterable/Index of trading dates (ascending), return the DatetimeIndex of
    the last trading day for each calendar month present in `dates`.
    """
    idx = pd.DatetimeIndex(dates)
    if idx.empty:
        return idx
    # group by period and take max (last trading day)
    last_days = idx.to_series().groupby(idx.to_period("M")).max()
    # ensure ordering and unique
    vals = pd.DatetimeIndex(last_days.sort_values().unique())
    return vals


def month_end_flags(dates: Iterable[pd.Timestamp]) -> pd.Series:
    """
    Return boolean Series indexed by `dates` with True on last trading day of each month.
    The returned Series uses native Python bools (dtype=object) to allow identity
    checks like `series.loc[date,] is True` in tests.
    """
    idx = pd.DatetimeIndex(dates)
    if idx.empty:
        return pd.Series(dtype=object)
    last_days = set(month_end_index(idx))
    # build a list of native Python bools
    data = [bool(d in last_days) for d in idx]
    flags = pd.Series(data, index=idx, dtype=object)
    flags.index.name = None
    return flags


def _nan_safe_prod(series: pd.Series) -> float:
    """
    Compute product(1 + x) - 1 ignoring NaNs. If no non-NaN entries, return np.nan.
    Implemented via exp(sum(log1p(x))) - 1 for numerical stability.
    """
    arr = series.to_numpy(dtype=float)
    if arr.size == 0:
        return np.nan
    # select finite values
    mask = np.isfinite(arr)
    if not mask.any():
        return np.nan
    vals = arr[mask]
    try:
        s = np.sum(np.log1p(vals))
        return float(np.expm1(s))
    except Exception:
        # fallback to multiplicative loop
        prod = 1.0
        for v in vals:
            prod *= (1.0 + v)
        return prod - 1.0


def month_returns_from_daily(ret_d: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate wide daily simple returns into monthly returns indexed by month-end trading dates.

    For each calendar month (based on the provided trading dates), compute:
      monthly_ret = product(1 + ret_d_days) - 1
    The resulting index is the last trading day of each month (as present in ret_d.index).
    """
    if ret_d.empty:
        return pd.DataFrame(index=ret_d.index, columns=ret_d.columns)

    dates = pd.DatetimeIndex(ret_d.index)
    month_ends = month_end_index(dates)
    # We'll iterate months and compute nan-safe products per column
    records = []
    rows = []
    for m_end in month_ends:
        # select dates in same calendar month as m_end
        period = m_end.to_period("M")
        mask = dates.to_period("M") == period
        sub = ret_d.loc[mask]
        # compute per-column nan-safe product
        row = {}
        for col in ret_d.columns:
            row[col] = _nan_safe_prod(sub[col])
        records.append(row)
        rows.append(m_end)
    out = pd.DataFrame.from_records(records, index=pd.DatetimeIndex(rows))
    out.index.name = None
    # ensure column order matches input
    out = out.reindex(columns=ret_d.columns)
    return out.astype(float)


def cum_return_skip(ret_d: pd.DataFrame, J_months: int, skip_days: int = 5) -> pd.DataFrame:
    """
    For each formation month-end t, compute cumulative simple return over the window:
      from (month_end_{i-J_months} exclusive) up to (t - skip_days) inclusive.
    Returns a wide DataFrame indexed by formation month-ends (dates) with columns=tickers.

    Requirements:
      - ret_d: wide daily simple returns with index of trading dates (ascending)
      - J_months: positive integer
      - skip_days: non-negative integer

    Behavior:
      - If there is insufficient history (i.e., month_end_{i-J_months} not available or
        the computed window is empty), the output for that formation date is NaN.
      - All windows are defined purely from available trading dates to avoid look-ahead.
    """
    if J_months <= 0:
        raise ValueError("J_months must be a positive integer")
    if skip_days < 0:
        raise ValueError("skip_days must be non-negative")

    if ret_d.empty:
        return pd.DataFrame(columns=ret_d.columns)

    dates = pd.DatetimeIndex(ret_d.index)
    # mapping date -> position
    pos_map = {d: i for i, d in enumerate(dates)}
    month_ends = month_end_index(dates)
    # need at least J_months prior month-ends to compute for a formation month
    out_rows = []
    out_records = []
    for i, t in enumerate(month_ends):
        # require i - J_months >= 0
        start_idx_in_month_ends = i - J_months
        if start_idx_in_month_ends < 0:
            # Not enough prior month-ends: fall back to start from earliest available date
            # (this allows computing returns for initial months when full history isn't present).
            s_pos = -1
        else:
            start_month = month_ends[start_idx_in_month_ends]
            s_pos = pos_map.get(start_month, None)
        t_pos = pos_map.get(t, None)
        # t_pos must exist; s_pos may be -1 to indicate start from first available date
        if t_pos is None or s_pos is None:
            out_rows.append(t)
            out_records.append({col: np.nan for col in ret_d.columns})
            continue
        # last included date position
        end_pos = t_pos - skip_days
        # the window of returns to include are dates (s_pos+1) .. end_pos inclusive
        first_included = s_pos + 1
        last_included = end_pos
        if last_included < first_included:
            # empty window
            out_rows.append(t)
            out_records.append({col: np.nan for col in ret_d.columns})
            continue
        # slice by iloc
        window = ret_d.iloc[first_included : (last_included + 1)]
        row = {}
        for col in ret_d.columns:
            row[col] = _nan_safe_prod(window[col])
        out_rows.append(t)
        out_records.append(row)
    out = pd.DataFrame.from_records(out_records, index=pd.DatetimeIndex(out_rows))
    out.index.name = None
    out = out.reindex(columns=ret_d.columns)
    return out.astype(float)
