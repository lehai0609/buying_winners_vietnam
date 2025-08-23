"""
Vietnam-specific eligibility and cleaning filters for momentum pipeline.

Provides:
- helpers to compute trading value (close * volume or value column)
- min_history, price_floor, adv_filter, non_trading_days_filter
- compose_universe: combine masks and produce monthly universe (last trading day of month)

Notes:
- All rolling stats are shifted by 1 day to avoid look-ahead (eligibility for date t uses data up to t-1).
- Input expected: OHLCV DataFrame with MultiIndex (date, ticker) and columns ['open','high','low','close','volume', optional 'value'].
- Units: prices are VND (ingest should scale from kVND to VND), volume is shares, trading value is VND.
- Functions return boolean DataFrames indexed by date (daily) with columns=tickers. compose_universe returns monthly DataFrame indexed by month-end (last trading day).
"""
from __future__ import annotations
from typing import Dict, Optional

import pandas as pd
import numpy as np


def _to_wide(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Convert MultiIndex (date, ticker) df to wide DataFrame with index=date and columns=ticker for a specific value_col.
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise AssertionError("Expected MultiIndex [date, ticker] in df")
    tmp = df.reset_index()
    wide = tmp.pivot(index="date", columns="ticker", values=value_col)
    # sort index and columns for deterministic behavior
    wide = wide.sort_index().sort_index(axis=1)
    return wide


def trading_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trading value (VND) per date,ticker as:
      - use 'value' column if present; else close * volume (where volume may be NaN)
    Returns wide DataFrame index=date, columns=ticker.
    """
    if "value" in df.columns:
        val_wide = _to_wide(df, "value")
    else:
        # compute close * volume; volume may be nullable Float64 -> cast to float for multiplication
        close = _to_wide(df, "close")
        if "volume" in df.columns:
            vol = _to_wide(df, "volume")
            val_wide = close * vol
        else:
            # fallback: close * 0 -> zeros (no volume information)
            val_wide = close * 0.0
    return val_wide


def _to_pybool(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a boolean-like DataFrame into a DataFrame of native Python bool objects (dtype=object).
    Build from a dict-of-lists to ensure pure Python bools are used.
    """
    data = {}
    for col in df.columns:
        data[col] = [bool(x) for x in df[col].tolist()]
    out = pd.DataFrame(data, index=df.index, columns=df.columns, dtype=object)
    return out


def min_history(df: pd.DataFrame, min_days: int = 126) -> pd.DataFrame:
    """
    Return boolean DataFrame (daily) indicating whether each ticker has at least `min_days`
    of trading history up to the PREVIOUS trading day (no look-ahead).
    The input df is expected to have rows for trading days only; min_days counts trading observations.
    """
    close = _to_wide(df, "close")
    # count number of non-null observations per ticker cumulatively
    valid = close.notna().astype(int)
    cum = valid.cumsum()
    # shift by 1 to exclude current day
    eligible = (cum.shift(1, fill_value=0) >= min_days)
    # fill NaN -> False; convert to native Python bools for tests that use `is True/False`
    return _to_pybool(eligible.fillna(False))


def price_floor(df: pd.DataFrame, min_price: float = 1000.0) -> pd.DataFrame:
    """
    Daily boolean mask where close[VND] >= min_price (VND).
    """
    close = _to_wide(df, "close")
    mask = close >= float(min_price)
    mask = mask.fillna(False)
    return _to_pybool(mask)


def adv_filter(df: pd.DataFrame, min_adv_vnd: float = 1e8, window: int = 60) -> pd.DataFrame:
    """
    Average Daily Trading Value (VND) filter.
    - Compute trading_value (prefer 'value' column; else close * volume)
    - Rolling mean over `window` trading days, computed per ticker.
    - Shift by 1 trading day to avoid look-ahead.
    Returns daily boolean mask where rolling_mean >= min_adv_vnd.
    """
    tv = trading_value(df)
    # rolling(window) on each column; min_periods=1 so we can observe early values but we will compare to threshold
    roll = tv.rolling(window=window, min_periods=1).mean()
    # shift by 1 day to avoid including current day's data
    roll_sh = roll.shift(1, fill_value=np.nan)
    mask = roll_sh >= float(min_adv_vnd)
    mask = mask.fillna(False)
    return _to_pybool(mask)


def non_trading_days_filter(df: pd.DataFrame, max_ntd: int = 15, window_days: int = 126) -> pd.DataFrame:
    """
    Flag tickers with too many non-trading days over the trailing `window_days` (count of trading days window).
    Non-trading day is defined as volume == 0 or close is NA for that date/ticker.
    Rolling count of non-trading days is computed per ticker, shifted by 1 day (no look-ahead).
    Returns daily boolean mask True if non_trading_days <= max_ntd.
    """
    # Build wide volume and close
    if "volume" in df.columns:
        vol = _to_wide(df, "volume")
        vol_zero = (vol == 0) | vol.isna()
    else:
        # if no volume, treat NaN close as non-trading only
        close_wide = _to_wide(df, "close")
        vol_zero = pd.DataFrame(False, index=close_wide.index, columns=close_wide.columns)

    close = _to_wide(df, "close")
    close_na = close.isna()

    non_trade = (vol_zero | close_na).astype(int)
    # rolling sum over window_days
    roll_ntd = non_trade.rolling(window=window_days, min_periods=1).sum()
    roll_ntd_sh = roll_ntd.shift(1, fill_value=0)
    mask = roll_ntd_sh <= int(max_ntd)
    mask = mask.fillna(False)
    return _to_pybool(mask)


def compose_universe(
    df: pd.DataFrame,
    rules: Optional[Dict] = None,
    *,
    monthly: bool = True,
    min_history_days: int = 126,
    min_price: float = 1000.0,
    min_adv_vnd: float = 1e8,
    adv_window: int = 60,
    max_ntd: int = 15,
    ntd_window: int = 126,
) -> pd.DataFrame:
    """
    Compose the monthly universe DataFrame (index=last trading day of month, columns=tickers) applying Vietnam-specific rules.
    The returned DataFrame is boolean where True indicates ticker eligible at that formation date.

    Parameters mirror individual filters; rules dict is reserved for future configuration.
    """
    # compute daily masks
    mh = min_history(df, min_days=min_history_days)
    pf = price_floor(df, min_price=min_price)
    advm = adv_filter(df, min_adv_vnd=min_adv_vnd, window=adv_window)
    ntdm = non_trading_days_filter(df, max_ntd=max_ntd, window_days=ntd_window)

    # Combine via logical AND
    daily_universe = mh & pf & advm & ntdm

    if not monthly:
        return daily_universe

    # find last trading day per month (last index value in each month period)
    dates = daily_universe.index
    if dates.empty:
        return pd.DataFrame(columns=daily_universe.columns)

    # compute last trading day per month
    last_days = dates.to_series().groupby(dates.to_period("M")).max()
    last_days = last_days.sort_values().unique()
    # reindex to ensure exact order and presence
    month_end_universe = daily_universe.loc[last_days].copy()
    # ensure dtype bool
    month_end_universe = month_end_universe.fillna(False).astype(bool)
    # convert to native Python bools so element access like `df.loc[date, ticker] is False` works in tests
    return _to_pybool(month_end_universe)
