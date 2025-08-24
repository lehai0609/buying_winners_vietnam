"""Trading frictions (costs, slippage, impact) implementation (M6)

Provides:
- apply_costs(trades, adv, fee_bps=25, slip_model="linear", slip_params=None, 
              impact_params=None, participation_cap=None) -> pd.Series

Notes:
- Costs are applied per trade (or per trade day and ticker) and aggregated by date.
- Supports linear slippage model with caps and optional market impact for large orders.
- All costs are non-negative and in VND.
- Uses Vietnam-specific features like ADV-based slippage and participation caps.
"""

from __future__ import annotations

import warnings
import logging
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd


def apply_costs(
    trades: pd.DataFrame,
    adv: pd.DataFrame,
    fee_bps: float = 25.0,
    slip_model: str = "linear",
    slip_params: Optional[Dict] = None,
    impact_params: Optional[Dict] = None,
    participation_cap: Optional[float] = None,
) -> pd.Series:
    """Apply transaction costs, slippage, and market impact to trades.

    Args:
        trades: DataFrame of trades with at least:
            - Index or column for 'date' (trade date)
            - Column for 'ticker' (if not in index)
            - Column for 'notional_vnd' (signed notional in VND, + for buy, - for sell)
            Alternatively, if 'notional_vnd' not provided, must have 'shares' and 'price' columns.
        adv: DataFrame of average daily volume (ADV) in VND. Expected to be:
            - Indexed by [date, ticker] with column 'adv_vnd', or
            - Wide format with dates as index and tickers as columns.
        fee_bps: Transaction fee per side in basis points (default 25 bps).
        slip_model: Slippage model type. Currently only "linear" supported.
        slip_params: Dictionary for slippage parameters:
            - bps_per_1pct_participation: float (default 2.0)
            - cap_bps: float (default 50.0)
        impact_params: Dictionary for market impact parameters:
            - threshold_participation: float (default 0.10)
            - impact_bps: float (default 10.0)
        participation_cap: Optional cap on participation rate (e.g., 0.10 for 10%).
                          If provided, participation is min(actual, cap) for slippage and impact.

    Returns:
        pd.Series of total costs in VND indexed by trade date (daily).

    Raises:
        ValueError: If trades or adv are missing required data.
    """
    # Default parameters
    if slip_params is None:
        slip_params = {}
    if impact_params is None:
        impact_params = {}
    
    bps_per_pct = slip_params.get("bps_per_1pct_participation", 2.0)
    slip_cap_bps = slip_params.get("cap_bps", 50.0)
    impact_threshold = impact_params.get("threshold_participation", 0.10)
    impact_bps_val = impact_params.get("impact_bps", 10.0)

    # Handle empty trades
    if trades.empty:
        return pd.Series([], dtype=float, name='cost_vnd').rename_axis('date')
    
    # Prepare trades DataFrame
    trades_df = trades.copy()
    if 'notional_vnd' not in trades_df.columns:
        # Compute notional from shares and price if not provided
        if 'shares' not in trades_df.columns or 'price' not in trades_df.columns:
            raise ValueError("Trades must contain 'notional_vnd' or both 'shares' and 'price'")
        trades_df['notional_vnd'] = trades_df['shares'] * trades_df['price']
    
    # Ensure we have date and ticker columns for merging
    if 'date' not in trades_df.columns and trades_df.index.names != ['date']:
        raise ValueError("Trades must have 'date' column or index named 'date'")
    if 'ticker' not in trades_df.columns and trades_df.index.names != ['ticker']:
        raise ValueError("Trades must have 'ticker' column or index named 'ticker'")
    
    # Reset index to make 'date' and 'ticker' columns if they are in index
    if trades_df.index.names == ['date', 'ticker']:
        trades_df = trades_df.reset_index()
    elif trades_df.index.names == ['date'] and 'ticker' in trades_df.columns:
        pass  # Already have ticker column
    else:
        # Assume 'date' and 'ticker' are columns
        pass
    
    # Ensure date columns are datetime
    trades_df['date'] = pd.to_datetime(trades_df['date'])

    # Prepare adv DataFrame (allow empty/None -> use NaN ADV so we still charge base fees/slippage cap)
    if adv is None or (isinstance(adv, pd.DataFrame) and adv.empty):
        # Build a minimal ADV frame from trades with NaN values -> triggers slippage cap path
        adv_df = trades_df[['date', 'ticker']].drop_duplicates().copy()
        adv_df['adv_vnd'] = np.nan
    else:
        adv_df = adv.copy()
        if adv_df.index.names == ['date', 'ticker']:
            # MultiIndex with 'adv_vnd' column
            if 'adv_vnd' not in adv_df.columns:
                raise ValueError("ADV DataFrame must have 'adv_vnd' column when indexed by [date, ticker]")
            adv_df = adv_df.reset_index()
        elif (adv_df.index.name == 'date' or isinstance(adv_df.index, pd.DatetimeIndex)) and getattr(adv_df.columns, "nlevels", 1) == 1:
            # Wide format: dates in index, tickers as columns
            adv_df = adv_df.stack().reset_index()
            adv_df.columns = ['date', 'ticker', 'adv_vnd']
        elif set(['date', 'ticker', 'adv_vnd']).issubset(adv_df.columns):
            # Already in long format with required columns, do nothing
            pass
        else:
            raise ValueError("ADV DataFrame must be indexed by [date, ticker] or have dates as index and tickers as columns, or have columns ['date','ticker','adv_vnd']")
    
    # Ensure date columns are datetime
    adv_df['date'] = pd.to_datetime(adv_df['date'])

    # Ensure ADV has unique (date, ticker) pairs - use the latest available ADV
    adv_unique = adv_df.groupby(['date', 'ticker'])['adv_vnd'].last().reset_index()
    
    # Merge trades with ADV
    merged = pd.merge(
        trades_df, 
        adv_unique, 
        on=['date', 'ticker'], 
        how='left'
    )
    
    # Handle missing ADV: set to NaN and log (avoid raising a pytest warnings summary)
    missing_adv = merged['adv_vnd'].isna()
    n_missing = int(missing_adv.sum())
    if n_missing > 0:
        msg = "1 trade has missing ADV; using slippage cap for this" if n_missing == 1 else f"{n_missing} trades have missing ADV; using slippage cap for these"
        # log at DEBUG so normal test runs remain quiet; users can enable logging to see details
        logging.getLogger(__name__).debug(msg)
        merged.loc[missing_adv, 'adv_vnd'] = np.nan

    # Compute participation rate: abs(notional) / ADV
    merged['participation'] = merged.apply(
        lambda row: abs(row['notional_vnd']) / row['adv_vnd'] if pd.notna(row['adv_vnd']) and row['adv_vnd'] > 0 else np.nan,
        axis=1
    )

    # Apply participation cap if specified
    if participation_cap is not None:
        merged['capped_participation'] = merged['participation'].apply(
            lambda p: min(p, participation_cap) if pd.notna(p) else np.nan
        )
    else:
        merged['capped_participation'] = merged['participation']

    # Compute slippage basis points
    if slip_model == "linear":
        merged['slippage_bps'] = merged['capped_participation'].apply(
            lambda p: min(bps_per_pct * (p * 100), slip_cap_bps) if pd.notna(p) else slip_cap_bps
        )
    else:
        raise ValueError(f"Unsupported slippage model: {slip_model}")

    # Compute impact basis points (if participation above threshold)
    merged['impact_bps'] = 0.0
    if impact_params:
        merged['impact_bps'] = merged['capped_participation'].apply(
            lambda p: impact_bps_val if pd.notna(p) and p > impact_threshold else 0.0
        )

    # Total cost basis points: fee + slippage + impact
    merged['total_bps'] = fee_bps + merged['slippage_bps'] + merged['impact_bps']

    # Cost in VND: abs(notional_vnd) * total_bps / 10000
    merged['cost_vnd'] = abs(merged['notional_vnd']) * merged['total_bps'] / 10000.0

    # Aggregate costs by date
    daily_costs = merged.groupby('date')['cost_vnd'].sum()

    return daily_costs


def compute_participation(notional_vnd: float, adv_vnd: float) -> float:
    """Compute participation rate with guards for zero/NaN ADV."""
    if pd.isna(adv_vnd) or adv_vnd <= 0:
        return np.nan
    return abs(notional_vnd) / adv_vnd


def linear_slippage_bps(participation: float, slope_bps_per_pct: float = 2.0, cap_bps: float = 50.0) -> float:
    """Compute linear slippage in basis points."""
    if pd.isna(participation):
        return cap_bps
    return min(slope_bps_per_pct * (participation * 100), cap_bps)


def impact_bps_above_threshold(participation: float, threshold: float = 0.10, impact_bps: float = 10.0) -> float:
    """Compute impact basis points if participation exceeds threshold."""
    if pd.isna(participation) or participation <= threshold:
        return 0.0
    return impact_bps
