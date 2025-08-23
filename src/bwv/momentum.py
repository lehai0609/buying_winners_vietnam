"""
Momentum signal computation (M4)

Provides:
- momentum_scores(ret_d, universe_mask, J, skip_days=5)
- decile_ranks(scores, q=10)
- top_decile_mask(ranks, decile=9)

Design notes:
- Relies on bwv.returns.cum_return_skip to compute formation-month cumulative returns.
- Deterministic tie-breaking by ticker code (ascending).
- Decile bins assigned via ordinal ranks with floor(r * q / n) mapping (0 = worst, q-1 = best).
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .returns import cum_return_skip, month_end_index


def momentum_scores(ret_d: pd.DataFrame, universe_mask: pd.DataFrame, J: int, skip_days: int = 5) -> pd.DataFrame:
    """
    Compute formation-month momentum scores (J-month cumulative simple returns with skip).

    Parameters
    ----------
    ret_d : pd.DataFrame
        Wide daily simple returns (index = trading dates ascending, columns = tickers).
    universe_mask : pd.DataFrame
        Monthly boolean mask indexed by formation dates (month-ends) with columns = tickers.
        Values should be truthy for eligible tickers on each formation date.
    J : int
        Number of months to look back for momentum.
    skip_days : int
        Number of most recent trading days to skip before the formation date.

    Returns
    -------
    pd.DataFrame
        Monthly momentum scores indexed by formation dates (universe_mask.index) and
        columns aligned to `universe_mask.columns` (plus any extra tickers present in ret_d).
        Ineligible entries (universe_mask is False) are NaN.
    """
    if universe_mask is None:
        raise ValueError("universe_mask must be provided (monthly mask of formation dates)")

    # If ret_d empty, return an appropriately-shaped empty/NaN-filled DataFrame
    if ret_d.empty:
        cols = list(universe_mask.columns) if not universe_mask.empty else list(ret_d.columns)
        return pd.DataFrame(index=universe_mask.index, columns=cols, dtype=float)

    # compute raw J-month cumulative returns using provided utility
    cum = cum_return_skip(ret_d, J, skip_days)

    # target index and columns: align to universe_mask (formation dates) primarily
    target_index = universe_mask.index
    # preserve universe column order, but append any tickers present in cum but missing from universe
    cols = list(universe_mask.columns)
    extra = [c for c in cum.columns if c not in cols]
    cols = cols + extra

    # reindex cumulative returns to target grid (missing entries become NaN)
    cum_reindexed = cum.reindex(index=target_index, columns=cols)

    # reindex mask to same grid, treat missing mask entries as False (ineligible)
    mask = universe_mask.reindex(index=target_index, columns=cols)

    # Build boolean mask: fill missing entries, infer types to avoid object downcast warnings, then cast to native bool
    mask_filled = mask.fillna(False).infer_objects(copy=False)
    bool_mask = mask_filled.astype(bool)

    # Apply mask: set ineligible entries to NaN
    out = cum_reindexed.copy()
    out[~bool_mask] = np.nan

    # Enforce insufficient-history rule: require at least J formation months available overall.
    me = month_end_index(ret_d.index)
    if len(me) < J:
        out.loc[:, :] = np.nan

    # Ensure float dtype for scores
    return out.astype(float)


def _stable_rank_order(scores_row: pd.Series) -> List[str]:
    """
    Return ticker ordering sorted by (-score, ticker) for deterministic tie-breaking.

    This helper is primarily useful for diagnostics or alternative ranking implementations.
    """
    # select finite scores
    try:
        arr = scores_row.to_numpy(dtype=float)
    except Exception:
        # fallback: coerce via pandas
        arr = scores_row.astype(float).to_numpy()
    finite_mask = np.isfinite(arr)
    tickers = list(scores_row.index)
    valid = [t for t, m in zip(tickers, finite_mask) if m]
    if not valid:
        return []
    sub = pd.DataFrame({"score": scores_row.loc[valid]})
    sub["ticker"] = sub.index
    # sort by score descending, ticker ascending (mergesort keeps stability)
    sub_sorted = sub.sort_values(by=["score", "ticker"], ascending=[False, True], kind="mergesort")
    return list(sub_sorted.index)


def decile_ranks(scores: pd.DataFrame, q: int = 10) -> pd.DataFrame:
    """
    Map per-month momentum scores to cross-sectional decile ranks (nullable Int64).

    Rules:
    - Only consider tickers with finite scores (np.isfinite).
    - Sort tickers by score ascending (lowest first) with ticker ascending as secondary key.
      This yields ordinal ranks r=0..n-1 where r=0 is worst and r=n-1 is best.
    - Map ordinal rank r to decile via floor(r * q / n).
    - Return dtype is pandas nullable integer 'Int64' with pd.NA for missing/invalid entries.
    """
    if q <= 0:
        raise ValueError("q must be a positive integer")

    # Prepare empty result with nullable integer dtype
    ranks = pd.DataFrame(index=scores.index, columns=scores.columns, dtype="Int64")

    if scores.empty:
        return ranks

    # Iterate formation dates (rows)
    for dt in scores.index:
        row = scores.loc[dt]
        # coerce to numpy float to check finiteness (this handles NaN/inf)
        try:
            arr = row.to_numpy(dtype=float)
        except Exception:
            arr = row.astype(float).to_numpy()
        finite_mask = np.isfinite(arr)
        tickers = list(row.index)
        valid_tickers = [t for t, m in zip(tickers, finite_mask) if m]
        n = len(valid_tickers)
        if n == 0:
            # leave as NA
            continue
        # Build DataFrame for deterministic sorting
        sub = pd.DataFrame({"score": row.loc[valid_tickers]})
        sub["ticker"] = sub.index
        # Sort ascending so r=0 is worst; tie-breaker by ticker ascending
        sub_sorted = sub.sort_values(by=["score", "ticker"], ascending=[True, True], kind="mergesort")
        # ordinal ranks r = 0..n-1
        r = np.arange(n)
        deciles = np.floor(r * q / n).astype(int)
        # Assign deciles back into result (nullable Int64)
        for ticker, d in zip(sub_sorted.index, deciles):
            ranks.at[dt, ticker] = int(d)

    return ranks


def top_decile_mask(ranks: pd.DataFrame, decile: int = 9) -> pd.DataFrame:
    """
    Return a Python-bool-valued DataFrame mask where True indicates rank == decile.

    The returned DataFrame uses dtype=object and native Python bools for entries so tests
    may rely on identity checks (e.g., `mask.loc[date, t] is True`).
    """
    # empty shortcut
    if ranks.empty:
        return pd.DataFrame(index=ranks.index, columns=ranks.columns, dtype=object)

    # Equality yields boolean (or NA) values; treat NA as False
    eq = (ranks == decile).fillna(False)
    # Convert to nested Python-bool lists to ensure native bools and object dtype
    vals = [[bool(x) for x in row] for row in eq.values.tolist()]
    mask_df = pd.DataFrame(vals, index=ranks.index, columns=ranks.columns, dtype=object)
    return mask_df
