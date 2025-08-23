"""Portfolio construction utilities (M5)

Provides:
- build_cohorts(top_mask, K)
- target_weights_from_cohorts(cohorts, formation_months, K, constraints=None, universe_mask=None)
- generate_trades(prev_weights, target_weights, formation_date, trading_dates, t_plus=1, settlement_days=2)

Notes:
- Conforms to long-only, equal-weight-within-cohort, cohort-weight=1/K design.
- Input masks use formation month-end dates as index and tickers as columns.
- Uses formation_months ordering to compute overlapping cohorts (most recent K cohorts active).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd

from .returns import month_end_index


def build_cohorts(top_mask: pd.DataFrame, K: int) -> Dict[pd.Timestamp, pd.Index]:
    """Freeze cohorts at each formation month.

    Args:
        top_mask: DataFrame indexed by formation month-ends (ascending) and columns=tickers.
                  Entries are native Python bools (True for inclusion).
        K: cohort holding length in months (used for validation only).

    Returns:
        dict mapping formation_month -> pd.Index(sorted tickers) of cohort members.
    """
    if top_mask is None:
        raise ValueError("top_mask must be provided")
    if K <= 0:
        raise ValueError("K must be positive integer")

    cohorts: Dict[pd.Timestamp, pd.Index] = {}
    for dt in top_mask.index:
        row = top_mask.loc[dt]
        # treat truthy values as inclusion; guard for missing columns
        members = [t for t, v in row.items() if bool(v)]
        cohorts[pd.Timestamp(dt)] = pd.Index(sorted(members))
    return cohorts


def _active_cohort_starts(formation_months: Sequence[pd.Timestamp], idx: int, K: int) -> List[pd.Timestamp]:
    """Return list of cohort start months active at formation_months[idx]."""
    # take last K formation months ending at idx (inclusive)
    start = max(0, idx - (K - 1))
    return [pd.Timestamp(formation_months[i]) for i in range(start, idx + 1)]


def _equal_weights_for_members(members: Sequence[str], cohort_weight: float) -> Dict[str, float]:
    if len(members) == 0:
        return {}
    per = cohort_weight / float(len(members))
    return {m: per for m in members}


def target_weights_from_cohorts(
    cohorts: Dict[pd.Timestamp, pd.Index],
    formation_months: Sequence[pd.Timestamp],
    K: int,
    constraints: Optional[Dict] = None,
    universe_mask: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build target weights per formation month implementing overlapping K-month cohorts.

    Args:
      cohorts: dict from build_cohorts (cohort start -> tickers)
      formation_months: ordered sequence (ascending) of formation month-ends to produce weights for.
      K: holding length in months (int)
      constraints: optional dict with keys:
        - sum_cap: float <= 1.0 (default 1.0)
        - per_name_cap: Optional[float] (e.g., 0.05)
        - renorm_within_cohort: bool (default True)
        - allow_cash_buffer: bool (default True)
      universe_mask: optional monthly boolean mask indexed by formation_months with same columns as cohorts' tickers.
                     If provided, tickers not eligible at formation date are dropped from cohort at that formation date.

    Returns:
      DataFrame of shape (len(formation_months), all_tickers) dtype float64 with target weights.
      Rows correspond to formation_months (index), columns sorted tickers union.
    """
    if K <= 0:
        raise ValueError("K must be positive integer")
    if constraints is None:
        constraints = {}
    sum_cap = float(constraints.get("sum_cap", 1.0))
    per_name_cap = constraints.get("per_name_cap", None)
    renorm_within_cohort = bool(constraints.get("renorm_within_cohort", True))
    allow_cash_buffer = bool(constraints.get("allow_cash_buffer", True))

    # canonicalize formation months as DatetimeIndex
    formation_months = pd.DatetimeIndex(formation_months)
    # union of all tickers across cohorts
    all_tickers = sorted({t for members in cohorts.values() for t in members})

    # prepare output DataFrame
    weights = pd.DataFrame(0.0, index=formation_months, columns=all_tickers, dtype=float)

    # If universe_mask provided, ensure alignment
    if universe_mask is not None:
        # reindex to formation_months and columns; fill missing and infer types to avoid downcast FutureWarning
        u_mask = universe_mask.reindex(index=formation_months, columns=all_tickers)
        u_mask = u_mask.fillna(False).infer_objects(copy=False).astype(bool)
    else:
        u_mask = None

    # iterate over formation months
    for idx, t in enumerate(formation_months):
        active_starts = _active_cohort_starts(formation_months, idx, K)
        cohort_w = 1.0 / float(K)  # each cohort gets 1/K
        # accumulate per-cohort allocations into a per-ticker dict
        per_ticker_alloc: Dict[str, float] = {}
        for start in active_starts:
            members = list(cohorts.get(start, pd.Index([])))
            # if universe mask provided, drop members not eligible at formation date t
            if u_mask is not None:
                members = [m for m in members if u_mask.at[t, m]]
            if len(members) == 0:
                # cohort weight remains as cash (not reallocated across cohorts)
                continue
            # initial equal-weight in cohort
            raw = _equal_weights_for_members(members, cohort_w)
            if per_name_cap is None:
                # merge into per_ticker_alloc
                for m, w in raw.items():
                    per_ticker_alloc[m] = per_ticker_alloc.get(m, 0.0) + w
            else:
                # enforce per-name cap with optional renormalization within cohort
                cap = float(per_name_cap)
                wvec = raw.copy()
                # iterative redistribution within cohort
                capped = {}
                uncapped = {m: w for m, w in wvec.items() if w <= cap}
                capped_members = {m: w for m, w in wvec.items() if w > cap}
                # cap those above cap
                for m, w in capped_members.items():
                    capped[m] = cap
                residual = sum(w for w in capped_members.values()) - sum(capped.values())
                # Redistribute residual among uncapped proportionally if renorm_within_cohort
                if residual > 0 and renorm_within_cohort and len(uncapped) > 0:
                    total_uncapped = sum(uncapped.values())
                    # distribute residual proportionally to uncapped initial shares
                    for m in list(uncapped.keys()):
                        add = residual * (uncapped[m] / total_uncapped)
                        uncapped[m] += add
                        # if adding pushes above cap, move to capped in next pass
                        if uncapped[m] > cap:
                            # move to capped; will be handled in next redistribution loop
                            capped[m] = cap
                            del uncapped[m]
                    # Note: this simple one-pass redistribution handles common cases; if complex
                    # cascades occur (many hits), we conservatively leave any leftover as cash.
                # Merge capped + uncapped into final_w
                final_w = {}
                for m in wvec.keys():
                    if m in capped:
                        final_w[m] = capped[m]
                    elif m in uncapped:
                        final_w[m] = uncapped[m]
                    else:
                        # if m was moved to capped by redistribution pass
                        if m in capped:
                            final_w[m] = capped[m]
                        else:
                            # fallback to original
                            final_w[m] = min(wvec[m], cap)
                # Any small numerical differences handled downstream
                for m, w in final_w.items():
                    per_ticker_alloc[m] = per_ticker_alloc.get(m, 0.0) + float(w)

        # at this point per_ticker_alloc holds raw weights across active cohorts (sum <= 1.0)
        # enforce sum_cap by scaling down proportionally if needed; otherwise allow cash buffer
        total_alloc = sum(per_ticker_alloc.values())
        if total_alloc > sum_cap:
            scale = sum_cap / total_alloc
            for m in list(per_ticker_alloc.keys()):
                per_ticker_alloc[m] *= scale

        # write into weights DataFrame row
        for m, w in per_ticker_alloc.items():
            # guard: ensure non-negative
            if w < 0:
                w = 0.0
            weights.at[t, m] = float(w)

        # numerical tolerance: zero tiny values
        tiny_mask = weights.loc[t].abs() <= 1e-12
        if tiny_mask.any():
            weights.loc[t, tiny_mask] = 0.0

    # final sanity: ensure no negative, sum per row <= sum_cap + tiny_eps
    weights = weights.clip(lower=0.0)
    row_sums = weights.sum(axis=1)
    too_big = row_sums > (sum_cap + 1e-9)
    if too_big.any():
        # scale down rows slightly to respect sum_cap
        for dt in weights.index[too_big]:
            s = row_sums.loc[dt]
            if s <= 0:
                continue
            weights.loc[dt] = weights.loc[dt] * (sum_cap / float(s))

    return weights.astype(float)


def generate_trades(
    prev_weights: Optional[pd.Series],
    target_weights: pd.Series,
    formation_date: pd.Timestamp,
    trading_dates: pd.DatetimeIndex,
    t_plus: int = 1,
    settlement_days: int = 2,
) -> pd.DataFrame:
    """Generate trade list (delta weights and schedule) between prev_weights and target_weights.

    Args:
      prev_weights: Series indexed by ticker (weights before rebalance). If None, assumed zeros.
      target_weights: Series indexed by ticker (target weights at formation date).
      formation_date: formation month-end (should exist in trading_dates)
      trading_dates: daily trading date index (ascending)
      t_plus: int, execution lag in trading days (default 1 -> next trading day)
      settlement_days: int, settlement days (default 2 -> T+2)

    Returns:
      DataFrame of trades with columns:
        ['formation_date', 'exec_date', 'settle_date', 'ticker', 'prev_w', 'target_w', 'delta_w', 'action']
      exec_date/settle_date are pandas.Timestamp or pd.NaT if beyond available trading_dates.
    """
    # align indices
    tickers = sorted(set(target_weights.index).union(prev_weights.index if prev_weights is not None else []))
    # Build prev weights: if provided, use them (reindexed and filled); otherwise zeros
    if prev_weights is not None:
        prev = prev_weights.reindex(tickers).fillna(0.0).astype(float)
    else:
        prev = pd.Series(0.0, index=tickers, dtype=float)
    targ = target_weights.reindex(tickers).fillna(0.0).astype(float)

    # compute delta
    delta = targ - prev
    # threshold tiny deltas to zero
    delta[np.isclose(delta, 0.0, atol=1e-12)] = 0.0

    # find formation_date position in trading_dates
    td = pd.DatetimeIndex(trading_dates)
    # prefer exact match; otherwise use first trading date > formation_date as anchor
    pos = td.get_indexer([formation_date])[0]
    if pos == -1:
        pos = int(td.searchsorted(formation_date))
        # if searchsorted returns len(td), exec/settle will be NaT later

    exec_pos = pos + t_plus
    settle_pos = exec_pos + settlement_days

    def _pos_to_dt(p: int) -> pd.Timestamp:
        if p is None or p < 0 or p >= len(td):
            return pd.NaT
        return td[p]

    exec_date = _pos_to_dt(exec_pos)
    settle_date = _pos_to_dt(settle_pos)

    records = []
    for t in tickers:
        d = float(delta.at[t])
        if np.isclose(d, 0.0, atol=1e-12):
            action = "HOLD"
        elif d > 0:
            action = "BUY"
        else:
            action = "SELL"
        records.append(
            {
                "formation_date": pd.Timestamp(formation_date),
                "exec_date": exec_date,
                "settle_date": settle_date,
                "ticker": t,
                "prev_w": float(prev.at[t]),
                "target_w": float(targ.at[t]),
                "delta_w": d,
                "action": action,
            }
        )
    trades = pd.DataFrame.from_records(records, columns=["formation_date", "exec_date", "settle_date", "ticker", "prev_w", "target_w", "delta_w", "action"])
    trades = trades.set_index(["formation_date", "ticker"])
    return trades
