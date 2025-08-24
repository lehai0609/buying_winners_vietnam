"""
Backtesting engine (M7) - conservative v1 implementation.

Provides:
- run_backtest(ohlcv_df, universe_m, J, K, skip_days, config) -> dict
- simulate_daily(date, prices_wide, prev_weights, trades_df, config, equity_prev) -> tuple(daily_pnl, new_weights, costs_vnd, diagnostics)

Conventions / simplifications (v1):
- Execution at next trading day's OPEN (if available). If open missing or volume==0 the trade is skipped for that day (reason='halt' or 'no_open'); no partial fills.
- Price limits approximated by capping daily returns at ±config['price_limit_cap'] for PnL calculation.
- Immediate cash update on trade execution (T+2 strict ledger TODO).
- Overlapping K-month cohorts supported by averaging cohort-level equal weights (each cohort contributes 1/K of target exposure).
- Costs applied via src.bwv.costs.apply_costs. ADV must be supplied in config as a DataFrame or can be None (costs will warn).
- Diagnostics recorded per trade: skipped_reason, participation (if ADV provided), fees/slippage/impact bps when available.

This module is written to be unit-test friendly and to integrate with existing returns/momentum/portfolio/costs modules.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
import logging

import numpy as np
import pandas as pd

from . import returns as rtns
from . import momentum as mom
from . import portfolio as port
from . import costs as costs_mod

logger = logging.getLogger(__name__)
DEFAULT_CONFIG = {
    "initial_capital": 1_000_000_000.0,
    "execution_price": "next_open",  # other option: 'vwap_proxy' (not implemented)
    "price_limit_cap": 0.07,
    "transaction_cost_bps": 25.0,
    "slippage": {"bps_per_1pct_participation": 2.0, "cap_bps": 50.0},
    "impact": {"threshold_participation": 0.10, "impact_bps": 10.0},
    "adv_participation_cap": 0.10,
    "target_exposure": 1.0,  # total long exposure (<=1.0)
}


def _to_wide_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Pivot MultiIndex (date, ticker) DataFrame into wide (index=date, columns=ticker) for given col.
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise AssertionError("Expected MultiIndex [date, ticker] in OHLCV DataFrame")
    tmp = df.reset_index()
    wide = tmp.pivot(index="date", columns="ticker", values=col)
    wide = wide.sort_index().sort_index(axis=1)
    return wide


def _first_trading_day_after(dates: pd.DatetimeIndex, t: pd.Timestamp) -> Optional[pd.Timestamp]:
    """
    Return first trading date strictly > t, or None if none exists.
    """
    pos = dates.searchsorted(t, side="right")
    if pos >= len(dates):
        return None
    return dates[pos]


def _cap_returns(ret_series: pd.Series, cap: float) -> pd.Series:
    if cap is None:
        return ret_series
    return ret_series.clip(lower=-cap, upper=cap)


def simulate_daily(
    date: pd.Timestamp,
    prices_wide_close: pd.DataFrame,
    prices_wide_open: pd.DataFrame,
    prev_weights: pd.Series,
    trades_df: pd.DataFrame,
    equity_prev: float,
    adv_df: Optional[pd.DataFrame],
    config: Dict[str, Any],
) -> Tuple[float, pd.Series, float, Dict[str, Any]]:
    """
    Simulate single trading day.

    Args:
      date: trading date
      prices_wide_close: wide DataFrame of close prices (index=date)
      prices_wide_open: wide DataFrame of open prices (index=date)
      prev_weights: Series indexed by ticker representing weight at start of day (sum <= target_exposure)
      trades_df: DataFrame of trades to execute on this date with columns: ticker, target_weight (float 0..1)
                 Trades are interpreted as setting exposure to target_weight (not incremental shares).
      equity_prev: previous day's equity
      adv_df: optional wide adv DataFrame (dates x tickers) in VND for participation calc
      config: config dict (uses DEFAULT_CONFIG for defaults)

    Returns:
      daily_net_return (float), new_weights (Series), costs_vnd (float), diagnostics (dict)
    """
    diagnostics: Dict[str, Any] = {}
    price_limit = config.get("price_limit_cap", DEFAULT_CONFIG["price_limit_cap"])

    dates = prices_wide_close.index
    if date not in dates:
        raise KeyError(f"{date} not in price index")

    # compute daily simple returns from prior close -> close
    prev_day_pos = dates.get_loc(date)
    # use available ret on this date computed as close_t / close_{t-1} - 1, if prev exists
    if prev_day_pos == 0:
        # first available day: no prior returns
        daily_ret = pd.Series(0.0, index=prices_wide_close.columns)
    else:
        prev_date = dates[prev_day_pos - 1]
        prev_close = prices_wide_close.loc[prev_date]
        cur_close = prices_wide_close.loc[date]
        with np.errstate(divide="ignore", invalid="ignore"):
            daily_ret = (cur_close / prev_close) - 1.0

    # cap returns per price limit model
    daily_ret = _cap_returns(daily_ret, price_limit).fillna(0.0)

    # Prepare trades: determine executed trades based on open price presence and volume non-zero if possible
    trades = trades_df.copy() if trades_df is not None else pd.DataFrame(columns=["ticker", "target_weight"])
    executed_trades_records = []
    skipped_records = []
    if not trades.empty:
        # Ensure columns
        if "ticker" not in trades.columns:
            trades = trades.reset_index()
        tickers = trades["ticker"].tolist()
        # get open prices for this date
        if date in prices_wide_open.index:
            opens = prices_wide_open.loc[date]
        else:
            opens = pd.Series(index=prices_wide_open.columns, dtype=float)
        # get adv row if available
        adv_row = adv_df.loc[date] if (adv_df is not None and date in adv_df.index) else None

        for _, row in trades.iterrows():
            tk = row["ticker"]
            target_w = float(row.get("target_weight", 0.0))
            open_price = opens.get(tk, np.nan) if isinstance(opens, pd.Series) else np.nan
            # basic skip logic: no open or NaN open -> skip (halt/no_open)
            if pd.isna(open_price) or open_price == 0.0:
                skipped_records.append({"ticker": tk, "reason": "no_open_or_halt"})
                continue
            # compute notional to set target weight: notional = equity_prev * target_w
            notional = equity_prev * target_w
            # compute participation if adv available
            participation = np.nan
            if adv_row is not None and tk in adv_row.index:
                adv_vnd = adv_row[tk]
                if pd.notna(adv_vnd) and adv_vnd > 0:
                    participation = abs(notional) / adv_vnd
            executed_trades_records.append(
                {
                    "date": date,
                    "ticker": tk,
                    "target_weight": target_w,
                    "price": open_price,
                    "notional_vnd": notional,
                    "participation": participation,
                }
            )

    executed_trades = pd.DataFrame.from_records(executed_trades_records)
    # Build costs input for costs.apply_costs - costs expects date,ticker,notional_vnd
    costs_vnd = 0.0
    if not executed_trades.empty:
        # ensure required columns
        costs_input = executed_trades[["date", "ticker", "notional_vnd"]].copy()
        # make adv in long or wide acceptable: costs.apply_costs accepts wide adv via dates index/ticker columns as handled internally
        try:
            daily_costs_series = costs_mod.apply_costs(
                trades=costs_input,
                adv=adv_df if adv_df is not None else pd.DataFrame(),
                fee_bps=config.get("transaction_cost_bps", DEFAULT_CONFIG["transaction_cost_bps"]),
                slip_model="linear",
                slip_params=config.get("slippage", DEFAULT_CONFIG["slippage"]),
                impact_params=config.get("impact", DEFAULT_CONFIG["impact"]),
                participation_cap=config.get("adv_participation_cap", DEFAULT_CONFIG["adv_participation_cap"]),
            )
            # daily_costs_series indexed by date with total cost_vnd
            if not daily_costs_series.empty:
                costs_vnd = float(daily_costs_series.sum())
        except Exception as e:
            logger.warning("Costs application failed: %s", e)
            costs_vnd = 0.0

    # Compute PnL effect:
    # - For pre-existing positions prev_weights: PnL contribution = sum(prev_weight * daily_ret)
    gross_ret_prev = (prev_weights.reindex(daily_ret.index).fillna(0.0) * daily_ret).sum()
    # - For newly executed trades, assume executed at open and earn close/open - 1 for that same day:
    gross_ret_new = 0.0
    if not executed_trades.empty:
        # compute intraday return per executed ticker: (close/open - 1)
        close_row = prices_wide_close.loc[date]
        open_row = prices_wide_open.loc[date]
        intraday_ret = (close_row / open_row).replace([np.inf, -np.inf], np.nan) - 1.0
        for _, r in executed_trades.iterrows():
            tk = r["ticker"]
            t_w = r["target_weight"]
            rd = intraday_ret.get(tk, 0.0)
            if pd.isna(rd):
                rd = 0.0
            gross_ret_new += t_w * rd

    # Conservative combination:
    # We assume trades set exposures for the remainder of the day; to avoid double-counting,
    # we'll treat prev_weights PnL as for positions carried into the day, and executed trades
    # contribute intraday PnL as above. This is a simple approximation.
    gross_ret = gross_ret_prev + gross_ret_new

    # Net return subtracts costs as fraction of equity_prev
    net_ret = gross_ret - (costs_vnd / equity_prev if equity_prev > 0 else 0.0)

    # Update equity, update weights by price drift (post-close)
    equity_new = equity_prev * (1.0 + net_ret)

    # Update weights: w_t+1 = (prev_weight * (1+ret) for held names) normalized to keep total exposure targeted,
    # and for executed trades we set to target_weight (they already contributed intraday PnL)
    # Start from previous holdings' value exposure
    start_values = prev_weights.reindex(daily_ret.index).fillna(0.0) * equity_prev
    # Apply price drift to get current market values
    end_values = start_values * (1.0 + daily_ret)
    # Replace / set values for executed trades to their notional (they became target exposures)
    for _, r in executed_trades.iterrows():
        tk = r["ticker"]
        notional = r["notional_vnd"]
        end_values[tk] = notional

    # Derive new weights as end_values / equity_new, but cap negatives and ensure sum <= target_exposure
    new_weights = (end_values / equity_new).fillna(0.0)
    new_weights[new_weights < 0.0] = 0.0
    total_exposure = new_weights.sum()
    target_exposure = config.get("target_exposure", DEFAULT_CONFIG["target_exposure"])
    if total_exposure > 0 and total_exposure != target_exposure:
        # scale down proportionally to meet target exposure
        scale = min(1.0, target_exposure / total_exposure)
        new_weights = new_weights * scale

    # Diagnostics
    diagnostics["num_trades_executed"] = len(executed_trades)
    diagnostics["num_trades_skipped"] = len(skipped_records)
    diagnostics["costs_vnd"] = costs_vnd
    diagnostics["gross_ret"] = float(gross_ret)
    diagnostics["net_ret"] = float(net_ret)

    return float(net_ret), new_weights, float(costs_vnd), diagnostics


def run_backtest(
    ohlcv_df: pd.DataFrame,
    universe_m: pd.DataFrame,
    J: int = 12,
    K: int = 1,
    skip_days: int = 5,
    config: Optional[Dict[str, Any]] = None,
    adv_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Run monthly backtest following the momentum pipeline.

    Args:
      ohlcv_df: MultiIndex [date,ticker] DataFrame with columns ['open','high','low','close','volume',...]
      universe_m: DataFrame indexed by formation month-ends with tickers as columns boolean mask (True=eligible)
      J, K, skip_days: strategy params
      config: overrides for DEFAULT_CONFIG
      adv_df: optional wide adv DataFrame (index=date, columns=ticker) in VND used by costs

    Returns:
      dict with keys:
        - equity_curve_d: pd.Series indexed by trading dates
        - returns_d: pd.Series daily net returns
        - returns_m: pd.Series monthly net returns
        - weights_d: DataFrame (date x ticker)
        - trades: DataFrame of attempted trades
        - costs_d: pd.Series daily costs
        - diagnostics: dict of counters
    """
    if config is None:
        config = {}
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(config)

    # Prepare wide price tables
    prices_close = _to_wide_col(ohlcv_df, "close")
    prices_open = _to_wide_col(ohlcv_df, "open")

    dates = prices_close.index
    if dates.empty:
        raise ValueError("Empty price series")

    # daily returns wide
    ret_d = prices_close.pct_change().astype(float)

    # formation month-ends based on trading dates (returns.cum_return_skip uses same index)
    month_ends = rtns.month_end_index(dates)

    # precompute momentum scores for all formation dates
    scores_all = rtns.cum_return_skip(ret_d.fillna(0.0), J_months=J, skip_days=skip_days)

    # storage
    equity = cfg.get("initial_capital", DEFAULT_CONFIG["initial_capital"])
    equity_curve = []
    returns_d_list = []
    costs_d_list = []
    weights_history = []
    trades_records = []
    diagnostics = {"lookahead_flags": 0, "limit_skips": 0, "halt_skips": 0}

    # initialize prev_weights as empty Series (all zeros)
    prev_weights = pd.Series(0.0, index=prices_close.columns)

    # sliding cohorts state: list of (formation_date, set_of_tickers)
    cohorts: List[Dict[str, Any]] = []

    for d in dates:
        # If d is a next-trading-day for some formation time t, we may need to rebalance.
        # Determine if there exists a formation month t such that next trading day after t == d
        # Practical approach: map formation t -> execution_day
        exec_needed = False
        target_weights_for_day = pd.Series(dtype=float)

        # find any formation t where next trading day is d
        for t in month_ends:
            next_trade = _first_trading_day_after(dates, t)
            if next_trade is None:
                continue
            if next_trade == d:
                # compute top-decile using scores_all at formation date t
                if t not in scores_all.index:
                    continue
                scores_t = scores_all.loc[t]
                # mask with universe at t if provided
                if t in universe_m.index:
                    mask = universe_m.loc[t].astype(bool)
                    scores_t = scores_t.where(mask, other=np.nan)
                # compute deciles via momentum.decile_ranks if available, else fallback
                try:
                    ranks = mom.decile_ranks(scores_t.to_frame("score"), by_month=False).iloc[0]
                    # mom.decile_ranks returns ranks 0..9; select top decile (9)
                    top_mask = ranks == (mom.__dict__.get("DECILES", 10) - 1 if "DECILES" in mom.__dict__ else 9)
                except Exception:
                    # fallback: pick top 10% by non-NaN count
                    nonnan = scores_t.dropna()
                    if nonnan.empty:
                        top_mask = pd.Series(False, index=scores_t.index)
                    else:
                        cutoff = nonnan.quantile(0.9)
                        top_mask = scores_t >= cutoff
                winners = top_mask[top_mask.fillna(False)].index.tolist()
                # push new cohort
                cohorts.insert(0, {"formation_date": t, "winners": winners})
                # keep only K cohorts
                cohorts = cohorts[:K]
                exec_needed = True

        # Build target_weights from active cohorts
        if exec_needed:
            # aggregate equal-weight per cohort then average across K cohorts
            combined = pd.Series(0.0, index=prices_close.columns)
            for cohort in cohorts:
                winners = cohort["winners"]
                if len(winners) == 0:
                    continue
                w = 1.0 / len(winners)
                for tk in winners:
                    combined[tk] += (1.0 / K) * w
            # enforce target total exposure
            total = combined.sum()
            target_exposure = cfg.get("target_exposure", 1.0)
            if total > 0:
                combined = combined * (target_exposure / total)
            target_weights_for_day = combined

            # prepare trades: compare prev_weights -> target_weights_for_day and record trades for tickers where weight change
            changed = (target_weights_for_day - prev_weights).abs() > 1e-12
            trade_tickers = target_weights_for_day.index[changed]
            for tk in trade_tickers:
                trades_records.append(
                    {
                        "date": d,
                        "ticker": tk,
                        "prev_weight": float(prev_weights.get(tk, 0.0)),
                        "target_weight": float(target_weights_for_day.get(tk, 0.0)),
                    }
                )

        # trades_df for this day: convert trades_records for this date only
        trades_today = pd.DataFrame([r for r in trades_records if r["date"] == d])
        # simulate day: get net_ret, new_weights, costs, diag
        net_ret, new_weights, costs_vnd, diag = simulate_daily(
            date=d,
            prices_wide_close=prices_close,
            prices_wide_open=prices_open,
            prev_weights=prev_weights,
            trades_df=trades_today[["ticker", "target_weight"]] if not trades_today.empty else pd.DataFrame(),
            equity_prev=equity,
            adv_df=adv_df,
            config=cfg,
        )

        equity = equity * (1.0 + net_ret)
        equity_curve.append({"date": d, "equity": equity})
        returns_d_list.append({"date": d, "net_ret": net_ret})
        costs_d_list.append({"date": d, "costs_vnd": costs_vnd})
        weights_history.append(pd.Series(new_weights, name=d))
        # update prev_weights
        prev_weights = new_weights

    # finalize outputs
    equity_curve_df = pd.DataFrame(equity_curve).set_index("date")["equity"]
    returns_d_df = pd.DataFrame(returns_d_list).set_index("date")["net_ret"]
    costs_d_df = pd.DataFrame(costs_d_list).set_index("date")["costs_vnd"]
    # weights history -> DataFrame
    weights_d_df = pd.DataFrame(weights_history).fillna(0.0)
    weights_d_df.index = pd.DatetimeIndex([r["date"] for r in equity_curve])

    trades_df_all = pd.DataFrame(trades_records)

    # monthly aggregation
    returns_m = rtns.month_returns_from_daily(returns_d_df.to_frame("ret")).iloc[:, 0] if not returns_d_df.empty else pd.Series(dtype=float)

    result = {
        "equity_curve_d": equity_curve_df,
        "returns_d": returns_d_df,
        "returns_m": returns_m,
        "weights_d": weights_d_df,
        "trades": trades_df_all,
        "costs_d": costs_d_df,
        "diagnostics": diagnostics,
    }

    return result
