"""
Performance metrics utilities for monthly strategy returns.

Functions:
- perf_summary: comprehensive set of performance statistics
- drawdown_stats: detailed drawdown metrics
- turnover_stats: simple turnover calculations from weights matrix

All functions operate on pandas Series / DataFrame with a DateTimeIndex at monthly frequency.
"""
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd


def _annualize_mean(mean_period: float, periods_per_year: int) -> float:
    return mean_period * periods_per_year


def _annualize_vol(std_period: float, periods_per_year: int) -> float:
    return std_period * np.sqrt(periods_per_year)


def _period_rf_from_annual(rf_annual: float, periods_per_year: int) -> float:
    """
    Convert an annual risk-free rate (simple) to equivalent per-period simple rate.
    Uses discrete compounding: (1+rf_annual)^(1/periods_per_year) - 1
    """
    if rf_annual is None:
        return 0.0
    return (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0


def cagr(returns: pd.Series, periods_per_year: int = 12) -> float:
    """
    Compute Compound Annual Growth Rate from periodic returns.
    returns: pd.Series of periodic returns (simple returns, e.g., monthly)
    """
    returns = returns.dropna()
    n = returns.shape[0]
    if n == 0:
        return float("nan")
    total_return = (1.0 + returns).prod()
    years = n / periods_per_year
    if years <= 0:
        return float("nan")
    return total_return ** (1.0 / years) - 1.0


def drawdown_stats(returns: pd.Series) -> Dict[str, Any]:
    """
    Compute drawdown metrics from periodic returns.
    Returns:
      {
        'mdd': float (max drawdown, fraction, e.g., 0.25 for 25%),
        'mdd_start': Timestamp,
        'mdd_end': Timestamp,
        'duration_months': int,
        'wealth_index': pd.Series (1-indexed),
        'drawdowns': pd.Series (per-period drawdown)
      }
    """
    s = returns.dropna()
    if s.empty:
        return {
            "mdd": np.nan,
            "mdd_start": None,
            "mdd_end": None,
            "duration_months": 0,
            "wealth_index": pd.Series(dtype=float),
            "drawdowns": pd.Series(dtype=float),
        }

    wealth = (1.0 + s).cumprod()
    running_max = wealth.cummax()
    drawdowns = (running_max - wealth) / running_max
    if drawdowns.empty:
        return {
            "mdd": 0.0,
            "mdd_start": None,
            "mdd_end": None,
            "duration_months": 0,
            "wealth_index": wealth,
            "drawdowns": drawdowns,
        }

    mdd = drawdowns.max()
    # index where drawdown is maximum (first occurrence)
    mdd_end = drawdowns.idxmax()
    # start is the last date before mdd_end where wealth reached the running max
    # which corresponds to the previous peak
    # find the position
    try:
        peaks = running_max[running_max.index <= mdd_end]
        if peaks.empty:
            mdd_start = running_max.index[0]
        else:
            last_peak_idx = peaks[peaks == peaks.max()].index[-1]
            mdd_start = last_peak_idx
    except Exception:
        mdd_start = running_max.index[0]

    # duration until recovery: find first date after mdd_end where wealth >= previous peak
    recovery_idx = None
    prev_peak_value = running_max.loc[mdd_start]
    for ts in wealth.loc[mdd_end:].index:
        if wealth.loc[ts] >= prev_peak_value:
            recovery_idx = ts
            break
    if recovery_idx is None:
        duration = (returns.index[-1] - mdd_start).days // 30 + 1
    else:
        duration = int((recovery_idx.to_period("M") - pd.Period(mdd_start, freq="M")).n)

    return {
        "mdd": float(mdd),
        "mdd_start": pd.Timestamp(mdd_start),
        "mdd_end": pd.Timestamp(mdd_end),
        "duration_months": int(duration),
        "wealth_index": wealth,
        "drawdowns": drawdowns,
    }


def perf_summary(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    rf: float = 0.0,
    periods_per_year: int = 12,
) -> Dict[str, Any]:
    """
    Compute a performance summary for periodic returns (default monthly).

    Parameters:
      returns: pd.Series of strategy returns (simple returns) indexed by date
      benchmark: optional pd.Series of benchmark returns (simple returns)
      rf: annual risk-free rate (simple). If provided, converted to per-period via discrete compounding.
      periods_per_year: periods per year e.g., 12 for monthly

    Returns: dict with keys:
      'CAGR', 'ann_vol', 'Sharpe', 'IR', 'beta', 'alpha_ann', 'alpha_monthly',
      'maxDD', 'mdd_start', 'mdd_end', 'dd_duration_months', 'VaR_95', 'VaR_99',
      'downside_dev', 'hit_rate', 'n_obs'
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)

    returns = returns.dropna()
    n = returns.shape[0]
    if n == 0:
        return {}

    rf_period = _period_rf_from_annual(rf, periods_per_year)

    mean_period = returns.mean()
    std_period = returns.std(ddof=1)

    mean_ann = _annualize_mean(mean_period, periods_per_year)
    vol_ann = _annualize_vol(std_period, periods_per_year)

    # Sharpe (annualized): use per-period rf
    excess = returns - rf_period
    mean_excess = excess.mean()
    std_excess = excess.std(ddof=1)
    if std_excess == 0 or np.isnan(std_excess):
        sharpe = float("nan")
    else:
        sharpe = (mean_excess * periods_per_year) / (_annualize_vol(std_excess, periods_per_year))

    # Basic results
    cagr_val = cagr(returns, periods_per_year=periods_per_year)

    # Max drawdown
    dd = drawdown_stats(returns)
    max_dd = dd.get("mdd", float("nan"))
    dd_start = dd.get("mdd_start")
    dd_end = dd.get("mdd_end")
    dd_dur = dd.get("duration_months", 0)

    # VaR (historical)
    var_95 = -np.percentile(returns.dropna(), 5)
    var_99 = -np.percentile(returns.dropna(), 1)

    # downside deviation (semi-deviation relative to zero)
    neg_returns = returns[returns < 0]
    if len(neg_returns) == 0:
        downside_dev = 0.0
    else:
        downside_dev = np.sqrt((neg_returns ** 2).mean()) * np.sqrt(periods_per_year)

    # hit rate
    hit_rate = (returns > 0).mean()

    result = {
        "CAGR": float(cagr_val),
        "ann_vol": float(vol_ann),
        "Sharpe": float(sharpe),
        "maxDD": float(max_dd),
        "mdd_start": dd_start,
        "mdd_end": dd_end,
        "dd_duration_months": int(dd_dur),
        "VaR_95": float(var_95),
        "VaR_99": float(var_99),
        "downside_dev": float(downside_dev),
        "hit_rate": float(hit_rate),
        "n_obs": int(n),
    }

    if benchmark is not None:
        if not isinstance(benchmark, pd.Series):
            benchmark = pd.Series(benchmark)

        # align
        aligned = pd.concat([returns, benchmark], axis=1, join="inner")
        aligned.columns = ["strategy", "benchmark"]
        if aligned.shape[0] < 2:
            result.update({"IR": float("nan"), "beta": float("nan"), "alpha_monthly": float("nan"), "alpha_ann": float("nan")})
            return result

        strat = aligned["strategy"]
        bench = aligned["benchmark"]

        excess = strat - bench
        mean_excess_period = excess.mean()
        std_excess_period = excess.std(ddof=1)
        if std_excess_period == 0 or np.isnan(std_excess_period):
            ir = float("nan")
        else:
            # IR as annualized mean excess / annualized std dev of excess
            mean_excess_ann = _annualize_mean(mean_excess_period, periods_per_year)
            std_excess_ann = _annualize_vol(std_excess_period, periods_per_year)
            ir = mean_excess_ann / std_excess_ann

        # Beta by covariance
        cov = np.cov(strat.values, bench.values, ddof=1)
        # cov matrix: [[var_s, cov_sb], [cov_sb, var_b]]
        cov_sb = cov[0, 1]
        var_b = cov[1, 1]
        if var_b == 0 or np.isnan(var_b):
            beta = float("nan")
        else:
            beta = float(cov_sb / var_b)

        # Alpha monthly (simple)
        alpha_m = strat.mean() - beta * bench.mean()
        # Annualize alpha geometrically for reporting
        try:
            alpha_ann = (1.0 + alpha_m) ** periods_per_year - 1.0
        except Exception:
            alpha_ann = alpha_m * periods_per_year

        result.update(
            {
                "IR": float(ir),
                "beta": float(beta),
                "alpha_monthly": float(alpha_m),
                "alpha_ann": float(alpha_ann),
            }
        )

    return result


def turnover_stats(weights: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute average and median turnover from a weights DataFrame indexed by date.
    We expect weights to sum to <= 1 per row. Missing tickers are treated as weight 0.

    turnover_t = 0.5 * sum(|w_t - w_{t-1}|)
    returns mean_turnover, median_turnover, turnover_series
    """
    if not isinstance(weights, pd.DataFrame):
        weights = pd.DataFrame(weights)

    if weights.empty or weights.shape[0] < 2:
        return {"mean_turnover": 0.0, "median_turnover": 0.0, "turnover_series": pd.Series(dtype=float)}

    # ensure sorted by index
    weights_sorted = weights.sort_index().fillna(0.0)
    # reindex columns to union (already aligned)
    deltas = weights_sorted.diff().abs().fillna(0.0)
    turnover_series = 0.5 * deltas.sum(axis=1)
    # drop the first period (diff NaN -> 0)
    turnover_series = turnover_series.iloc[1:]
    mean_turn = float(turnover_series.mean())
    median_turn = float(turnover_series.median())

    return {"mean_turnover": mean_turn, "median_turnover": median_turn, "turnover_series": turnover_series}
