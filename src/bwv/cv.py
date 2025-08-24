"""Cross-validation and robustness utilities for momentum pipeline (M9).

Functions:
- grid_search(ohlcv_df, universe_m, J_list, K_list, config, adv_df=None, benchmark_m=None)
- walk_forward(ohlcv_df, universe_m, J_list, K_list, config, adv_df=None, benchmark_m=None)
- cost_sensitivity(ohlcv_df, universe_m, J, K, cost_bps_list, participation_caps, config, adv_df=None, benchmark_m=None)
- subperiod_robustness(returns_m, benchmark_m, periods, lags=6, rf=0.0)

Notes:
- Uses existing backtest.run_backtest, metrics.perf_summary, and stats.alpha_newey_west/subperiod_stats.
- Returns tidy pandas DataFrames; does not write artifacts (scripts should handle saving/logging).
"""
from __future__ import annotations

import copy
import itertools
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import backtest as bt
from . import metrics as metrics_mod
from . import stats as stats_mod
from . import returns as rtns

logger = logging.getLogger(__name__)


def _make_grid(J_list: Iterable[int], K_list: Iterable[int]) -> List[Tuple[int, int]]:
    return [(int(J), int(K)) for J, K in itertools.product(J_list, K_list)]


def _safe_metrics(returns_m: pd.Series, benchmark_m: Optional[pd.Series], cfg: Dict[str, Any]) -> Dict[str, Any]:
    try:
        pm = metrics_mod.perf_summary(returns_m, benchmark=benchmark_m, rf=cfg.get("rf", 0.0))
    except Exception as e:
        logger.warning("perf_summary failed: %s", e)
        pm = {}
    an = {}
    try:
        if benchmark_m is not None and not returns_m.dropna().empty and not benchmark_m.dropna().empty:
            an = stats_mod.alpha_newey_west(returns_m, benchmark_m, lags=cfg.get("nw_lags", 6), rf=cfg.get("rf", 0.0))
    except Exception as e:
        logger.warning("alpha_newey_west failed: %s", e)
        an = {}
    out = {}
    out.update(pm)
    # include alpha fields with prefix to avoid collisions
    out.update(
        {
            "alpha_monthly": an.get("alpha_monthly", np.nan),
            "alpha_annual": an.get("alpha_annual", np.nan),
            "t_alpha": an.get("t_alpha", np.nan),
            "p_alpha": an.get("p_alpha", np.nan),
        }
    )
    return out


def grid_search(
    ohlcv_df: pd.DataFrame,
    universe_m: pd.DataFrame,
    J_list: Iterable[int],
    K_list: Iterable[int],
    config: Optional[Dict[str, Any]] = None,
    adv_df: Optional[pd.DataFrame] = None,
    benchmark_m: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Run grid search over (J,K).

    Returns a DataFrame with one row per (J,K) containing key metrics.
    """
    if config is None:
        config = {}

    grid = _make_grid(J_list, K_list)
    records = []
    for J, K in grid:
        cfg = copy.deepcopy(config)
        try:
            res = bt.run_backtest(ohlcv_df, universe_m, J=J, K=K, skip_days=cfg.get("skip_days", 5), config=cfg, adv_df=adv_df)
            returns_m = res.get("returns_m", pd.Series(dtype=float))
            # align benchmark to returns_m if provided
            bench_m_aligned = None
            if benchmark_m is not None and not returns_m.empty:
                bench_m_aligned = benchmark_m.reindex(returns_m.index).dropna()
                # if benchmark has fewer dates, align with inner join for stats
                if bench_m_aligned.empty:
                    bench_m_aligned = None
            metrics = _safe_metrics(returns_m, bench_m_aligned, cfg)
            # turnover from weights if available
            turnover = np.nan
            if "weights_d" in res and isinstance(res["weights_d"], pd.DataFrame):
                try:
                    to = metrics_mod.turnover_stats(res["weights_d"])
                    turnover = to.get("mean_turnover", np.nan)
                except Exception:
                    turnover = np.nan
            rec = {
                "J": int(J),
                "K": int(K),
                "n_months": int(metrics.get("n_obs", 0)),
                "CAGR": float(metrics.get("CAGR", np.nan)),
                "Sharpe": float(metrics.get("Sharpe", np.nan)),
                "IR": float(metrics.get("IR", np.nan)) if metrics.get("IR", None) is not None else np.nan,
                "alpha_monthly": float(metrics.get("alpha_monthly", np.nan)),
                "alpha_annual": float(metrics.get("alpha_annual", np.nan)),
                "t_alpha": float(metrics.get("t_alpha", np.nan)),
                "maxDD": float(metrics.get("maxDD", np.nan)),
                "turnover": float(turnover),
            }
        except Exception as e:
            logger.exception("Grid point J=%s K=%s failed: %s", J, K, e)
            rec = {"J": int(J), "K": int(K), "error": str(e)}
        # attach config snapshot keys of interest
        rec["transaction_cost_bps"] = float(config.get("transaction_cost_bps", cfg.get("transaction_cost_bps", np.nan)))
        rec["adv_participation_cap"] = float(config.get("adv_participation_cap", cfg.get("adv_participation_cap", np.nan)))
        records.append(rec)
    df = pd.DataFrame.from_records(records)
    # consistent ordering
    df = df.sort_values(["J", "K"]).reset_index(drop=True)
    return df


def walk_forward(
    ohlcv_df: pd.DataFrame,
    universe_m: pd.DataFrame,
    J_list: Iterable[int],
    K_list: Iterable[int],
    config: Optional[Dict[str, Any]] = None,
    adv_df: Optional[pd.DataFrame] = None,
    benchmark_m: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Perform walk-forward validation.

    Default behavior: sliding windows of train_months and valid_months advanced by `step` months.
    For each fold:
      - evaluate all (J,K) on train (using run_backtest restricted to train period)
      - select best (config['cv']['selection_metric'], default 'Sharpe') on train
      - evaluate selected (J,K) on validation period and record metrics

    Returns a DataFrame with one row per fold per (J,K) evaluated on validation if selected,
    and also returns train metrics to enable stability analysis.
    """
    if config is None:
        config = {}
    cv_cfg = config.get("cv", {})
    train_months = int(cv_cfg.get("train_months", 36))
    valid_months = int(cv_cfg.get("valid_months", 12))
    step = int(cv_cfg.get("step", valid_months))
    selection_metric = cv_cfg.get("selection_metric", "Sharpe")

    # prepare monthly calendar from prices inside ohlcv_df
    # we rely on returns.month_end_index to create formation dates
    dates = pd.DatetimeIndex(sorted(set(ohlcv_df.index.get_level_values("date"))))
    month_ends = rtns.month_end_index(dates)
    # build folds: start positions where we can fit train+valid
    folds = []
    for i in range(0, len(month_ends) - (train_months + valid_months) + 1, step):
        train_start = month_ends[i]
        train_end = month_ends[i + train_months - 1]
        valid_start = month_ends[i + train_months]
        valid_end = month_ends[i + train_months + valid_months - 1]
        folds.append((train_start, train_end, valid_start, valid_end))

    if not folds:
        raise ValueError("Not enough data to form any CV folds with given train/valid lengths")

    rows = []
    grid = _make_grid(J_list, K_list)
    for fold_id, (ts, te, vs, ve) in enumerate(folds):
        logger.info("Walk-forward fold %d: train %s - %s ; valid %s - %s", fold_id, ts, te, vs, ve)
        # restrict ohlcv and universe to the union of train+valid for running backtests
        mask_dates = (ohlcv_df.index.get_level_values("date") >= ts) & (ohlcv_df.index.get_level_values("date") <= ve)
        ohlcv_sub = ohlcv_df.loc[mask_dates]
        # restrict monthly universe to formation dates within train+valid
        universe_sub = universe_m.loc[(universe_m.index >= ts) & (universe_m.index <= ve)]
        # For each (J,K) compute train metrics and valid metrics by slicing returns_m
        train_metrics = {}
        valid_metrics = {}
        train_scores = []
        for J, K in grid:
            try:
                res = bt.run_backtest(ohlcv_sub, universe_sub, J=J, K=K, skip_days=config.get("skip_days", 5), config=config, adv_df=adv_df)
                returns_m = res.get("returns_m", pd.Series(dtype=float))
                # slice train / valid portions by date
                returns_m_train = returns_m[(returns_m.index >= ts) & (returns_m.index <= te)]
                returns_m_valid = returns_m[(returns_m.index >= vs) & (returns_m.index <= ve)]
                bench_train = benchmark_m.reindex(returns_m_train.index) if benchmark_m is not None else None
                bench_valid = benchmark_m.reindex(returns_m_valid.index) if benchmark_m is not None else None
                train_perf = _safe_metrics(returns_m_train, bench_train, config)
                valid_perf = _safe_metrics(returns_m_valid, bench_valid, config)
                train_metrics[(J, K)] = train_perf
                valid_metrics[(J, K)] = valid_perf
                # extract selection metric (fallback to Sharpe)
                score = train_perf.get(selection_metric, train_perf.get("Sharpe", np.nan))
                train_scores.append(((J, K), float(score if score is not None else np.nan)))
            except Exception as e:
                logger.exception("Fold %s evaluation failed for J=%s K=%s : %s", fold_id, J, K, e)
                train_metrics[(J, K)] = {}
                valid_metrics[(J, K)] = {}
                train_scores.append(((J, K), np.nan))

        # choose best by selection metric (highest)
        train_scores_sorted = sorted(train_scores, key=lambda x: (-np.nan_to_num(x[1], nan=-np.inf), x[0]))
        best_pair = train_scores_sorted[0][0]
        # record summary for each (J,K)
        for (J, K), _ in train_scores:
            tr = train_metrics.get((J, K), {})
            va = valid_metrics.get((J, K), {})
            selected = (J, K) == best_pair
            row = {
                "fold_id": int(fold_id),
                "train_start": pd.Timestamp(ts),
                "train_end": pd.Timestamp(te),
                "valid_start": pd.Timestamp(vs),
                "valid_end": pd.Timestamp(ve),
                "J": int(J),
                "K": int(K),
                "selected": bool(selected),
                "train_Sharpe": float(tr.get("Sharpe", np.nan)),
                "valid_Sharpe": float(va.get("Sharpe", np.nan)),
                "train_alpha_monthly": float(tr.get("alpha_monthly", np.nan)),
                "valid_alpha_monthly": float(va.get("alpha_monthly", np.nan)),
                "train_n": int(tr.get("n_obs", 0)),
                "valid_n": int(va.get("n_obs", 0)),
            }
            rows.append(row)

    df = pd.DataFrame.from_records(rows)
    return df


def cost_sensitivity(
    ohlcv_df: pd.DataFrame,
    universe_m: pd.DataFrame,
    J: int,
    K: int,
    cost_bps_list: Iterable[float],
    participation_caps: Iterable[float],
    config: Optional[Dict[str, Any]] = None,
    adv_df: Optional[pd.DataFrame] = None,
    benchmark_m: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Sweep transaction_cost_bps and adv participation caps for fixed (J,K)."""
    if config is None:
        config = {}
    records = []
    for cost_bps, part_cap in itertools.product(cost_bps_list, participation_caps):
        cfg = copy.deepcopy(config)
        cfg["transaction_cost_bps"] = float(cost_bps)
        cfg["adv_participation_cap"] = float(part_cap)
        try:
            res = bt.run_backtest(ohlcv_df, universe_m, J=J, K=K, skip_days=cfg.get("skip_days", 5), config=cfg, adv_df=adv_df)
            returns_m = res.get("returns_m", pd.Series(dtype=float))
            bench = benchmark_m.reindex(returns_m.index) if (benchmark_m is not None and not returns_m.empty) else None
            metrics = _safe_metrics(returns_m, bench, cfg)
            turnover = np.nan
            if "weights_d" in res and isinstance(res["weights_d"], pd.DataFrame):
                try:
                    to = metrics_mod.turnover_stats(res["weights_d"])
                    turnover = to.get("mean_turnover", np.nan)
                except Exception:
                    turnover = np.nan
            rec = {
                "J": int(J),
                "K": int(K),
                "transaction_cost_bps": float(cost_bps),
                "adv_participation_cap": float(part_cap),
                "CAGR": float(metrics.get("CAGR", np.nan)),
                "Sharpe": float(metrics.get("Sharpe", np.nan)),
                "IR": float(metrics.get("IR", np.nan)) if metrics.get("IR", None) is not None else np.nan,
                "alpha_monthly": float(metrics.get("alpha_monthly", np.nan)),
                "t_alpha": float(metrics.get("t_alpha", np.nan)),
                "turnover": float(turnover),
            }
        except Exception as e:
            logger.exception("Cost sensitivity run failed for cost=%s part=%s: %s", cost_bps, part_cap, e)
            rec = {"J": int(J), "K": int(K), "transaction_cost_bps": float(cost_bps), "adv_participation_cap": float(part_cap), "error": str(e)}
        records.append(rec)
    df = pd.DataFrame.from_records(records)
    return df


def subperiod_robustness(
    returns_m: pd.Series,
    benchmark_m: Optional[pd.Series],
    periods: Iterable[Tuple[Any, Any]],
    lags: int = 6,
    rf: float = 0.0,
) -> pd.DataFrame:
    """Compute subperiod stats (perf + alpha NW) for provided (start,end) windows.

    `periods` is an iterable of (start, end) where start/end are parseable by pd.Timestamp.
    """
    df = stats_mod.subperiod_stats(returns_m, benchmark_m, periods=list(periods), lags=lags, rf=rf)
    return df
