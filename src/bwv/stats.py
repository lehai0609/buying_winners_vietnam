"""
Statistical inference utilities: Newey-West alpha, bootstrap CIs, and subperiod reports.

Functions:
- alpha_newey_west: estimate alpha and beta with Newey-West HAC standard errors
- bootstrap_cis: moving block bootstrap for Sharpe or alpha CIs
- subperiod_stats: run perf_summary + alpha_newey_west over supplied windows

Notes:
- Accepts monthly pd.Series inputs. Returns dicts with numeric summaries.
- Tries to use statsmodels when available for HAC; falls back to a numpy implementation otherwise.
"""
from typing import Optional, Dict, Any, Tuple, List

import math
import numpy as np
import pandas as pd

# local import
from .metrics import perf_summary

# try to import statsmodels for parity / reference; optional
try:
    import statsmodels.api as sm  # type: ignore

    _HAS_STATSM = True
except Exception:
    _HAS_STATSM = False

# for p-values
try:
    from scipy import stats as _scipy_stats  # type: ignore

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _normal_cdf(x: float) -> float:
    if _HAS_SCIPY:
        return float(_scipy_stats.norm.cdf(x))
    # fallback using error function
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def alpha_newey_west(
    returns: pd.Series,
    benchmark: pd.Series,
    lags: int = 6,
    rf: float = 0.0,
    periods_per_year: int = 12,
) -> Dict[str, Any]:
    """
    Estimate alpha and beta from monthly returns using OLS with Newey-West HAC standard errors.

    Model: (r_s - rf_period) = alpha + beta * (r_b - rf_period) + eps

    Parameters:
      returns: strategy returns (pd.Series)
      benchmark: benchmark returns (pd.Series)
      lags: NW lag parameter (integer)
      rf: annual risk-free rate (simple). Converted to per-period.
      periods_per_year: for annualization of alpha.

    Returns:
      dict with keys:
        'alpha_monthly', 'alpha_annual', 'beta', 'se_alpha', 't_alpha', 'p_alpha',
        'se_beta', 't_beta', 'p_beta', 'n_obs'
    """
    if not isinstance(returns, pd.Series) or not isinstance(benchmark, pd.Series):
        returns = pd.Series(returns)
        benchmark = pd.Series(benchmark)

    # align series
    df = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    df.columns = ["strategy", "benchmark"]
    n = df.shape[0]
    if n < 2:
        return {
            "alpha_monthly": float("nan"),
            "alpha_annual": float("nan"),
            "beta": float("nan"),
            "se_alpha": float("nan"),
            "t_alpha": float("nan"),
            "p_alpha": float("nan"),
            "se_beta": float("nan"),
            "t_beta": float("nan"),
            "p_beta": float("nan"),
            "n_obs": int(n),
        }

    rf_period = (1.0 + rf) ** (1.0 / periods_per_year) - 1.0 if rf is not None else 0.0

    y = df["strategy"].values - rf_period
    x_raw = (df["benchmark"].values - rf_period).reshape(-1, 1)
    # design matrix with constant
    X = np.column_stack([np.ones(len(y)), x_raw])  # shape (T,2)

    # OLS estimate
    XtX = X.T.dot(X)
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        # ill-conditioned; return NaNs
        return {
            "alpha_monthly": float("nan"),
            "alpha_annual": float("nan"),
            "beta": float("nan"),
            "se_alpha": float("nan"),
            "t_alpha": float("nan"),
            "p_alpha": float("nan"),
            "se_beta": float("nan"),
            "t_beta": float("nan"),
            "p_beta": float("nan"),
            "n_obs": int(n),
        }

    beta_hat = XtX_inv.dot(X.T).dot(y)  # shape (2,)

    # If statsmodels available, use it for HAC standard errors (parity)
    if _HAS_STATSM:
        try:
            mod = sm.OLS(y, X)
            res = mod.fit(cov_type="HAC", cov_kwds={"maxlags": lags})
            params = res.params  # alpha, beta
            bse = res.bse
            tvalues = res.tvalues
            pvalues = res.pvalues
            alpha_m = float(params[0])
            beta = float(params[1])
            se_alpha = float(bse[0])
            se_beta = float(bse[1])
            t_alpha = float(tvalues[0])
            t_beta = float(tvalues[1])
            p_alpha = float(pvalues[0])
            p_beta = float(pvalues[1])
        except Exception:
            # fall back to numpy NW if statsmodels fails unexpectedly
            _HAS_STATSM_local = False
            # continue to numpy NW below
            alpha_m = float(beta_hat[0])
            beta = float(beta_hat[1])
            # compute NW below
            # (we will compute S and V and set se/t/p)
            # compute residuals and continue
            residuals = y - X.dot(beta_hat)
            # compute S matrix
            T, k = X.shape
            L = int(lags)
            S = np.zeros((k, k))
            for lag in range(0, L + 1):
                weight = 1.0 if lag == 0 else 1.0 - lag / (L + 1)
                Gamma = np.zeros((k, k))
                for t in range(lag, T):
                    xt = X[t, :].reshape(k, 1)
                    xt_lag = X[t - lag, :].reshape(k, 1)
                    Gamma += residuals[t] * residuals[t - lag] * (xt.dot(xt_lag.T))
                if lag == 0:
                    S += Gamma
                else:
                    S += weight * (Gamma + Gamma.T)
            V_hat = XtX_inv.dot(S).dot(XtX_inv)
            se = np.sqrt(np.diag(V_hat))
            se_alpha = float(se[0])
            se_beta = float(se[1])
            t_alpha = float(alpha_m / se_alpha) if se_alpha > 0 else float("nan")
            t_beta = float(beta / se_beta) if se_beta > 0 else float("nan")
            p_alpha = 2.0 * (1.0 - _normal_cdf(abs(t_alpha)))
            p_beta = 2.0 * (1.0 - _normal_cdf(abs(t_beta)))
    else:
        # numpy-based Newey-West
        alpha_m = float(beta_hat[0])
        beta = float(beta_hat[1])
        residuals = y - X.dot(beta_hat)
        T, k = X.shape
        L = int(lags)
        S = np.zeros((k, k))
        for lag in range(0, L + 1):
            weight = 1.0 if lag == 0 else 1.0 - lag / (L + 1)
            Gamma = np.zeros((k, k))
            for t in range(lag, T):
                xt = X[t, :].reshape(k, 1)
                xt_lag = X[t - lag, :].reshape(k, 1)
                Gamma += residuals[t] * residuals[t - lag] * (xt.dot(xt_lag.T))
            if lag == 0:
                S += Gamma
            else:
                S += weight * (Gamma + Gamma.T)
        V_hat = XtX_inv.dot(S).dot(XtX_inv)
        # ensure symmetry
        V_hat = 0.5 * (V_hat + V_hat.T)
        se = np.sqrt(np.maximum(np.real(np.diag(V_hat)), 0.0))
        se_alpha = float(se[0])
        se_beta = float(se[1])
        t_alpha = float(alpha_m / se_alpha) if se_alpha > 0 else float("nan")
        t_beta = float(beta / se_beta) if se_beta > 0 else float("nan")
        p_alpha = 2.0 * (1.0 - _normal_cdf(abs(t_alpha)))
        p_beta = 2.0 * (1.0 - _normal_cdf(abs(t_beta)))

    # annualize alpha
    try:
        alpha_ann = (1.0 + alpha_m) ** periods_per_year - 1.0
    except Exception:
        alpha_ann = alpha_m * periods_per_year

    return {
        "alpha_monthly": float(alpha_m),
        "alpha_annual": float(alpha_ann),
        "beta": float(beta),
        "se_alpha": float(se_alpha),
        "t_alpha": float(t_alpha),
        "p_alpha": float(p_alpha),
        "se_beta": float(se_beta),
        "t_beta": float(t_beta),
        "p_beta": float(p_beta),
        "n_obs": int(n),
    }


def _moving_block_bootstrap_indices(T: int, block_size: int, rng: np.random.RandomState) -> np.ndarray:
    """
    Generate an array of length T of indices sampled via moving block bootstrap (with wrap).
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    n_blocks = int(np.ceil(T / block_size))
    starts = rng.randint(0, T, size=n_blocks)
    idx = []
    for s in starts:
        for j in range(block_size):
            idx.append((s + j) % T)
            if len(idx) >= T:
                break
        if len(idx) >= T:
            break
    return np.array(idx[:T], dtype=int)


def bootstrap_cis(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    n: int = 500,
    block_size: int = 6,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    lags: int = 6,
    rf: float = 0.0,
    periods_per_year: int = 12,
) -> Dict[str, Any]:
    """
    Compute bootstrap percentile CIs for either Sharpe (if no benchmark) or alpha (if benchmark provided).

    Returns:
      dict with keys:
        - 'sharpe_ci' -> (lower, upper) if benchmark is None
        - 'alpha_ann_ci' -> (lower, upper) if benchmark provided
        - 'boot_distribution' -> np.ndarray of bootstrap statistics (length n)
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    T = returns.dropna().shape[0]
    if T <= 0:
        return {}

    rng = np.random.RandomState(seed if seed is not None else 0)
    stats = []

    if benchmark is not None:
        # align and dropna
        df = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
        df.columns = ["strategy", "benchmark"]
        T2 = df.shape[0]
        if T2 <= 0:
            return {}
        y_full = df["strategy"].values
        b_full = df["benchmark"].values
        for i in range(int(n)):
            idx = _moving_block_bootstrap_indices(T2, block_size, rng)
            y_bs = pd.Series(y_full[idx]).reset_index(drop=True)
            b_bs = pd.Series(b_full[idx]).reset_index(drop=True)
            # compute alpha on bootstrap sample (use numpy NW path via calling alpha_newey_west)
            try:
                r_y = pd.Series(y_bs)
                r_b = pd.Series(b_bs)
                res = alpha_newey_west(r_y, r_b, lags=lags, rf=rf, periods_per_year=periods_per_year)
                stats.append(res["alpha_annual"])
            except Exception:
                stats.append(float("nan"))
    else:
        # bootstrap Sharpe of returns
        r_full = returns.dropna().values
        T2 = r_full.shape[0]
        for i in range(int(n)):
            idx = _moving_block_bootstrap_indices(T2, block_size, rng)
            r_bs = r_full[idx]
            mean_period = np.mean(r_bs)
            std_period = np.std(r_bs, ddof=1)
            if std_period == 0 or np.isnan(std_period):
                stats.append(float("nan"))
            else:
                sharpe = (mean_period * periods_per_year) / (std_period * math.sqrt(periods_per_year))
                stats.append(sharpe)

    stats_arr = np.array(stats)
    stats_arr = stats_arr[~np.isnan(stats_arr)]
    if stats_arr.size == 0:
        return {}

    lower = float(np.percentile(stats_arr, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(stats_arr, 100.0 * (1.0 - alpha / 2.0)))

    out = {"boot_distribution": stats_arr}
    if benchmark is not None:
        out["alpha_ann_ci"] = (lower, upper)
    else:
        out["sharpe_ci"] = (lower, upper)
    return out


def subperiod_stats(
    returns: pd.Series,
    benchmark: Optional[pd.Series],
    periods: List[Tuple[pd.Timestamp, pd.Timestamp]],
    lags: int = 6,
    rf: float = 0.0,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """
    Compute perf_summary and alpha_newey_west for each period provided.

    periods: list of (start_ts, end_ts) tuples inclusive. Both can be pd.Timestamp or parseable strings.

    Returns: pd.DataFrame indexed by period label with columns for key metrics.
    """
    rows = []
    labels = []
    for start, end in periods:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        mask = (returns.index >= s) & (returns.index <= e)
        r_sub = returns.loc[mask]
        if benchmark is not None:
            b_sub = benchmark.loc[mask]
        else:
            b_sub = None

        pm = perf_summary(r_sub, benchmark=b_sub, rf=rf, periods_per_year=periods_per_year)
        an = {}
        if b_sub is not None and not r_sub.dropna().empty and not b_sub.dropna().empty:
            try:
                an = alpha_newey_west(r_sub, b_sub, lags=lags, rf=rf, periods_per_year=periods_per_year)
            except Exception:
                an = {}
        # flatten results of interest
        row = {
            "start": s,
            "end": e,
            "n_obs": pm.get("n_obs", 0),
            "CAGR": pm.get("CAGR", float("nan")),
            "ann_vol": pm.get("ann_vol", float("nan")),
            "Sharpe": pm.get("Sharpe", float("nan")),
            "IR": pm.get("IR", float("nan")),
            "maxDD": pm.get("maxDD", float("nan")),
            "alpha_monthly": an.get("alpha_monthly", float("nan")),
            "alpha_annual": an.get("alpha_annual", float("nan")),
            "t_alpha": an.get("t_alpha", float("nan")),
            "p_alpha": an.get("p_alpha", float("nan")),
        }
        rows.append(row)
        labels.append(f"{s.date().isoformat()}_{e.date().isoformat()}")

    df_out = pd.DataFrame(rows, index=labels)
    return df_out
