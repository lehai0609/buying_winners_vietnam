#!/usr/bin/env python
"""
Run a single (J, K) momentum backtest and optionally produce a report.

Examples:
  poetry run python scripts/run_backtest.py --J 12 --K 3 --skip-days 5 --out results/runs/J12K3
  poetry run python scripts/run_backtest.py --J 12 --K 3 --skip-days 5 --benchmark index.csv --report --out results/runs/J12K3

Inputs (defaults to cleaned artifacts from scripts/make_clean.py):
  --ohlcv data/clean/ohlcv.parquet
  --universe data/clean/universe_monthly.csv

Optional:
  --benchmark index.csv        # daily index series (with 'close' or 'ret' column); will be converted to monthly returns
  --adv data/clean/adv.parquet # ADV by date x ticker in VND (optional; used for costs/slippage models)

Outputs:
  - <out>/results.pkl      # pickled results dict consumable by scripts/make_report.py
  - <out>/results.json     # minimal JSON (metrics/stats only)
  - <out>/manifest.json    # created file index
  - if --report: <out>/artifacts/...  # figures/tables/report

Notes:
  - Imports from the installed 'bwv' package; falls back to adding ./src to sys.path if needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Tuple
import re

import pandas as pd

# Try to import from installed package; fallback to local src
try:
    from bwv import backtest, metrics, stats, report as report_mod  # type: ignore
except Exception:
    sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
    from bwv import backtest, metrics, stats, report as report_mod  # type: ignore


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_ohlcv(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    # if parquet missing or suffix csv, try csv
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(path, parse_dates=["date"]).set_index(["date", "ticker"]).sort_index()


def _load_universe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index)
    # ensure boolean dtype
    return df.astype(bool)


def _load_adv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    if not path.exists():
        print(f"[WARN] ADV file not found: {path}")
        return None
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _load_benchmark_monthly(path: Optional[Path], ticker: Optional[str] = None) -> Optional[pd.Series]:
    """Load a daily index file and return monthly simple returns Series (end-of-month).

    Robust to vendor-style headers like <Ticker>, <DTYYYYMMDD>, <Close>.
    If multiple tickers exist, selects the one provided via `ticker`.
    """
    if path is None:
        return None
    if not path.exists():
        print(f"[WARN] Benchmark file not found: {path}")
        return None

    df = pd.read_csv(path)

    # Normalize column names: strip angle brackets/quotes/whitespace and lowercase
    def _norm(col: str) -> str:
        return re.sub(r'^[<\s"]*|[>\s"]*$', "", str(col)).strip().lower()

    df.columns = [_norm(c) for c in df.columns]

    # If a ticker column exists and a desired ticker is specified, filter
    if "ticker" in df.columns:
        if ticker:
            df = df[df["ticker"].astype(str).str.upper() == str(ticker).upper()]
        else:
            # If only one ticker present, keep it; else prefer VNINDEX if present, else first
            uniq = sorted(df["ticker"].astype(str).str.upper().unique().tolist())
            pick = "VNINDEX" if "VNINDEX" in uniq else uniq[0]
            df = df[df["ticker"].astype(str).str.upper() == pick]

    # Identify and parse a date column
    date_idx = None
    if "date" in df.columns:
        date_idx = pd.to_datetime(df["date"], errors="coerce")
    elif "dtyyyymmdd" in df.columns:
        date_idx = pd.to_datetime(df["dtyyyymmdd"].astype(str), format="%Y%m%d", errors="coerce")
    elif "yyyymmdd" in df.columns:
        date_idx = pd.to_datetime(df["yyyymmdd"].astype(str), format="%Y%m%d", errors="coerce")
    else:
        # Try any column name that contains 'date'
        cand = next((c for c in df.columns if "date" in c), None)
        if cand is not None:
            date_idx = pd.to_datetime(df[cand], errors="coerce")

    if date_idx is None:
        print("[WARN] Could not find/parse a date column in benchmark file; skipping benchmark.")
        return None

    df = df.assign(_date=date_idx).dropna(subset=["_date"]).set_index("_date").sort_index()

    # Prefer explicit daily return column if present
    for col in ["ret", "return", "r", "excess_return", "excess"]:
        if col in df.columns:
            ser = pd.to_numeric(df[col], errors="coerce").dropna()
            ser_m = (1.0 + ser).groupby(ser.index.to_period("M")).prod() - 1.0
            ser_m.index = ser_m.index.to_timestamp("M")
            return ser_m

    # Else compute monthly returns from close
    for col in ["close", "px_close"]:
        if col in df.columns:
            s_close = pd.to_numeric(df[col], errors="coerce").dropna()
            m_close = s_close.groupby(s_close.index.to_period("M")).last()
            ret_m = m_close.pct_change().dropna()
            ret_m.index = ret_m.index.to_timestamp("M")
            return ret_m

    print("[WARN] Could not parse benchmark file (needs 'ret' or 'close' column); skipping benchmark.")
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a single (J, K) momentum backtest and optionally produce a report.")
    # Inputs
    p.add_argument("--ohlcv", type=str, default="data/clean/ohlcv.parquet", help="Clean OHLCV (parquet or csv)")
    p.add_argument("--universe", type=str, default="data/clean/universe_monthly.csv", help="Monthly universe CSV")
    p.add_argument("--benchmark", type=str, default=None, help="Optional benchmark CSV (daily close or daily returns)")
    p.add_argument(
        "--benchmark-ticker",
        type=str,
        default="VNINDEX",
        help="If benchmark CSV has multiple tickers, select this ticker (e.g., VNINDEX, HNX-INDEX)",
    )
    p.add_argument("--adv", type=str, default=None, help="Optional ADV wide table (parquet/csv) in VND for costs")
    # Strategy
    p.add_argument("--J", type=int, required=True, help="Formation window (months)")
    p.add_argument("--K", type=int, required=True, help="Holding window (months, overlapping)")
    p.add_argument("--skip-days", type=int, default=5, help="Skip days before formation to avoid reversal")
    # Backtest config (costs/execution)
    p.add_argument("--initial-capital", type=float, default=1_000_000_000.0, help="Initial capital in VND")
    p.add_argument("--transaction-cost-bps", type=float, default=25.0, help="Per-side fee in bps")
    p.add_argument("--adv-participation-cap", type=float, default=0.10, help="Max ADV participation (fraction)")
    p.add_argument("--price-limit-cap", type=float, default=0.07, help="Daily return cap for price limits (fraction)")
    p.add_argument("--execution-price", type=str, default="next_open", choices=["next_open"], help="Execution price model")
    # Inference config
    p.add_argument("--rf", type=float, default=0.0, help="Risk-free rate (monthly when used in metrics)")
    p.add_argument("--nw-lags", type=int, default=6, help="Newey-West lags for monthly alpha")
    # Output
    p.add_argument("--out", type=str, required=True, help="Output directory for run artifacts")
    p.add_argument("--title", type=str, default=None, help="Optional report title")
    p.add_argument("--report", action="store_true", help="Also render a report under out/artifacts")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    out_dir = Path(args.out)
    _ensure_dir(out_dir)

    # Load inputs
    ohlcv_path = Path(args.ohlcv)
    uni_path = Path(args.universe)
    adv_path = Path(args.adv) if args.adv else None
    bench_path = Path(args.benchmark) if args.benchmark else None

    ohlcv = _load_ohlcv(ohlcv_path)
    universe_m = _load_universe(uni_path)
    adv_df = _load_adv(adv_path)
    bench_m = _load_benchmark_monthly(bench_path, ticker=args.benchmark_ticker)

    # Backtest config dict
    cfg = {
        "initial_capital": float(args.initial_capital),
        "transaction_cost_bps": float(args.transaction_cost_bps),
        "slippage": {"bps_per_1pct_participation": 2.0, "cap_bps": 50.0},
        "impact": {"threshold_participation": float(args.adv_participation_cap), "impact_bps": 10.0},
        "adv_participation_cap": float(args.adv_participation_cap),
        "execution_price": args.execution_price,
        "price_limit_cap": float(args.price_limit_cap),
        "rf": float(args.rf),
        "nw_lags": int(args.nw_lags),
    }

    # Run backtest
    print(f"[INFO] Running backtest J={args.J} K={args.K} skip_days={args.skip_days}")
    res = backtest.run_backtest(
        ohlcv_df=ohlcv,
        universe_m=universe_m,
        J=int(args.J),
        K=int(args.K),
        skip_days=int(args.skip_days),
        config=cfg,
        adv_df=adv_df,
    )

    # Derive metrics and stats
    returns_m = res.get("returns_m", pd.Series(dtype=float))
    bench_aligned = None
    if bench_m is not None and not returns_m.empty:
        bench_aligned = bench_m.reindex(returns_m.index).dropna()
        if bench_aligned.empty:
            bench_aligned = None

    perf = metrics.perf_summary(returns_m, benchmark=bench_aligned, rf=cfg["rf"]) if not returns_m.empty else {}
    nw = stats.alpha_newey_west(returns_m, bench_aligned, lags=cfg["nw_lags"], rf=cfg["rf"]) if (bench_aligned is not None and not returns_m.empty) else {}

    # Build results dict for reporting
    equity_curve_d = res.get("equity_curve_d")
    if equity_curve_d is not None and isinstance(equity_curve_d, pd.Series):
        equity_norm = equity_curve_d / float(args.initial_capital)
    else:
        # fallback from daily returns
        equity_norm = (1.0 + res.get("returns_d", pd.Series(dtype=float))).cumprod()

    results = {
        "equity_curve": equity_norm,
        "monthly_returns": returns_m,
        "benchmark_returns": bench_aligned,
        "metrics": perf,
        "stats": nw,
        # Optional raw artifacts
        "weights": res.get("weights_d"),
        "trades": res.get("trades"),
        "costs": res.get("costs_d"),
    }

    # Persist artifacts
    pkl_path = out_dir / "results.pkl"
    json_path = out_dir / "results.json"
    pd.to_pickle(results, pkl_path)

    # Minimal JSON for quick inspection (convertible fields only)
    minimal = {"J": int(args.J), "K": int(args.K), "skip_days": int(args.skip_days), "metrics": perf, "stats": nw}
    json_path.write_text(json.dumps(minimal, indent=2, default=str), encoding="utf-8")

    manifest = {"created": {"results_pkl": str(pkl_path), "results_json": str(json_path)}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[OK] Saved: {pkl_path}")
    print(f"[OK] Saved: {json_path}")

    if args.report:
        cfg_rep = {"html": {"title": args.title or f"Momentum Backtest J={args.J} K={args.K}"}}
        created = report_mod.generate_report(results, cfg_rep, str(out_dir))
        print("[OK] Report artifacts created:")
        for k, v in created.items():
            print(f"  - {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
