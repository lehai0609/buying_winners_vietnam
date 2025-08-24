#!/usr/bin/env python
"""
Run a J x K grid search for the momentum backtest and optionally produce a report.

Examples:
  poetry run python scripts/grid_search.py --J-list 3 6 9 12 --K-list 1 3 6 12 --out results/runs/grid
  poetry run python scripts/grid_search.py --J-list 6 12 --K-list 1 3 --benchmark index.csv --out results/runs/grid --report --selection-metric Sharpe

Inputs (defaults to cleaned artifacts from scripts/make_clean.py):
  --ohlcv data/clean/ohlcv.parquet
  --universe data/clean/universe_monthly.csv

Optional:
  --benchmark index.csv        # daily index series (with 'close' or 'ret' column); converted to monthly returns
  --adv data/clean/adv.parquet # ADV by date x ticker in VND (optional; used for costs/slippage models)

Outputs:
  - <out>/results.pkl      # pickled results dict containing {'grid_results': DataFrame, 'metrics': {...}}
  - <out>/results.json     # minimal JSON summary with best combo by selection metric
  - <out>/manifest.json    # created file index
  - if --report: <out>/artifacts/...  # figures/tables/report (heatmap will use first numeric col e.g. Sharpe/CAGR)

Notes:
  - Imports from the installed 'bwv' package; falls back to adding ./src to sys.path if needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence

import pandas as pd

# Try to import from installed package; fallback to local src
try:
    from bwv import cv, report as report_mod  # type: ignore
except Exception:
    sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
    from bwv import cv, report as report_mod  # type: ignore


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


def _load_benchmark_monthly(path: Optional[Path]) -> Optional[pd.Series]:
    """Load a daily index file and return monthly simple returns Series (end-of-month)."""
    if path is None:
        return None
    if not path.exists():
        print(f"[WARN] Benchmark file not found: {path}")
        return None
    df = pd.read_csv(path)
    # try to find date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    else:
        # if no 'date', try first column
        first = df.columns[0]
        df[first] = pd.to_datetime(df[first])
        df = df.set_index(first).sort_index()
    # prefer explicit return column if present
    for col in ["ret", "return", "r", "excess_return", "excess"]:
        if col in df.columns:
            ser = pd.to_numeric(df[col], errors="coerce").dropna()
            ser_m = (1.0 + ser).groupby(ser.index.to_period("M")).prod() - 1.0
            ser_m.index = ser_m.index.to_timestamp("M")
            return ser_m
    # else try close column -> compute monthly pct_change of month-end close
    for col in ["close", "Close", "CLOSE", "px_close", "PX_CLOSE"]:
        if col in df.columns:
            s_close = pd.to_numeric(df[col], errors="coerce").dropna()
            m_close = s_close.groupby(s_close.index.to_period("M")).last()
            ret_m = m_close.pct_change().dropna()
            ret_m.index = ret_m.index.to_timestamp("M")
            return ret_m
    print("[WARN] Could not parse benchmark file (needs 'ret' or 'close' column); skipping benchmark.")
    return None


def _parse_list_int(vals: Sequence[int]) -> list[int]:
    return [int(v) for v in vals]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a J x K grid search for the momentum backtest.")
    # Inputs
    p.add_argument("--ohlcv", type=str, default="data/clean/ohlcv.parquet", help="Clean OHLCV (parquet or csv)")
    p.add_argument("--universe", type=str, default="data/clean/universe_monthly.csv", help="Monthly universe CSV")
    p.add_argument("--benchmark", type=str, default=None, help="Optional benchmark CSV (daily close or daily returns)")
    p.add_argument("--adv", type=str, default=None, help="Optional ADV wide table (parquet/csv) in VND for costs")
    # Grid
    p.add_argument("--J-list", type=int, nargs="+", required=True, help="List of J (months) values")
    p.add_argument("--K-list", type=int, nargs="+", required=True, help="List of K (months) values")
    p.add_argument("--skip-days", type=int, default=5, help="Skip days before formation to avoid reversal")
    # Backtest config (costs/execution)
    p.add_argument("--transaction-cost-bps", type=float, default=25.0, help="Per-side fee in bps")
    p.add_argument("--adv-participation-cap", type=float, default=0.10, help="Max ADV participation (fraction)")
    p.add_argument("--price-limit-cap", type=float, default=0.07, help="Daily return cap for price limits (fraction)")
    # Inference config
    p.add_argument("--rf", type=float, default=0.0, help="Risk-free rate (monthly when used in metrics)")
    p.add_argument("--nw-lags", type=int, default=6, help="Newey-West lags for monthly alpha")
    p.add_argument("--selection-metric", type=str, default="Sharpe", help="Metric to rank grid (e.g., Sharpe, CAGR, alpha_annual)")
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
    bench_m = _load_benchmark_monthly(bench_path)

    J_list = _parse_list_int(args.J_list)
    K_list = _parse_list_int(args.K_list)

    cfg = {
        "skip_days": int(args.skip_days),
        "transaction_cost_bps": float(args.transaction_cost_bps),
        "adv_participation_cap": float(args.adv_participation_cap),
        "price_limit_cap": float(args.price_limit_cap),
        "rf": float(args.rf),
        "nw_lags": int(args.nw_lags),
    }

    print(f"[INFO] Grid search J={J_list} x K={K_list} skip_days={args.skip_days}")
    grid_df = cv.grid_search(
        ohlcv_df=ohlcv,
        universe_m=universe_m,
        J_list=J_list,
        K_list=K_list,
        config=cfg,
        adv_df=adv_df,
        benchmark_m=bench_m,
    )

    # Determine best by selection metric (fallback to Sharpe)
    sel = args.selection_metric
    if sel not in grid_df.columns:
        print(f"[WARN] Selection metric '{sel}' not found in grid results; falling back to 'Sharpe'.")
        sel = "Sharpe" if "Sharpe" in grid_df.columns else None

    best = {}
    if sel is not None:
        try:
            s = pd.to_numeric(grid_df[sel], errors="coerce")
            idx = s.idxmax() if not s.dropna().empty else None
            if idx is not None:
                row = grid_df.loc[idx]
                best = {
                    "J": int(row.get("J", -1)),
                    "K": int(row.get("K", -1)),
                    "selection_metric": sel,
                    "selection_value": float(row.get(sel, float("nan"))),
                }
        except Exception:
            best = {}
    else:
        print("[WARN] No numeric selection metric available to choose best configuration.")

    results = {
        "grid_results": grid_df,
        "metrics": {"selection_metric": sel, "best": best},
    }

    # Persist artifacts
    pkl_path = out_dir / "results.pkl"
    json_path = out_dir / "results.json"
    pd.to_pickle(results, pkl_path)

    minimal = {"selection_metric": sel, "best": best}
    json_path.write_text(json.dumps(minimal, indent=2, default=str), encoding="utf-8")

    manifest = {"created": {"results_pkl": str(pkl_path), "results_json": str(json_path)}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[OK] Saved: {pkl_path}")
    print(f"[OK] Saved: {json_path}")

    if args.report:
        title = args.title or f"Momentum Grid Search ({sel} heatmap)"
        cfg_rep = {"html": {"title": title}}
        created = report_mod.generate_report(results, cfg_rep, str(out_dir))
        print("[OK] Report artifacts created:")
        for k, v in created.items():
            print(f"  - {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
