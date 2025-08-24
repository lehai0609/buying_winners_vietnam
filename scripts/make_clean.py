#!/usr/bin/env python3
"""
CLI to load HSX/HNX CSVs and/or per-ticker parquet directories, scale prices from kVND to VND on ingest,
validate, and emit:
- data/clean/ohlcv.parquet (VND prices)
- data/clean/universe_monthly.csv (boolean matrix [month_end × ticker])
- optionally: data/clean/indices_monthly.csv (monthly returns for selected indices)

Defaults are read from config/data.yml. You can override inputs/outputs via CLI flags.

Usage:
  poetry run python scripts/make_clean.py
  poetry run python scripts/make_clean.py --config config/data.yml
  poetry run python scripts/make_clean.py --inputs HSX.csv HNX.csv --out-parquet data/clean/ohlcv.parquet --out-universe data/clean/universe_monthly.csv
  poetry run python scripts/make_clean.py --inputs-dir HSX HNX --out-parquet data/clean/ohlcv.parquet --out-universe data/clean/universe_monthly.csv
  poetry run python scripts/make_clean.py --inputs HSX.csv HNX.csv --inputs-dir HSX HNX
  poetry run python scripts/make_clean.py --indices-dir vn_indices --indices VNINDEX HNX-INDEX --out-indices-monthly data/clean/indices_monthly.csv

Notes:
- CSV headers are normalized (strip angle brackets, symbol->ticker, dtYYYYMMDD->date).
- Prices in vendor files are assumed kVND; they are scaled to VND internally via price_scale (default 1000.0).
- Universe is sampled at the last trading day of each month (no look-ahead; all rolling stats are shifted by 1 day).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import yaml

# Import using repository layout (matching tests)
from bwv.data_io import load_ohlcv, validate_ohlcv, load_ohlcv_parquet_dirs, load_indices_from_dir
from bwv import filters
from bwv.returns import month_returns_from_daily


def read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_universe_csv(universe_m: pd.DataFrame, out_path: str | Path) -> None:
    outp = Path(out_path)
    ensure_parent(outp)
    # Write with ISO dates in index
    universe_m.to_csv(outp, index=True, date_format="%Y-%m-%d")


def _try_write_parquet(df: pd.DataFrame, out_path: str | Path) -> None:
    outp = Path(out_path)
    ensure_parent(outp)
    try:
        df.to_parquet(outp, index=True)
    except Exception as e:
        # Fallback to CSV if parquet engine is missing
        alt = outp.with_suffix(".csv")
        df.to_csv(alt)
        print(f"[WARN] Failed to write parquet at {outp} ({e}); wrote CSV fallback at {alt}", file=sys.stderr)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate cleaned OHLCV parquet and monthly eligibility universe CSV.")
    p.add_argument("--config", type=str, default="config/data.yml", help="YAML config path with defaults")
    p.add_argument("--inputs", type=str, nargs="*", help="Input CSV paths (default from config)")
    p.add_argument("--out-parquet", type=str, help="Output parquet path for OHLCV (default from config)")
    p.add_argument("--out-universe", type=str, help="Output CSV path for monthly universe (default from config)")
    # Eligibility knobs (override config defaults)
    p.add_argument("--price-scale", type=float, help="Price scaling factor (default from config)")
    p.add_argument("--min-history-days", type=int, help="Minimum trading history days (default from config)")
    p.add_argument("--min-price-vnd", type=float, help="Price floor in VND (default from config)")
    p.add_argument("--min-adv-vnd", type=float, help="Minimum ADV (VND) (default from config)")
    p.add_argument("--adv-window-days", type=int, help="ADV rolling window (trading days) (default from config)")
    p.add_argument("--max-non-trading-days", type=int, help="Max non-trading days allowed (default from config)")
    p.add_argument("--ntd-window-days", type=int, help="Non-trading days rolling window (default from config)")
    # New ingestion/indices options
    p.add_argument("--inputs-dir", type=str, nargs="*", help="Input directories of per-ticker parquet files (e.g., HSX HNX)")
    p.add_argument("--indices-dir", type=str, help="Directory of index CSVs (e.g., vn_indices)")
    p.add_argument("--indices", type=str, nargs="*", help="Index tickers to include (e.g., VNINDEX HNX-INDEX VN30)")
    p.add_argument("--out-indices-monthly", type=str, help="Output CSV path for indices monthly simple returns")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg_path = Path(args.config)
    cfg = read_config(cfg_path) if cfg_path.exists() else {}

    # Inputs (CSV and/or parquet directories)
    cfg_inputs_csv = cfg.get("inputs") or []
    cfg_inputs_dirs = cfg.get("inputs_parquet_dirs") or []
    inputs_dir: List[str] = args.inputs_dir if args.inputs_dir is not None else cfg_inputs_dirs

    # Rule:
    # - If user explicitly passes --inputs on CLI, honor it.
    # - Else if user passes --inputs-dir on CLI, DO NOT implicitly include CSV inputs from config to avoid accidental duplication.
    # - Else (no CLI overrides), fall back to config CSV inputs.
    if args.inputs is not None:
        inputs: List[str] = args.inputs
    else:
        inputs = [] if args.inputs_dir is not None else cfg_inputs_csv

    if not inputs and not inputs_dir:
        print(
            "No inputs provided. Specify CSV via --inputs HSX.csv HNX.csv or parquet dirs via --inputs-dir HSX HNX "
            "or set them in config/data.yml.",
            file=sys.stderr,
        )
        return 2

    # If both families are provided explicitly via CLI, warn that overlap may occur (we will deduplicate later).
    if (args.inputs is not None) and (args.inputs_dir is not None):
        print(
            "[INFO] Both CSV inputs and parquet dirs provided on CLI; overlapping (date,ticker) pairs will be deduplicated, preferring parquet.",
            file=sys.stderr,
        )

    # Outputs
    outputs_cfg = cfg.get("outputs") or {}
    out_parquet = Path(args.out_parquet or outputs_cfg.get("ohlcv_parquet", "data/clean/ohlcv.parquet"))
    out_universe = Path(args.out_universe or outputs_cfg.get("universe_monthly_csv", "data/clean/universe_monthly.csv"))
    out_indices_monthly = Path(args.out_indices_monthly or outputs_cfg.get("indices_monthly_csv", "data/clean/indices_monthly.csv"))

    # Eligibility/config
    price_scale = args.price_scale if args.price_scale is not None else float(cfg.get("price_scale", 1000.0))
    min_history_days = args.min_history_days if args.min_history_days is not None else int(cfg.get("min_history_days", 126))
    min_price_vnd = args.min_price_vnd if args.min_price_vnd is not None else float(cfg.get("min_price_vnd", 1000.0))
    min_adv_vnd = args.min_adv_vnd if args.min_adv_vnd is not None else float(cfg.get("min_adv_vnd", 100_000_000.0))
    adv_window_days = args.adv_window_days if args.adv_window_days is not None else int(cfg.get("adv_window_days", 60))
    max_non_trading_days = args.max_non_trading_days if args.max_non_trading_days is not None else int(cfg.get("max_non_trading_days", 15))
    ntd_window_days = args.ntd_window_days if args.ntd_window_days is not None else int(cfg.get("ntd_window_days", 126))

    # Load and scale to VND (support mixed CSV + parquet directories)
    parts: list[pd.DataFrame] = []
    if inputs:
        parts.append(load_ohlcv(inputs, price_scale=price_scale))
    if inputs_dir:
        parts.append(load_ohlcv_parquet_dirs(inputs_dir, price_scale=price_scale))

    # Concatenate parts (if both provided)
    df_raw = parts[0] if len(parts) == 1 else pd.concat(parts).sort_index()

    # If both CSV and parquet are present, report potential overlaps before pre-clean.
    # Downstream dedup keeps the last occurrence per (date,ticker) after a stable sort;
    # since we append parquet after CSV, parquet rows are preferred on overlap.
    if len(parts) > 1:
        gr = df_raw.reset_index().groupby(["date", "ticker"]).size()
        dup_pairs = int((gr > 1).sum())
        if dup_pairs:
            print(
                f"[INFO] Found {dup_pairs} overlapping (date,ticker) pairs across sources; preferring parquet where overlap.",
                file=sys.stderr,
            )

    df = df_raw

    # Pre-clean: drop non-equity index series and invalid OHLC rows before strict validation
    df = df.reset_index()
    # Drop ticker rows that look like indices (e.g., VNINDEX, VN30, VNALL-INDEX, HNX-INDEX)
    idx_mask = df["ticker"].str.contains("INDEX", na=False)
    n_idx = int(idx_mask.sum())
    if n_idx:
        print(f"[INFO] Dropping {n_idx} index rows (ticker contains 'INDEX').", file=sys.stderr)
    df = df[~idx_mask]
    # Drop rows with non-positive OHLC values
    ohlc_cols = ["open", "high", "low", "close"]
    nonpos_mask = (df[ohlc_cols] <= 0).any(axis=1)
    n_nonpos = int(nonpos_mask.sum())
    if n_nonpos:
        print(f"[INFO] Dropping {n_nonpos} rows with non-positive OHLC values.", file=sys.stderr)
    df = df[~nonpos_mask]
    # Drop rows where high < low
    hl_mask = df["high"] < df["low"]
    n_hl = int(hl_mask.sum())
    if n_hl:
        print(f"[INFO] Dropping {n_hl} rows where high < low.", file=sys.stderr)
    df = df[~hl_mask]


    # Compute 20% tolerance band and count rows outside, but do NOT drop them (per user request)
    rng = df["high"] - df["low"]
    tol = 0.2 * rng
    lo = df["low"] - tol
    hi = df["high"] + tol
    o_in_range = (df["open"] >= lo) & (df["open"] <= hi)
    c_in_range = (df["close"] >= lo) & (df["close"] <= hi)
    range_mask = o_in_range & c_in_range
    n_out = int((~range_mask).sum())
    if n_out:
        print(f"[INFO] Found {n_out} rows where open/close fall > 20% outside [low, high] band. Keeping rows (no drop).", file=sys.stderr)

    # Drop rows with NaN in OHLC
    nan_mask = df[ohlc_cols].isna().any(axis=1)
    n_nan = int(nan_mask.sum())
    if n_nan:
        print(f"[INFO] Dropping {n_nan} rows with NaN in OHLC.", file=sys.stderr)
    df = df[~nan_mask]

    # Deduplicate (date, ticker) pairs, keeping the last occurrence
    before_n = len(df)
    df = df.sort_values(["date", "ticker"]).drop_duplicates(subset=["date", "ticker"], keep="last")
    n_dupes = before_n - len(df)
    if n_dupes:
        print(f"[INFO] Dropped {n_dupes} duplicate (date, ticker) rows.", file=sys.stderr)

    # Restore MultiIndex
    df = df.set_index(["date", "ticker"]).sort_index()

    # Validate: tolerate open/close slightly outside [low, high] (within 20% band) but fail on other errors
    report = validate_ohlcv(df, raise_on_error=False)
    residual_errors = [e for e in report["errors"] if "outside [low, high]" not in e]
    if residual_errors:
        msg = "; ".join(residual_errors[:5])
        raise AssertionError(f"validate_ohlcv failed: {msg}")
    if report.get("warnings"):
        for w in report["warnings"]:
            print(f"[WARN] {w}", file=sys.stderr)

    # Write full OHLCV parquet (or CSV fallback)
    _try_write_parquet(df, out_parquet)

    # Compose monthly universe
    uni_m = filters.compose_universe(
        df,
        monthly=True,
        min_history_days=min_history_days,
        min_price=min_price_vnd,
        min_adv_vnd=min_adv_vnd,
        adv_window=adv_window_days,
        max_ntd=max_non_trading_days,
        ntd_window=ntd_window_days,
    )

    # Write monthly universe CSV
    write_universe_csv(uni_m, out_universe)

    print(f"Wrote OHLCV to {out_parquet} (or CSV fallback).")
    print(f"Wrote monthly universe to {out_universe}.")

    # Optional: export indices monthly returns if configured/requested
    indices_cfg = (cfg.get("indices") or {}) if isinstance(cfg, dict) else {}
    indices_dir = args.indices_dir if args.indices_dir is not None else indices_cfg.get("dir")
    indices_names = args.indices if args.indices is not None else indices_cfg.get("tickers")
    if indices_dir and indices_names:
        try:
            prices = load_indices_from_dir(indices_dir, names=indices_names, price_scale=price_scale)
            # compute monthly simple returns from daily close via daily returns aggregation
            ret_d = prices.pct_change(fill_method=None)
            ret_m = month_returns_from_daily(ret_d)
            ensure_parent(out_indices_monthly)
            ret_m.to_csv(out_indices_monthly, index=True, date_format="%Y-%m-%d")
            print(f"Wrote indices monthly returns to {out_indices_monthly}.")
        except Exception as e:
            print(f"[WARN] Failed to export indices monthly returns: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
