#!/usr/bin/env python3
"""
CLI to load HSX/HNX CSVs, scale prices from kVND to VND on ingest, validate, and emit:
- data/clean/ohlcv.parquet (VND prices)
- data/clean/universe_monthly.csv (boolean matrix [month_end × ticker])

Defaults are read from config/data.yml. You can override inputs/outputs via CLI flags.

Usage:
  poetry run python scripts/make_clean.py
  poetry run python scripts/make_clean.py --config config/data.yml
  poetry run python scripts/make_clean.py --inputs HSX.csv HNX.csv --out-parquet data/clean/ohlcv.parquet --out-universe data/clean/universe_monthly.csv

Notes:
- This script expects CSV headers similar to the vendor files; it will normalize headers (strip angle brackets, etc).
- Prices in input CSVs are assumed kVND; they are scaled to VND internally via price_scale (default 1000.0).
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
from bwv.data_io import load_ohlcv, validate_ohlcv
from bwv import filters


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
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg_path = Path(args.config)
    cfg = read_config(cfg_path) if cfg_path.exists() else {}

    # Inputs
    cfg_inputs = cfg.get("inputs") or []
    inputs: List[str] = args.inputs if args.inputs is not None else cfg_inputs
    if not inputs:
        print("No inputs provided. Specify via --inputs HSX.csv HNX.csv or in config/data.yml.", file=sys.stderr)
        return 2

    # Outputs
    outputs_cfg = cfg.get("outputs") or {}
    out_parquet = Path(args.out_parquet or outputs_cfg.get("ohlcv_parquet", "data/clean/ohlcv.parquet"))
    out_universe = Path(args.out_universe or outputs_cfg.get("universe_monthly_csv", "data/clean/universe_monthly.csv"))

    # Eligibility/config
    price_scale = args.price_scale if args.price_scale is not None else float(cfg.get("price_scale", 1000.0))
    min_history_days = args.min_history_days if args.min_history_days is not None else int(cfg.get("min_history_days", 126))
    min_price_vnd = args.min_price_vnd if args.min_price_vnd is not None else float(cfg.get("min_price_vnd", 1000.0))
    min_adv_vnd = args.min_adv_vnd if args.min_adv_vnd is not None else float(cfg.get("min_adv_vnd", 100_000_000.0))
    adv_window_days = args.adv_window_days if args.adv_window_days is not None else int(cfg.get("adv_window_days", 60))
    max_non_trading_days = args.max_non_trading_days if args.max_non_trading_days is not None else int(cfg.get("max_non_trading_days", 15))
    ntd_window_days = args.ntd_window_days if args.ntd_window_days is not None else int(cfg.get("ntd_window_days", 126))

    # Load and scale to VND
    df = load_ohlcv(inputs, price_scale=price_scale)

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

    # Drop rows where open/close outside [low, high]
    o_in_range = (df["open"] >= df["low"]) & (df["open"] <= df["high"])
    c_in_range = (df["close"] >= df["low"]) & (df["close"] <= df["high"])
    range_mask = o_in_range & c_in_range
    n_out = int((~range_mask).sum())
    if n_out:
        print(f"[INFO] Dropping {n_out} rows where open/close outside [low, high].", file=sys.stderr)
    df = df[range_mask]

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

    # Validate (fail fast on hard errors)
    report = validate_ohlcv(df, raise_on_error=True)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
