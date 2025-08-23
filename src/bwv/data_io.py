"""
Minimal data I/O and validation utilities for OHLCV and index CSVs.

Provides:
- load_ohlcv(paths, start, end, expect_bracket_headers=True, add_exchange_col=True, price_scale=1000.0)
- validate_ohlcv(df, raise_on_error=True)
- load_index(path, names, price_field='close', align_to=None, price_scale=1000.0)

The functions are conservative (fail-fast on hard errors) and return
structured reports suitable for programmatic checks in tests and later modules.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Iterable, List, Optional, Union, Dict, Any

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _normalize_columns(columns: Iterable[str]) -> List[str]:
    """
    Normalize incoming column names by:
    - stripping angle brackets <> if present
    - lower-casing
    - removing surrounding whitespace
    """
    out = []
    for c in columns:
        c2 = str(c).strip()
        # remove < and >
        c2 = re.sub(r"[<>]", "", c2)
        c2 = c2.strip().lower()
        out.append(c2)
    return out


def _canonical_col_map(cols: List[str]) -> Dict[str, str]:
    """
    Map various possible date column names to canonical ones.
    """
    mapping = {}
    for c in cols:
        if c in ("dt", "date", "dtyyyymmdd", "dtyyyymmdd"):
            mapping[c] = "date"
        elif c in ("dt20180101",):
            mapping[c] = "date"
        elif c in ("ticker", "symbol"):
            mapping[c] = "ticker"
        elif c in ("open", "high", "low", "close", "volume", "value"):
            mapping[c] = c
        elif c.startswith("dt") and re.fullmatch(r"dt\d+", c):
            # fallback: treat as date
            mapping[c] = "date"
    return mapping


def _infer_exchange_from_path(p: Union[str, Path]) -> Optional[str]:
    name = Path(p).stem.upper()
    # simple inference: HSX.csv -> HSX, HNX.csv -> HNX, anything else -> None
    if "HSX" in name:
        return "HSX"
    if "HNX" in name:
        return "HNX"
    return None


def load_ohlcv(
    paths: Iterable[Union[str, Path]],
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    *,
    expect_bracket_headers: bool = True,
    add_exchange_col: bool = True,
    price_scale: float = 1000.0,
) -> pd.DataFrame:
    """
    Load one or more OHLCV CSVs and normalize into a tidy DataFrame
    indexed by [date, ticker] with columns [open, high, low, close, volume, value(opt), exchange(opt)].

    Notes
    - Price units: input CSVs often have prices in thousands of VND (kVND). Use price_scale (default 1000.0) to scale to VND on ingest.
    - Volume units: shares.
    - Trading value units: VND (price[VND] × volume[shares]).

    Parameters
    - paths: iterable of file paths to CSVs
    - start, end: optional date bounds (strings parseable by pandas or pd.Timestamp)
    - expect_bracket_headers: whether to strip <> from headers (True for these files)
    - add_exchange_col: add column 'exchange' inferred from filename

    Returns
    - DataFrame with MultiIndex (date, ticker) sorted by index ascending
    """
    frames = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        # read raw first row to inspect header then read with pandas
        df = pd.read_csv(p, dtype=str)
        # normalize column names
        cols_norm = _normalize_columns(df.columns)
        df.columns = cols_norm
        # canonical mapping (ensure expected columns exist)
        # Accept variations like dtYYYYMMDD -> date
        colmap = _canonical_col_map(cols_norm)
        # build a rename map to canonical names for common variants (dtYYYYMMDD, symbol, etc)
        rename_map = {}
        for c in cols_norm:
            if c in colmap:
                rename_map[c] = colmap[c]
            elif re.fullmatch(r"dt\d{6,8}", c):
                rename_map[c] = "date"
            elif c.startswith("dt") and any(ch.isdigit() for ch in c):
                rename_map[c] = "date"
            elif c in ("symbol",):
                rename_map[c] = "ticker"
        if rename_map:
            df = df.rename(columns=rename_map)
            cols_norm = _normalize_columns(df.columns)
        # ensure required cols are present after normalization
        required = {"date", "ticker", "open", "high", "low", "close"}
        present = set(cols_norm)
        if not required.issubset(present):
            missing = required - present
            raise AssertionError(f"Missing required columns {missing} in {p.name}; got {cols_norm}")
        # Keep volume if present
        keep_cols = ["date", "ticker", "open", "high", "low", "close"]
        if "volume" in cols_norm:
            keep_cols.append("volume")
        if "value" in cols_norm:
            keep_cols.append("value")
        # slice
        df = df[keep_cols].copy()
        # parse types
        # date in format YYYYMMDD or parseable
        try:
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="raise")
        except Exception:
            # fallback to pandas inference
            df["date"] = pd.to_datetime(df["date"], errors="raise")
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        # numeric conversions - coerce then check in validation
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        # scale OHLC prices from kVND to VND (or according to provided price_scale)
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]] * float(price_scale)
        if "volume" in df.columns:
            # volumes may be integers; allow NA
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Float64")
        if "value" in df.columns:
            # trading value column (assumed already in VND); ensure numeric
            df["value"] = pd.to_numeric(df["value"], errors="coerce").astype(float)
        if add_exchange_col:
            exchange = _infer_exchange_from_path(p.name)
            df["exchange"] = exchange
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.concat(frames, ignore_index=True)
    # optional date filtering
    if start is not None:
        start_ts = pd.to_datetime(start)
        df = df[df["date"] >= start_ts]
    if end is not None:
        end_ts = pd.to_datetime(end)
        df = df[df["date"] <= end_ts]
    # ensure sorting by ticker/date ascending
    df = df.sort_values(["ticker", "date"], ascending=[True, True]).reset_index(drop=True)
    # set MultiIndex (date, ticker) for downstream code
    df = df.set_index(["date", "ticker"]).sort_index()
    return df


def validate_ohlcv(df: pd.DataFrame, *, raise_on_error: bool = True) -> Dict[str, Any]:
    """
    Validate the OHLCV DataFrame produced by load_ohlcv.

    Checks:
    - no duplicates in index (date,ticker)
    - open/high/low/close are finite and > 0
    - high >= low
    - open/close within [low, high] (warn or error)
    - volume nullable; zero and NaN flagged as warnings
    - extreme daily pct moves (abs(pct) > 50%) flagged

    Returns a report dictionary:
    {
      "errors": [...],
      "warnings": [...],
      "flags": {"extreme_moves": DataFrame, "zero_volume": DataFrame},
      "coverage": DataFrame (per-year summary)
    }

    If raise_on_error is True and any errors exist, raises AssertionError with message.
    """
    errors = []
    warnings = []
    flags: Dict[str, Any] = {}

    # duplicates
    if df.index.duplicated().any():
        dup_idx = df.index[df.index.duplicated(keep=False)]
        sample = dup_idx[:10]
        errors.append(f"Duplicate (date,ticker) pairs detected; sample: {list(sample)}")

    # Check that required columns exist in df
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            errors.append(f"Missing required column '{col}' in dataframe")
    # numeric finite and >0
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            bad = ~np.isfinite(df[col].to_numpy())
            if bad.any():
                sample_idx = df.index[bad][:10]
                errors.append(f"Non-finite values in '{col}'; sample: {list(sample_idx)}")
            nonpos = df[col] <= 0
            if nonpos.any():
                sample_idx = df.index[nonpos][:10]
                errors.append(f"Non-positive values in '{col}'; sample: {list(sample_idx)}")
    # high >= low
    if ("high" in df.columns) and ("low" in df.columns):
        bad = df["high"] < df["low"]
        if bad.any():
            sample_idx = df.index[bad][:10]
            errors.append(f"Rows where high < low; sample: {list(sample_idx)}")
    # open/close within [low, high]
    for col in ("open", "close"):
        cond = (df[col] < df["low"]) | (df[col] > df["high"])
        if cond.any():
            sample_idx = df.index[cond][:10]
            errors.append(f"Rows where {col} outside [low, high]; sample: {list(sample_idx)}")

    # volume checks - flag zero or NaN as warnings (not hard errors)
    if "volume" in df.columns:
        vol_zero = df["volume"].isna() | (df["volume"] == 0)
        if vol_zero.any():
            flags["zero_volume"] = df[vol_zero].copy()
            warnings.append(f"{int(vol_zero.sum())} rows have zero or missing volume (flagged).")

    # extreme daily returns detection (close-to-close by ticker)
    # compute pct change grouped by ticker (we assume index is date,ticker)
    if not isinstance(df.index, pd.MultiIndex):
        raise AssertionError("Expected MultiIndex [date, ticker].")
    # pivot close to compute per-ticker pct change easily
    close = df["close"].copy()
    # compute pct change by shifting within each ticker group
    # Because index is (date,ticker), groupby level=1 (ticker) then pct_change on level=0 (date)
    pct_records = []
    # Use groupby on level ticker
    for ticker, sub in df.reset_index().groupby("ticker"):
        sub = sub.sort_values("date")
        sub = sub.set_index("date")
        pct = sub["close"].pct_change()
        large = pct.abs() > 0.5
        if large.any():
            tmp = sub[large].copy()
            tmp["pct_change"] = pct[large]
            tmp["ticker"] = ticker
            pct_records.append(tmp[["ticker", "pct_change"]])
    if pct_records:
        flags["extreme_moves"] = pd.concat(pct_records)

    # coverage summary: per-year unique tickers and rows
    dates = df.reset_index()["date"]
    years = dates.dt.year
    cov = (
        df.reset_index()
        .assign(year=years)
        .groupby("year")
        .agg(n_rows=("date", "size"), n_tickers=("ticker", lambda s: s.nunique()))
        .sort_index()
    )
    cov.index.name = "year"
    report = {"errors": errors, "warnings": warnings, "flags": flags, "coverage": cov}

    if errors and raise_on_error:
        # create a concise error message
        msg = "; ".join(errors[:5])
        raise AssertionError(f"validate_ohlcv failed: {msg}")
    return report


def load_index(
    path: Union[str, Path],
    names: Union[str, Iterable[str]] = None,
    *,
    price_field: str = "close",
    align_to: Optional[pd.DatetimeIndex] = None,
    price_scale: float = 1000.0,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Load index CSV (same schema as OHLCV). Filter rows with ticker equal to `names`.
    names can be a single string or an iterable of strings.

    Returns:
    - pd.Series (if single name) indexed by date containing price_field
    - pd.DataFrame (if multiple names) indexed by date with columns per ticker

    If align_to is provided, reindex to align_to leaving NaNs (no forward/backfill).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Index file not found: {p}")
    df = pd.read_csv(p, dtype=str)
    cols_norm = _normalize_columns(df.columns)
    df.columns = cols_norm
    # canonical mapping and rename (handle dtYYYYMMDD etc)
    colmap = _canonical_col_map(cols_norm)
    rename_map = {}
    for c in cols_norm:
        if c in colmap:
            rename_map[c] = colmap[c]
        elif re.fullmatch(r"dt\d{6,8}", c):
            rename_map[c] = "date"
        elif c.startswith("dt") and any(ch.isdigit() for ch in c):
            rename_map[c] = "date"
        elif c in ("symbol",):
            rename_map[c] = "ticker"
    if rename_map:
        df = df.rename(columns=rename_map)
        cols_norm = _normalize_columns(df.columns)
        df.columns = cols_norm
    if "date" not in df.columns or "ticker" not in df.columns:
        raise AssertionError("Index file missing required 'date' or 'ticker' columns.")
    try:
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="raise")
    except Exception:
        df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    if price_field not in df.columns:
        raise AssertionError(f"Price field '{price_field}' not in index file columns.")
    # filter names
    if names is None:
        sel = df
    else:
        if isinstance(names, str):
            names = [names]
        names_up = [n.upper() for n in names]
        sel = df[df["ticker"].isin(names_up)].copy()
    if sel.empty:
        raise AssertionError(f"No matching index series for {names} in {p.name}")
    # pivot to have columns per ticker
    sel = sel.sort_values("date")
    out = sel.pivot(index="date", columns="ticker", values=price_field)
    # coerce to numeric and scale to VND (price_scale default 1000.0)
    out = out.apply(pd.to_numeric, errors="coerce").astype(float)
    out = out * float(price_scale)
    # align if requested
    if align_to is not None:
        out = out.reindex(align_to)
    # if single name return series
    if out.shape[1] == 1:
        ser = out.iloc[:, 0]
        ser.name = out.columns[0]
        return ser
    return out
