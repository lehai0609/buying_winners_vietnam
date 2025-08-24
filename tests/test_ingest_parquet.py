import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from bwv.data_io import load_ohlcv_parquet_dirs, validate_ohlcv


def _make_parquet(p: Path, rows: list[dict]):
    df = pd.DataFrame(rows)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def test_load_ohlcv_parquet_dirs_basic(tmp_path):
    # Setup HSX/ and HNX/ with per-ticker parquet files
    hsx = tmp_path / "HSX"
    hnx = tmp_path / "HNX"

    _make_parquet(
        hsx / "AAA.parquet",
        [
            # dates as YYYYMMDD strings to exercise parser; prices in kVND
            {"date": "20250102", "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 1000},
            {"date": "20250103", "open": 10.2, "high": 10.5, "low": 10.0, "close": 10.4, "volume": 2000},
        ],
    )
    _make_parquet(
        hnx / "BBB.parquet",
        [
            {"date": "20250102", "open": 5.0, "high": 5.5, "low": 4.9, "close": 5.1, "volume": 0},
        ],
    )

    df = load_ohlcv_parquet_dirs([hsx, hnx], price_scale=1000.0)

    # Canonical index and columns
    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.names == ["date", "ticker"]
    for col in ("open", "high", "low", "close"):
        assert col in df.columns

    # Scaling from kVND to VND
    close_vnd = df.loc[(pd.Timestamp("2025-01-02"), "AAA"), "close"]
    assert close_vnd == pytest.approx(10500.0)

    # Ticker inferred from stem and exchange from parent folder
    assert df.loc[(pd.Timestamp("2025-01-02"), "AAA"), "exchange"] == "HSX"
    assert df.loc[(pd.Timestamp("2025-01-02"), "BBB"), "exchange"] == "HNX"

    # Validate should pass hard checks
    report = validate_ohlcv(df, raise_on_error=False)
    assert not report["errors"]


def test_load_ohlcv_parquet_dirs_date_bounds(tmp_path):
    hsx = tmp_path / "HSX"
    _make_parquet(
        hsx / "AAA.parquet",
        [
            {"date": "20250101", "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 1000},
            {"date": "20250110", "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 1000},
        ],
    )
    df = load_ohlcv_parquet_dirs([hsx], start="2025-01-05", end="2025-01-31", price_scale=1000.0)
    idx_dates = df.reset_index()["date"].dt.strftime("%Y-%m-%d").unique().tolist()
    assert "2025-01-10" in idx_dates
    assert "2025-01-01" not in idx_dates
