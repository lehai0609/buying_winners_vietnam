import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from bwv.data_io import load_indices_from_dir
import scripts.make_clean as make_clean


def _write_csv(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _make_parquet(p: Path, rows: list[dict]):
    df = pd.DataFrame(rows)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def test_load_indices_from_dir_prices_vnd(tmp_path):
    # Build vn_indices/ with vendor-like headers
    idx_dir = tmp_path / "vn_indices"
    csv = """<Ticker>,<DTYYYYMMDD>,<Open>,<High>,<Low>,<Close>,<Volume>
VNINDEX,20250102,1000,1010,990,1005,100000
VNINDEX,20250103,1005,1015,1000,1010,120000
HNX-INDEX,20250102,200,205,195,202,50000
HNX-INDEX,20250103,202,206,200,204,60000
"""
    _write_csv(idx_dir / "VNINDEX.csv", csv)
    _write_csv(idx_dir / "HNXINDEX.csv", csv)

    prices = load_indices_from_dir(idx_dir, names=["VNINDEX", "HNX-INDEX"], price_field="close", price_scale=1000.0)
    # Expect a wide table with two columns scaled to VND (kVND*1000)
    assert isinstance(prices, pd.DataFrame)
    assert set(["VNINDEX", "HNX-INDEX"]).issubset(set(prices.columns))
    # Check scaling: 1005 kVND -> 1_005_000 VND
    v = prices.loc[pd.Timestamp("2025-01-02"), "VNINDEX"]
    assert v == pytest.approx(1005_000.0)


def test_make_clean_optional_indices_monthly_export(tmp_path, monkeypatch):
    """
    End-to-end: Provide parquet OHLCV via --inputs-dir and indices via --indices-dir/--indices,
    and assert indices monthly CSV is created with expected shape.
    """
    # Prepare minimal OHLCV parquet in HSX/ to allow make_clean to proceed
    hsx = tmp_path / "HSX"
    _make_parquet(
        hsx / "AAA.parquet",
        [
            {"date": "2025-01-30", "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.2, "volume": 1000},
            {"date": "2025-01-31", "open": 10.2, "high": 10.6, "low": 10.0, "close": 10.4, "volume": 1500},
        ],
    )

    # Prepare vn_indices/ with two days -> one monthly return
    idx_dir = tmp_path / "vn_indices"
    idx_csv = """ticker,date,open,high,low,close,volume
VNINDEX,2025-01-30,1000,1010,990,1005,100000
VNINDEX,2025-01-31,1005,1015,1000,1010,120000
"""
    _write_csv(idx_dir / "VNINDEX.csv", idx_csv)

    out_parquet = tmp_path / "out" / "ohlcv.parquet"
    out_universe = tmp_path / "out" / "universe.csv"
    out_indices = tmp_path / "out" / "indices_monthly.csv"

    # Run script main with args (do not rely on config)
    argv = [
        "--inputs-dir", str(hsx),
        "--out-parquet", str(out_parquet),
        "--out-universe", str(out_universe),
        "--indices-dir", str(idx_dir),
        "--indices", "VNINDEX",
        "--out-indices-monthly", str(out_indices),
        "--price-scale", "1000.0",
        "--min-history-days", "1",  # relax for tiny fixture
        "--min-adv-vnd", "0",
        "--adv-window-days", "1",
        "--max-non-trading-days", "1000",
        "--ntd-window-days", "10",
    ]
    rc = make_clean.main(argv)
    assert rc == 0
    assert out_indices.exists()

    # Validate indices monthly CSV contents: one month-end row, one column
    mdf = pd.read_csv(out_indices, index_col=0, parse_dates=True)
    assert "VNINDEX" in mdf.columns
    # For 1005 -> 1010 close change: return = 1010/1005 - 1
    expected = (1010.0 - 1005.0) / 1005.0
    # Allow small float tolerance
    assert mdf.shape[0] >= 1
    first_val = float(mdf.iloc[0]["VNINDEX"])
    assert first_val == pytest.approx(expected, rel=1e-9, abs=1e-12)
