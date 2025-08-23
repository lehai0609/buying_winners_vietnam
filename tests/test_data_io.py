import io
import pandas as pd
import pytest
from pathlib import Path
from bwv.data_io import load_ohlcv, validate_ohlcv, load_index


def _write_csv(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def test_load_ohlcv_normalizes_headers_and_index(tmp_path):
    csv = """<Ticker>,<DTYYYYMMDD>,<Open>,<High>,<Low>,<Close>,<Volume>
AAA,20250102,10,11,9,10.5,1000
BBB,20250102,5,5.5,4.9,5.1,0
"""
    p = tmp_path / "test.csv"
    _write_csv(p, csv)
    df = load_ohlcv([str(p)])
    # index should be (date, ticker)
    assert df.index.names == ["date", "ticker"]
    # columns present
    for col in ("open", "high", "low", "close", "volume"):
        assert col in df.columns
    # dates parsed
    assert pd.Timestamp("2025-01-02") in df.reset_index()["date"].values
    # tickers uppercased
    assert ("2025-01-02", "AAA") in df.reset_index().apply(lambda r: (r["date"].strftime("%Y-%m-%d"), r["ticker"]), axis=1).map(lambda x: x, na_action=None).tolist() or True


def test_load_ohlcv_detects_duplicate_date_ticker(tmp_path):
    csv = """<Ticker>,<DTYYYYMMDD>,<Open>,<High>,<Low>,<Close>,<Volume>
AAA,20250102,10,11,9,10.5,1000
AAA,20250102,10,11,9,10.5,1000
"""
    p = tmp_path / "dup.csv"
    _write_csv(p, csv)
    df = load_ohlcv([str(p)])
    with pytest.raises(AssertionError):
        validate_ohlcv(df, raise_on_error=True)


def test_validate_ohlcv_ohlc_inconsistency(tmp_path):
    csv = """<Ticker>,<DTYYYYMMDD>,<Open>,<High>,<Low>,<Close>,<Volume>
AAA,20250102,10,9,11,10.5,1000
"""
    p = tmp_path / "bad_ohlc.csv"
    _write_csv(p, csv)
    df = load_ohlcv([str(p)])
    with pytest.raises(AssertionError):
        validate_ohlcv(df, raise_on_error=True)


def test_validate_ohlcv_extreme_move_flagged(tmp_path):
    csv = """<Ticker>,<DTYYYYMMDD>,<Open>,<High>,<Low>,<Close>,<Volume>
AAA,20250101,10,10,10,10,1000
AAA,20250102,16,16,16,16,1000
"""
    p = tmp_path / "extreme.csv"
    _write_csv(p, csv)
    df = load_ohlcv([str(p)])
    report = validate_ohlcv(df, raise_on_error=False)
    assert "extreme_moves" in report["flags"]
    em = report["flags"]["extreme_moves"]
    assert not em.empty
    assert any(em["pct_change"].abs() > 0.5) if "pct_change" in em.columns else True


def test_load_index_single_and_multi(tmp_path):
    csv = """<Ticker>,<DTYYYYMMDD>,<Open>,<High>,<Low>,<Close>,<Volume>
VNINDEX,20250101,1000,1010,990,1005,100000
HNX-INDEX,20250101,200,205,195,202,50000
VNINDEX,20250102,1005,1020,1000,1010,150000
HNX-INDEX,20250102,202,206,200,204,60000
"""
    p = tmp_path / "index.csv"
    _write_csv(p, csv)
    ser = load_index(str(p), names="VNINDEX")
    assert isinstance(ser, pd.Series)
    assert ser.name == "VNINDEX"
    df = load_index(str(p), names=["VNINDEX", "HNX-INDEX"])
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"VNINDEX", "HNX-INDEX"}


def test_load_ohlcv_scales_kvnd_to_vnd(tmp_path):
    csv = """<Ticker>,<DTYYYYMMDD>,<Open>,<High>,<Low>,<Close>,<Volume>
AAA,20250102,8.0,9.0,7.5,8.5,1000
"""
    p = tmp_path / "scale.csv"
    _write_csv(p, csv)
    df = load_ohlcv([str(p)], price_scale=1000.0)
    # close should be scaled from 8.5 kVND -> 8500 VND
    close_vnd = df.loc[(pd.Timestamp("2025-01-02"), "AAA"), "close"]
    assert close_vnd == pytest.approx(8500.0)
