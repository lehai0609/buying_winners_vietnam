import pandas as pd
from pathlib import Path

from bwv.data_io import load_ohlcv
from bwv import filters


def _write_csv(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def test_adv_threshold_respects_vnd_scaling_on_ingest(tmp_path):
    """
    Build a CSV with close in kVND and volume in shares.
    After ingest with price_scale=1000, ADV threshold at 100M VND should be applied correctly.
    Construct data so that rolling mean (window=2, shifted by 1) crosses threshold on day 4.
    """
    # close in kVND (10 -> 10,000 VND after scaling)
    # volumes chosen so trading value VND = close[VND] * volume -> [10M, 100M, 100M, 100M, 100M]
    csv = """<Ticker>,<DTYYYYMMDD>,<Open>,<High>,<Low>,<Close>,<Volume>
AAA,20250101,10,10,10,10,1000
AAA,20250102,10,10,10,10,10000
AAA,20250103,10,10,10,10,10000
AAA,20250104,10,10,10,10,10000
AAA,20250105,10,10,10,10,10000
"""
    p = tmp_path / "kvnd_adv.csv"
    _write_csv(p, csv)

    df = load_ohlcv([str(p)], price_scale=1000.0)

    # Apply ADV filter with 2-day window and 100M VND threshold
    mask = filters.adv_filter(df, min_adv_vnd=100_000_000.0, window=2)

    # Shifted behavior:
    # day2 uses avg(day1)=10M -> False
    # day3 uses avg(day1,day2)=(10M+100M)/2=55M -> False
    # day4 uses avg(day2,day3)=(100M+100M)/2=100M -> True
    # day5 uses avg(day3,day4)=(100M+100M)/2=100M -> True
    assert mask.loc[pd.Timestamp("2025-01-02"), "AAA"] is False
    assert mask.loc[pd.Timestamp("2025-01-03"), "AAA"] is False
    assert mask.loc[pd.Timestamp("2025-01-04"), "AAA"] is True
    assert mask.loc[pd.Timestamp("2025-01-05"), "AAA"] is True
