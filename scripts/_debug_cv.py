import pandas as pd
import numpy as np
import json
from src.bwv import cv

def _make_synthetic_ohlcv(start="2019-01-01", end="2020-12-31", tickers=None):
    if tickers is None:
        tickers = ["AAA", "BBB"]
    dates = pd.bdate_range(start=start, end=end)
    rows = []
    for tk_idx, tk in enumerate(tickers):
        drift = 0.0005 + 0.0005 * tk_idx
        price = 100.0 * (1.0 + drift) ** np.arange(len(dates))
        for d, p in zip(dates, price):
            rows.append({"date": d, "ticker": tk, "open": float(p), "high": float(p), "low": float(p), "close": float(p), "volume": 1000})
    df = pd.DataFrame(rows).set_index(["date", "ticker"])
    return df.sort_index()

def _make_universe_monthly(ohlcv_df):
    dates = pd.DatetimeIndex(sorted(set(ohlcv_df.index.get_level_values("date"))))
    month_ends = pd.DatetimeIndex(pd.Series(dates).groupby(dates.to_period("M")).max().values)
    tickers = sorted(set(ohlcv_df.index.get_level_values("ticker")))
    df = pd.DataFrame(True, index=month_ends, columns=tickers, dtype=object)
    return df

if __name__ == "__main__":
    ohlcv = _make_synthetic_ohlcv()
    universe = _make_universe_monthly(ohlcv)
    J_list = [3,6]
    K_list = [1,3]
    cfg = {"skip_days": 5}
    res1 = cv.grid_search(ohlcv, universe, J_list, K_list, config=cfg)
    res2 = cv.grid_search(ohlcv, universe, J_list, K_list, config=cfg)
    print("RES1\n", res1.to_dict(orient="list"))
    print("RES2\n", res2.to_dict(orient="list"))
    # show dtypes and repr
    print("\nRES1 dtypes\n", res1.dtypes.to_dict())
    print("\nRES2 dtypes\n", res2.dtypes.to_dict())
    # elementwise comparison
    eq = res1.equals(res2)
    print("\nDataFrame equals:", eq)
    # show where differs
    import numpy as np
    diffs = {}
    for col in res1.columns:
        a = res1[col].apply(lambda x: float(x) if (pd.notna(x) and isinstance(x, (int, float, np.number))) else (str(x) if pd.isna(x) else x)).tolist()
        b = res2[col].apply(lambda x: float(x) if (pd.notna(x) and isinstance(x, (int, float, np.number))) else (str(x) if pd.isna(x) else x)).tolist()
        if a != b:
            diffs[col] = {"res1": a, "res2": b}
    print("\nDIFFS:\n", json.dumps(diffs, indent=2, default=str))
