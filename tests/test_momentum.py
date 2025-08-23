import numpy as np
import pandas as pd
import pytest

from bwv import returns as rtrs
from bwv import momentum as mom


def _make_panel(dates, tickers, close_vals):
    """
    Build minimal OHLCV-like DataFrame with MultiIndex (date, ticker) and a 'close' column.
    close_vals can be a scalar, a dict of ticker->list, or a 2D list-like (dates x tickers).
    """
    rows = []
    for i, d in enumerate(dates):
        for j, t in enumerate(tickers):
            if isinstance(close_vals, dict):
                c = close_vals[t][i]
            elif hasattr(close_vals, "__len__") and not isinstance(close_vals, (str, bytes)):
                # assume 2D list-like with shape (len(dates), len(tickers))
                c = close_vals[i][j]
            else:
                c = close_vals
            rows.append({"date": d, "ticker": t, "close": c})
    df = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
    return df


def test_scores_window_and_skip_logic():
    # generate ~3 months of daily data
    dates = pd.date_range("2025-01-01", periods=90, freq="D")
    tickers = ["AAA", "BBB"]
    close_A = [100.0 * (1.01 ** i) for i in range(len(dates))]
    close_B = [100.0 * (1.005 ** i) for i in range(len(dates))]
    closes = {"AAA": close_A, "BBB": close_B}
    df = _make_panel(dates, tickers, closes)
    ret_d = rtrs.daily_simple_returns(df)

    # formation dates
    me = rtrs.month_end_index(ret_d.index)
    # all eligible
    universe = pd.DataFrame(True, index=me, columns=ret_d.columns, dtype=object)

    # compute momentum scores using implementation
    scores = mom.momentum_scores(ret_d, universe, J=1, skip_days=5)

    # compute expected using cum_return_skip and apply same mask
    expected = rtrs.cum_return_skip(ret_d, J_months=1, skip_days=5).reindex(index=me, columns=ret_d.columns)
    # apply mask (all True so identical)
    for dt in universe.index:
        for col in universe.columns:
            exp = expected.loc[dt, col]
            act = scores.loc[dt, col]
            if np.isnan(exp):
                assert np.isnan(act)
            else:
                assert act == pytest.approx(exp)


def test_scores_respect_universe_mask():
    dates = pd.date_range("2025-02-01", periods=60, freq="D")
    tickers = ["X", "Y"]
    close_X = [50.0 * (1.01 ** i) for i in range(len(dates))]
    close_Y = [80.0 * (1.02 ** i) for i in range(len(dates))]
    df = _make_panel(dates, tickers, {"X": close_X, "Y": close_Y})
    ret_d = rtrs.daily_simple_returns(df)
    me = rtrs.month_end_index(ret_d.index)
    universe = pd.DataFrame(True, index=me, columns=ret_d.columns, dtype=object)
    # mark Y in the last formation month as ineligible
    universe.iloc[-1, universe.columns.get_loc("Y")] = False

    scores = mom.momentum_scores(ret_d, universe, J=1, skip_days=5)
    last_form = me[-1]
    # Y should be NaN at last formation date
    assert np.isnan(scores.loc[last_form, "Y"])
    # X should still be numeric (unless insufficient history)
    assert (np.isfinite(scores.loc[last_form, "X"]) or np.isnan(scores.loc[last_form, "X"]))


def test_decile_ranks_deterministic_ties():
    # construct a single-formation score row with ties and shuffled column order
    date = pd.Timestamp("2025-03-31")
    # intentionally out-of-order columns to ensure ticker name tie-breaker is used
    cols = ["B", "A", "D", "C"]
    scores = pd.DataFrame(index=[date], columns=cols, dtype=float)
    # values: A=1.0, B=1.0, C=0.5, D=0.5
    scores.loc[date, "A"] = 1.0
    scores.loc[date, "B"] = 1.0
    scores.loc[date, "C"] = 0.5
    scores.loc[date, "D"] = 0.5

    ranks = mom.decile_ranks(scores, q=2)
    # Expectation with sorting ascending score, ticker ascending:
    # valid tickers sorted ascending score: C, D, A, B
    # ordinal r = 0,1,2,3 -> decile = floor(r * 2 / 4) = [0,0,1,1]
    assert ranks.loc[date, "A"] == 1
    assert ranks.loc[date, "B"] == 1
    assert ranks.loc[date, "C"] == 0
    assert ranks.loc[date, "D"] == 0
    # dtype should be pandas nullable Int64
    assert str(ranks.dtypes[0]) in ("Int64", "Int64Dtype", "Int64()")


def test_decile_bins_approximately_balanced():
    date = pd.Timestamp("2025-04-30")
    # exact divisible case: n=20, q=10 -> 2 per decile
    n = 20
    q = 10
    tickers = [f"T{i:02d}" for i in range(n)]
    scores = pd.DataFrame(index=[date], columns=tickers, dtype=float)
    # increasing scores so ordering is deterministic
    for i, t in enumerate(tickers):
        scores.loc[date, t] = float(i)
    ranks = mom.decile_ranks(scores, q=q)
    counts = [(ranks.loc[date] == d).sum() for d in range(q)]
    assert all(c == 2 for c in counts)

    # non-divisible case: n=23, q=10 -> counts differ by at most 1
    n2 = 23
    tickers2 = [f"U{i:02d}" for i in range(n2)]
    scores2 = pd.DataFrame(index=[date], columns=tickers2, dtype=float)
    for i, t in enumerate(tickers2):
        scores2.loc[date, t] = float(i)
    ranks2 = mom.decile_ranks(scores2, q=q)
    counts2 = [(ranks2.loc[date] == d).sum() for d in range(q)]
    assert max(counts2) - min(counts2) <= 1


def test_top_decile_mask_bool_dtype():
    date = pd.Timestamp("2025-05-30")
    tickers = ["A", "B", "C"]
    ranks = pd.DataFrame(index=[date], columns=tickers, dtype="Int64")
    # set ranks so that 'C' is top decile (2 with q=3)
    ranks.loc[date, "A"] = 0
    ranks.loc[date, "B"] = 1
    ranks.loc[date, "C"] = 2

    mask = mom.top_decile_mask(ranks, decile=2)
    # dtype should be object (native Python bools)
    assert mask.dtypes.unique().tolist() == [object]
    # the entry for C should be native Python True
    assert mask.loc[date, "C"] is True
    # other entries should be False (native bool)
    assert mask.loc[date, "A"] is False
    assert isinstance(mask.loc[date, "A"], bool)


def test_nan_and_insufficient_history():
    # create only 1 month of data but request J=3 months -> insufficient history
    dates = pd.date_range("2025-06-01", periods=20, freq="D")
    tickers = ["AA", "BB"]
    close_A = [10.0 * (1.01 ** i) for i in range(len(dates))]
    close_B = [20.0 * (1.02 ** i) for i in range(len(dates))]
    df = _make_panel(dates, tickers, {"AA": close_A, "BB": close_B})
    ret_d = rtrs.daily_simple_returns(df)
    me = rtrs.month_end_index(ret_d.index)
    universe = pd.DataFrame(True, index=me, columns=ret_d.columns, dtype=object)

    scores = mom.momentum_scores(ret_d, universe, J=3, skip_days=5)
    # all entries should be NaN because there's insufficient history
    assert scores.isna().all().all()

    ranks = mom.decile_ranks(scores)
    # ranks should be all NA (pandas NA)
    assert ranks.isna().all().all()

    mask = mom.top_decile_mask(ranks)
    # mask should be all False (native bool)
    assert (mask.values == False).all()
