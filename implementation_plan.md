# Implementation Plan

[Overview]
Implement the M4 Momentum signal computation module to produce monthly J-month momentum scores with a 5-trading-day skip, cross-sectional decile ranks, and a top-decile mask for long-only winner portfolios.

This implementation extends the existing M0–M3 pipeline by adding a momentum layer that consumes daily returns and a monthly eligibility universe to compute formation-month momentum signals. Scores are computed as cumulative simple returns over the past J calendar months excluding the most recent skip_days trading days to avoid reversal. Cross-sectional decile ranks are then assigned deterministically per month, with ties broken consistently by ticker code order, enabling stable long-only D10 portfolio construction in subsequent milestones. The design strictly avoids look-ahead by deriving all windows from available trading-date positions and by applying ranks only to the monthly universe determined with information up to the formation date. The module is parameterized (J, skip_days, q) and documented for traceability.

[Types]
Define precise DataFrame and Series shapes used across score computation and ranking.

Type specifications:
- DailyReturnsWide: pd.DataFrame
  - index: pd.DatetimeIndex of trading days (ascending)
  - columns: tickers (str)
  - dtype: float (simple daily returns, NaN allowed)
- UniverseMonthlyMask: pd.DataFrame
  - index: pd.DatetimeIndex of month-end trading days (formation dates)
  - columns: tickers (str)
  - dtype: bool or object[bool] (True if eligible on formation date)
- MomentumScoresMonthly: pd.DataFrame
  - index: pd.DatetimeIndex of month-end trading days (formation dates)
  - columns: tickers (str)
  - dtype: float (NaN where insufficient history or ineligible)
- DecileRanksMonthly: pd.DataFrame
  - index: month-end trading days
  - columns: tickers
  - dtype: Int64 (nullable integer) with values in {0..q-1} where 0 is worst, q-1 is best, and NA where score is NaN or ineligible
- TopDecileMaskMonthly: pd.DataFrame
  - index: month-end trading days
  - columns: tickers
  - dtype: bool or object[bool] (True where rank == q-1; False otherwise)

Ranking and tie-handling:
- Cross-sectional ranks computed per month only across tickers with finite scores and universe==True.
- Sort key: primary = score (descending), secondary = ticker code (ascending) for deterministic ties.
- Rank assignment: ordinal rank r in [0..n-1]; decile = floor(r * q / n) ensures near-equal bin sizes with deterministic assignment.
- Edge cases: if n < q, bins will be imbalanced but deterministic; if n == 0, all NaN/False for that month.

[Files]
Add a new momentum module and its tests; no deletions; optionally export names in package init.

New files:
- src/bwv/momentum.py
  - Purpose: Compute J-month momentum scores with skip, cross-sectional decile ranks, and top-decile masks.
- tests/test_momentum.py
  - Purpose: TDD coverage for score windows, skip logic, ranking, ties, NaN/ineligibility handling, and decile balance.

Existing files to be modified (optional):
- src/bwv/__init__.py
  - Add convenience re-exports for momentum functions (optional; not required since tests can import bwv.momentum).
  - Change (optional):
    from .momentum import momentum_scores, decile_ranks, top_decile_mask

Configuration updates:
- None (parameters are function arguments; integration with YAML configs will occur in later milestones).

[Functions]
Introduce three new public functions and an internal helper for stable ranking.

New functions:
- Name: momentum_scores
  - Signature: momentum_scores(ret_d: pd.DataFrame, universe_mask: pd.DataFrame, J: int, skip_days: int = 5) -> pd.DataFrame
  - File: src/bwv/momentum.py
  - Purpose: Compute formation-month momentum scores as product(1+ret_d)−1 within the window [month_end_{t−J}, t−skip_days] inclusive, aligned to formation month-ends, and masked by universe eligibility.
  - Behavior:
    - Uses bwv.returns.cum_return_skip(ret_d, J, skip_days) for score backbone.
    - Sets scores to NaN where universe_mask is False or score window insufficient.
    - Ensures index and columns are aligned to ret_d and universe_mask union intersection.
- Name: decile_ranks
  - Signature: decile_ranks(scores: pd.DataFrame, q: int = 10) -> pd.DataFrame
  - File: src/bwv/momentum.py
  - Purpose: Compute per-month cross-sectional ordinal ranks and map to 0..q−1 decile bins.
  - Behavior:
    - For each formation date, select finite scores; sort by (-score, ticker).
    - Assign ordinal ranks r=0..n−1; compute decile = floor(r * q / n).
    - Return nullable Int64 DataFrame; NA where score NaN or no eligible assets.
- Name: top_decile_mask
  - Signature: top_decile_mask(ranks: pd.DataFrame, decile: int = 9) -> pd.DataFrame
  - File: src/bwv/momentum.py
  - Purpose: Boolean mask for top decile (default D10 where q=10); coerces to Python bool dtype for strict equality tests.
  - Behavior:
    - True where ranks == decile; False elsewhere; preserves index/columns.

Internal helper:
- Name: _stable_rank_order
  - Signature: _stable_rank_order(scores_row: pd.Series) -> list[str]
  - File: src/bwv/momentum.py
  - Purpose: Return ticker order sorted by (-score, ticker) for deterministic tie-breaking used by decile_ranks.

Modified functions:
- None in existing modules (bwv.returns is consumed as-is).

Removed functions:
- None.

[Classes]
No new classes are introduced in M4; functional API suffices for signal computation.

New classes:
- None.

Modified classes:
- None.

Removed classes:
- None.

[Dependencies]
No new third-party dependencies are required; leverage existing numpy and pandas.

Details:
- Uses pandas operations, numpy for numeric stability (log1p/expm1 in underlying cum product).
- No changes to pyproject.toml.

[Testing]
Adopt TDD with comprehensive unit tests for momentum computation and ranking.

Tests to create in tests/test_momentum.py:
- test_scores_window_and_skip_logic
  - Construct daily returns with known pattern over multiple months; verify momentum_scores equals product(1+ret)−1 over [t−J months .. t−skip_days], matching bwv.returns.cum_return_skip.
- test_scores_respect_universe_mask
  - Provide a universe monthly mask with some tickers False; assert those entries are NaN at corresponding formation dates.
- test_decile_ranks_deterministic_ties
  - Craft scores with ties; assert ordering by ticker code yields stable, deterministic deciles.
- test_decile_bins_approximately_balanced
  - For n divisible by q, assert exact equal counts per decile; for non-divisible, counts differ by at most 1.
- test_top_decile_mask_bool_dtype
  - Assert mask dtype is object[bool] or bool and True only where rank == q−1.
- test_nan_and_insufficient_history
  - When J history insufficient or all-NaN scores that month, ranks are NA and mask is all False.

Validation strategy:
- Reuse returns.month_end_index for alignment checks.
- No look-ahead verified implicitly through cum_return_skip and monthly universe indexing.

[Implementation Order]
Implement momentum score computation first, then ranking, then masks, followed by tests; finalize with optional package exports.

Ordered steps:
1. Implement src/bwv/momentum.py:
   - momentum_scores using returns.cum_return_skip and universe masking.
   - _stable_rank_order and decile_ranks with deterministic ticker tie-breaks and decile mapping.
   - top_decile_mask with bool coercion.
2. Create tests/test_momentum.py with the cases listed above.
3. Optionally expose momentum functions in src/bwv/__init__.py for convenience.
4. Run tests with Poetry: poetry run pytest -q (as requested).
5. Refactor if needed to pass all tests while maintaining clarity and performance.
