# Implementation Plan

[Overview]
Design and integrate a new ingestion pipeline that reads per-ticker OHLCV parquet files under HSX/ and HNX/ and daily index CSVs under vn_indices/, normalizes to the project's canonical VND schema, and plugs into the existing cleaning, eligibility, and backtesting flow.

This plan adds directory-based parquet ingestion alongside the current CSV ingest, preserving existing outputs and contracts so downstream modules (filters, returns, momentum, backtest) work unchanged. The pipeline will (a) load and validate OHLCV across both exchanges, scaling prices from kVND to VND, (b) produce the canonical MultiIndex (date, ticker) DataFrame, (c) emit data/clean artifacts identical in shape to the current CSV-based process, and (d) optionally prepare benchmark indices from vn_indices/ for backtest/report usage. We will extend scripts/make_clean.py to accept parquet directory inputs while keeping CSV support, and add a couple of minimal data_io helpers for reuse and testability.

[Types]  
Extend the ingestion type surface to include parquet directory inputs while keeping canonical internal types unchanged.

- Input: Parquet files per ticker under HSX/ and HNX/
  - File path pattern: HSX/*.parquet, HNX/*.parquet
  - Each file contains a single ticker
  - Columns (required): time, open, high, low, close, volume
    - time: string or date, ISO (YYYY-MM-DD)
    - open/high/low/close: float, in kVND (thousands of VND)
    - volume: integer or float (shares)
- Canonical OHLCV DataFrame (unchanged contract)
  - Index: MultiIndex (date: pandas.Timestamp, ticker: str uppercase)
  - Columns: ["open", "high", "low", "close", "volume?" (nullable Float64), "value?" (float), "exchange?" (str)]
  - Units:
    - Prices in VND (ingestion scales kVND × price_scale where default price_scale=1000.0)
    - Volume in shares
    - value (optional) in VND; if absent, downstream trading_value uses close × volume
- Canonical Index Series/DataFrame
  - Input: vn_indices/*.csv with columns: time, open, high, low, close, volume (kVND)
  - Output for benchmark:
    - Either Series[monthly returns] if single index (e.g., VNINDEX) or DataFrame[monthly returns] for multiple tickers
    - Returns are simple percent monthly (product of (1 + daily) - 1)
  - Alternatively, canonical price Series/DataFrame in VND can be returned for alignment or diagnostics

Validation rules:
- Required OHLC present, all prices scaled to VND > 0, high >= low, open/close within [low, high]
- No duplicate (date, ticker)
- Index files must parse time; close must be present for price-based monthly aggregation

[Files]
Add parquet directory ingestion and optional benchmark preparation, by modifying existing script(s) and minimally extending shared utilities.

- New/Modified files:
  - src/bwv/data_io.py (modify)
    - Add:
      - load_ohlcv_parquet_dirs(paths: Iterable[str|Path], start=None, end=None, add_exchange_col=True, price_scale=1000.0) -> pd.DataFrame
        - Walk directories (non-recursive by default) and read *.parquet
        - Infer ticker from filename stem; infer exchange from parent folder name (HSX/HNX)
        - Normalize columns: lower-case, strip, map "time"->"date", enforce canonical, scale kVND->VND
        - Return canonical MultiIndex DataFrame sorted by (date, ticker)
      - load_indices_from_dir(dir_path: str|Path, names: list[str], price_field="close", price_scale=1000.0) -> pd.DataFrame
        - Read vn_indices/*.csv, select tickers, return wide DataFrame of prices in VND aligned by date
    - Keep existing load_ohlcv/load_index public APIs; new functions complement them.
  - scripts/make_clean.py (modify)
    - Extend CLI to accept parquet directory inputs:
      - --inputs-dir HSX HNX (or any dirs)
      - Existing --inputs for CSV still supported
    - Detect input types:
      - If any directory provided: use load_ohlcv_parquet_dirs on those; union with CSV load if also given
    - Optional: --indices-dir vn_indices and --indices VNINDEX VN30 HNXINDEX to materialize monthly benchmark returns CSV
    - Keep outputs unchanged:
      - data/clean/ohlcv.parquet (or CSV fallback)
      - data/clean/universe_monthly.csv
      - Optional new output if indices requested:
        - data/clean/indices_monthly.csv (wide month-end simple returns)
  - config/data.yml (modify)
    - Add optional keys:
      - inputs_parquet_dirs: ["HSX", "HNX"]
      - indices:
        - dir: "vn_indices"
        - tickers: ["VNINDEX", "VN30", "HNXINDEX"]
      - outputs:
        - indices_monthly_csv: "data/clean/indices_monthly.csv"
    - Keep existing keys/semantics backward compatible
  - README.md (modify)
    - Document new parquet ingestion usage and optional indices export
- No files deleted or moved at this stage.

[Functions]
Add minimal new data_io helpers and extend make_clean CLI paths; preserve existing contracts to avoid downstream changes.

- New functions
  - src/bwv/data_io.py
    - load_ohlcv_parquet_dirs(
        paths: Iterable[str|Path],
        start: Optional[str|Timestamp] = None,
        end: Optional[str|Timestamp] = None,
        *,
        add_exchange_col: bool = True,
        price_scale: float = 1000.0
      ) -> pd.DataFrame
      Purpose: Load per-ticker parquet files in HSX/HNX directories; normalize to canonical VND OHLCV (MultiIndex).
    - load_indices_from_dir(
        dir_path: str|Path,
        names: list[str],
        *,
        price_field: str = "close",
        price_scale: float = 1000.0
      ) -> pd.DataFrame
      Purpose: Load specified indices from vn_indices/*.csv into VND price table (wide, index=date).
- Modified functions
  - scripts/make_clean.py: main(), parse_args()
    - parse_args(): add --inputs-dir, --indices-dir, --indices (list), and --out-indices-monthly
    - main():
      - Resolve config + args
      - If inputs_dir present: call load_ohlcv_parquet_dirs; if inputs (CSV) also present: also load_ohlcv and concat
      - Proceed with current pre-clean + validate + write parquet logic (unchanged)
      - Compose universe (unchanged)
      - If indices requested: build monthly returns and write indices_monthly_csv
- Removed functions
  - None; keep all existing to maintain compatibility.

[Classes]
No new classes; functional extensions only.

- New classes: None
- Modified classes: None
- Removed classes: None

[Dependencies]
No new external dependencies required; pyarrow/fastparquet already present for parquet IO.

- Ensure pyarrow and/or fastparquet are available (already in pyproject)
- No version changes needed
- Continue using pandas for CSV IO and parquet write fallback remains as in make_clean.py

[Testing]
Extend unit and integration coverage for parquet ingestion and optional indices export.

- New tests
  - tests/test_ingest_parquet.py
    - Create tmp HSX/HNX dirs with small parquet fixtures
    - Validate:
      - load_ohlcv_parquet_dirs returns canonical MultiIndex with scaled VND prices
      - Ticker inferred from file stem, exchange inferred from folder
      - Deduplication and OHLC sanity consistent with CSV ingest pre-clean path
  - tests/test_indices_dir.py
    - Create tmp vn_indices dir with small CSV fixtures (time, open, high, low, close, volume)
    - Validate:
      - load_indices_from_dir returns wide VND price table for requested tickers
      - Monthly returns construction in make_clean optional path outputs expected CSV (content shape, index)
- Existing tests (unchanged, still passing)
  - tests/test_data_io.py
  - tests/test_filters.py and test_filters_scaling.py
  - tests/test_returns.py, etc.
- Validation strategy
  - Compare a small ticker’s CSV vs parquet ingestion parity (same VND prices and dates)
  - Ensure filters.compose_universe works unchanged on parquet-ingested OHLCV
  - If indices export enabled, ensure backtest runner can align provided benchmark easily

[Implementation Order]
Implement helpers first, then CLI wiring, then optional benchmark handling, followed by tests and docs.

1. src/bwv/data_io.py: Implement load_ohlcv_parquet_dirs and load_indices_from_dir with normalization and scaling.
2. config/data.yml: Add optional inputs_parquet_dirs and indices config keys; keep old keys intact.
3. scripts/make_clean.py: Extend CLI (parse_args) and main logic to support --inputs-dir and optional indices export, preserving existing outputs.
4. Optional: emit data/clean/indices_monthly.csv when indices config present or flags provided.
5. Tests: Add test_ingest_parquet.py and test_indices_dir.py; ensure full suite passes.
6. README.md: Document parquet ingestion usage, indices export flags, and examples.
7. Sanity run: poetry run python scripts/make_clean.py using {HSX,HNX} to produce ohlcv.parquet/universe CSV; then run a sample backtest.
8. Review/perf: confirm memory/perf acceptable on dataset size; minor batching if needed (not expected initially).
