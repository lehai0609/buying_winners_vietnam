# buying-winners-vietnam (M0: Environment & Repro)

Minimal project scaffold for Milestone M0: reproducible environment (Python 3.11) and deterministic tests.

Quickstart (Windows 11)

1. Verify Python 3.11 is available:
   - py -3.11 --version

   If missing, install via winget:
   - winget install --id Python.Python.3.11 -e

2. Install pipx (using Python 3.11) and Poetry:
   - py -3.11 -m pip install --user pipx
   - py -3.11 -m pipx ensurepath
   - (Close and re-open your terminal or re-login if PATH changes are required)
   - py -3.11 -m pipx install poetry
   - poetry --version

3. Use Poetry to create an in-project virtualenv and install dependencies:
   - poetry env use 3.11
   - poetry install

   This repository is configured to create `.venv` inside the project root (see `poetry.toml`).

4. Run tests:
   - poetry run pytest -q

Activating the in-project virtualenv (optional)
- PowerShell:
  - .\.venv\Scripts\Activate.ps1
- CMD:
  - .venv\Scripts\activate.bat

Notes
- Python is pinned to 3.11.* in `pyproject.toml`.
- Only `pytest` is added as a development dependency for M0.
- If `poetry install` fails complaining about a missing README, ensure this file exists (it is included in this repository).

---

M2: Vietnam-specific eligibility and VND unit conventions

Overview
- Goal: Implement Vietnam-specific eligibility and cleaning so the monthly investable universe (HOSE/HNX) is derived without look-ahead and fully in VND units.
- Prices in vendor CSVs are quoted in thousands of VND (kVND). The ingest scales to VND.
- Rules enforced (defaults):
  - Minimum trading history: 126 trading days
  - Price floor: ≥ 1,000 VND
  - ADV: ≥ 100,000,000 VND with 60-day rolling window
  - Non-trading days: ≤ 15 over trailing 126 days
- Outputs:
  - data/clean/ohlcv.parquet: OHLCV panel with prices scaled to VND
  - data/clean/universe_monthly.csv: boolean matrix [month_end × ticker] for eligibility

Units and conventions
- Price unit: VND (ingest multiplies kVND prices by 1,000)
- Volume unit: shares
- Trading value: VND (price[VND] × volume[shares])

Configuration
- config/data.yml centralizes defaults and I/O paths:
  - price_scale: 1000.0
  - min_history_days: 126
  - min_price_vnd: 1000.0
  - min_adv_vnd: 100000000.0
  - adv_window_days: 60
  - max_non_trading_days: 15
  - ntd_window_days: 126
  - inputs: [HSX.csv, HNX.csv]
  - outputs:
    - ohlcv_parquet: data/clean/ohlcv.parquet
    - universe_monthly_csv: data/clean/universe_monthly.csv

How to generate cleaned artifacts
- Place HSX.csv and HNX.csv in the project root (or update config/paths accordingly).
- Run the cleaning script (will normalize headers, scale prices to VND, validate, and emit outputs):
  - poetry run python scripts/make_clean.py
- Optional overrides:
  - poetry run python scripts/make_clean.py --inputs HSX.csv HNX.csv --out-parquet data/clean/ohlcv.parquet --out-universe data/clean/universe_monthly.csv
  - poetry run python scripts/make_clean.py --min-adv-vnd 150000000 --adv-window-days 40

Implementation notes
- src/bwv/data_io.py:
  - load_ohlcv(..., price_scale=1000.0): scales OHLC to VND on ingest; parses and normalizes columns.
  - load_index(..., price_scale=1000.0): scales selected price field to VND.
  - validate_ohlcv(df): conservative sanity checks and flags.
- src/bwv/filters.py:
  - trading_value: prefers 'value' column if present; otherwise close[VND] × volume.
  - min_history, price_floor, adv_filter, non_trading_days_filter: all shifted by one trading day to avoid look-ahead.
  - compose_universe(..., monthly=True): logical AND of masks; sampled at last trading day of each month.
- Tests:
  - tests/test_data_io.py: asserts scaling behavior and I/O normalization.
  - tests/test_filters.py: verifies no-look-ahead and composition mechanics.
  - tests/test_filters_scaling.py: ensures ADV thresholds behave correctly when ingesting kVND prices.

---

M3: Parquet ingestion and indices export (HSX/HNX directories + vn_indices)

New capabilities
- Ingest per-ticker OHLCV parquet files from HSX/ and HNX/ directories, normalizing to the canonical schema and scaling prices from kVND to VND.
- Optionally ingest daily index CSVs from vn_indices/ and export monthly simple returns for selected indices.

CLI usage examples
- From CSV inputs (legacy):
  - poetry run python scripts/make_clean.py --inputs HSX.csv HNX.csv
- From parquet directories (HSX/ and HNX/ at repo root):
  - poetry run python scripts/make_clean.py --inputs-dir HSX HNX
- Mixed CSV + parquet:
  - poetry run python scripts/make_clean.py --inputs HSX.csv HNX.csv --inputs-dir HSX HNX
- Optional indices monthly returns export (e.g., VNINDEX, HNX-INDEX, VN30):
  - poetry run python scripts/make_clean.py --indices-dir vn_indices --indices VNINDEX HNX-INDEX VN30 --out-indices-monthly data/clean/indices_monthly.csv

Config keys
- You can also set these in config/data.yml:
  - inputs_parquet_dirs: ["HSX", "HNX"]
  - indices:
    - dir: "vn_indices"
    - tickers: ["VNINDEX", "HNX-INDEX", "VN30"]
  - outputs:
    - indices_monthly_csv: "data/clean/indices_monthly.csv"

Implementation notes (parquet + indices)
- src/bwv/data_io.py:
  - load_ohlcv_parquet_dirs(dirs, start=None, end=None, price_scale=1000.0, add_exchange_col=True) -> canonical MultiIndex (date, ticker)
  - load_indices_from_dir(dir, names, price_field="close", price_scale=1000.0) -> wide VND price table (index=date)
- scripts/make_clean.py:
  - Adds --inputs-dir for parquet folders and optional --indices-dir/--indices/--out-indices-monthly
  - Still produces data/clean/ohlcv.parquet and data/clean/universe_monthly.csv
  - If indices options provided, writes data/clean/indices_monthly.csv with monthly simple returns (using bwv.returns.month_returns_from_daily)

Backtest example
- After generating cleaned artifacts:
  - poetry run python scripts/run_backtest.py --J 12 --K 3 --skip-days 5 --ohlcv data/clean/ohlcv.parquet --universe data/clean/universe_monthly.csv --out results/runs/J12K3
- Optional: provide a daily benchmark CSV (e.g., vn_indices/VNINDEX.csv):
  - poetry run python scripts/run_backtest.py --J 12 --K 3 --skip-days 5 --benchmark vn_indices/VNINDEX.csv --benchmark-ticker VNINDEX --out results/runs/J12K3
