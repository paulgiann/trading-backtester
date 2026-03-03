# trading-backtester

A Python trading backtester with market-data download, feature preparation, event-driven backtesting, strategy comparison, robustness testing, and an Alpaca-ready paper-trading extension point.

## Main workflow

From the repo root, run:

    python download_binance.py
    python prepare_data.py
    python run_all.py

## Quick checks

Run the baseline checks with:

    python self_test.py

Run the test suite with:

    .\.venv\Scripts\python -m pytest .\tests -q

Run robustness testing with:

    python seed_robustness.py

Test the Alpaca paper-trading placeholder with:

    python alpaca_paper.py

## Reproducible example runs

Default MA baseline smoke test:

    $env:SKIP_DOWNLOAD="1"; $env:SHOW_PLOTS="0"; python .\run_all.py

Selected Regime configuration used for the current final comparison:

    $env:SKIP_DOWNLOAD="1"; $env:SHOW_PLOTS="0"; $env:STRATEGY_NAME="regime"; $env:SPREAD_TH="0.0010"; $env:BREAKOUT_Z_MIN="0.5"; $env:VOL_RATIO_MIN="1.10"; $env:RANGE_CAP="0.020"; $env:MAX_HOLD_HOURS="6"; $env:TARGET_FRAC="0.35"; $env:COOLDOWN_MIN="3"; python .\run_all.py

## Data and artifact layout

- data/raw/market_data.csv - downloaded raw Binance intraday data
- data/processed/features.csv - processed feature dataset
- data/processed/features.parquet - processed feature dataset in Parquet format
- data/processed/orders_audit_run_all.csv - audit log from run_all.py
- artifacts/submission/ - current benchmark summary and clean-named submission plots

## Current project summary

The project currently compares a simpler MA baseline against a more selective Regime strategy on the current local dataset.

MA baseline
- lower-turnover benchmark
- current robust baseline setting uses SHORT_W = 10, LONG_W = 50, SPREAD_TH = 0.0005
- 10-seed robustness on the current local dataset:
  - mean_return ≈ 0.03585064
  - median_return ≈ 0.03131081
  - mean_trades ≈ 15.5
  - all seeds positive
  - all seeds flat at end

Regime strategy
- stronger return-oriented strategy on the current local dataset
- current selected robust setting:
  - SHORT_W = 10
  - LONG_W = 50
  - SPREAD_TH = 0.0010
  - BREAKOUT_Z_MIN = 0.5
  - VOL_RATIO_MIN = 1.10
  - RANGE_CAP = 0.020
  - MAX_HOLD_HOURS = 6
  - TARGET_FRAC = 0.35
  - COOLDOWN_MIN = 3
- 10-seed robustness on the current local dataset:
  - mean_return ≈ 0.06329184
  - median_return ≈ 0.06355183
  - mean_trades ≈ 65.1
  - all seeds positive
  - all seeds flat at end

Default-seed one-run comparison on current data
- MA:
  - end_equity ≈ 103348.2156
  - total_return ≈ 0.03348216
  - num_trades = 15
  - max_drawdown ≈ -0.01967336
- Regime:
  - end_equity ≈ 104268.7120
  - total_return ≈ 0.04268712
  - num_trades = 62
  - max_drawdown ≈ -0.03466817

## Important implementation notes

- run_all.py is the main backtester and supports env overrides for strategy selection and tuning.
- STRATEGY_NAME selects between the MA baseline and the Regime strategy.
- SHOW_PLOTS=0 keeps testing quiet while preserving saved figures.
- Saved figures are exported at 300 DPI.
- Forced final flatten ensures runs finish flat cleanly.
- The engine now preserves the last nonzero signal across neutral bars.
- Residual remainders after partial fills are canceled to avoid unintended later fills.
- The regime strategy uses stricter breakout confirmation than earlier versions.
- seed_robustness.py provides reusable environment-driven robustness checks.
- alpaca_paper.py provides a paper-only Alpaca configuration and gateway placeholder for the course extension.
- The Alpaca extension is intentionally restricted to the paper endpoint and is not designed for real-money trading.
- Generated CSV, PNG, Parquet, and backup files are ignored by git via .gitignore.
- If Parquet support is missing, install pyarrow.

## Submission-oriented note

The cleanest summary of the final current results is in:

- artifacts/submission/README.txt
- artifacts/submission/CURRENT_BENCHMARKS.txt
