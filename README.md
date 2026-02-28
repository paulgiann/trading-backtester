# trading-backtester

A Python trading backtester with market-data download, feature preparation, and event-driven backtesting.

## Main workflow

From the repo root, run:

    python download_binance.py
    python prepare_data.py
    python run_all.py

## Quick checks

Run the baseline tests with:

    python self_test.py

## Data layout

- data/raw/market_data.csv - downloaded raw Binance intraday data
- data/processed/features.csv - processed feature dataset
- data/processed/features.parquet - processed feature dataset in Parquet format
- data/processed/orders_audit_run_all.csv - audit log from run_all.py

## Notes

- run_all.py is the main backtester.
- Generated CSV, PNG, Parquet, and backup files are ignored by git via .gitignore.
- If Parquet support is missing, install pyarrow.
