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
- Current default windows in run_all.py are SHORT_W = 10 and LONG_W = 50.
- Recent 10-seed robustness check for the 10/50 setting was positive across all seeds, with mean return about 3.01% and median return about 2.74%.
- A recent single run on the current processed dataset finished flat with end_equity about 103181.01, total_return about 3.18%, and 21 trades.
- Generated CSV, PNG, Parquet, and backup files are ignored by git via .gitignore.
- If Parquet support is missing, install pyarrow.
