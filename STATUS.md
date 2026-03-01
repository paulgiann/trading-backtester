Project status

Folder: C:\Users\Pavlos\Desktop\trading-backtester
Main file: run_all.py
Asset: SOLUSDT 1m ~7 days

Notes:
- self_test.py passes (OrderBook / OrderManager) using stdlib unittest.
- Raw data is written to data/raw/market_data.csv.
- Processed features are written to data/processed/features.csv and data/processed/features.parquet.
- run_all.py is the main backtester and writes data/processed/orders_audit_run_all.csv.
- project_backtester.py also runs on processed data and now finishes flat with consistent reporting.
- Current committed default in run_all.py is SHORT_W = 10 and LONG_W = 50.
- Recent single run on the current processed dataset finished flat with end_equity about 103181.01, total_return about 3.18%, and 21 trades.
- A 10-seed robustness check for 10/50 was positive for all 10 seeds, with mean return about 3.01% and median return about 2.74%.
- Earlier 8/40 testing was not robust enough and underperformed the 10/50 setting on the same seed check.
- Generated CSV, PNG, Parquet, and backup files are ignored by git.
