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
- Generated CSV, PNG, Parquet, and backup files are ignored by git.
