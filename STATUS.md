Project status

Folder: C:\Users\Pavlos\Desktop\trading-backtester
Main file: run_all.py
Asset: SOLUSDT 1m ~7 days

Current state
- The repo is on main, cleanly organized, and pushed.
- The project now compares a simpler MA baseline against a more selective Regime strategy on the current local dataset.
- Core assignment components are implemented: data download, feature preparation, gateway-style ingestion, order book, order manager, matching engine simulation, strategy execution, reporting, and tests.
- Alpaca support is included as a paper-only extension point through alpaca_paper.py.

Current selected strategy settings
- MA baseline:
  - SHORT_W = 10
  - LONG_W = 50
  - SPREAD_TH = 0.0005
- Regime strategy:
  - SHORT_W = 10
  - LONG_W = 50
  - SPREAD_TH = 0.0010
  - BREAKOUT_Z_MIN = 0.5
  - VOL_RATIO_MIN = 1.10
  - RANGE_CAP = 0.020
  - MAX_HOLD_HOURS = 6
  - TARGET_FRAC = 0.35
  - COOLDOWN_MIN = 3

Current benchmark summary
- MA baseline 10-seed robustness:
  - mean_return ≈ 0.03585064
  - median_return ≈ 0.03131081
  - mean_trades ≈ 15.5
  - all seeds positive
  - all seeds flat at end
- Regime strategy 10-seed robustness:
  - mean_return ≈ 0.06329184
  - median_return ≈ 0.06355183
  - mean_trades ≈ 65.1
  - all seeds positive
  - all seeds flat at end

Default-seed one-run comparison
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

Important implementation notes
- run_all.py is the main backtester.
- Raw data is written to data/raw/market_data.csv.
- Processed features are written to data/processed/features.csv and data/processed/features.parquet.
- run_all.py writes data/processed/orders_audit_run_all.csv.
- The engine preserves the last nonzero signal across neutral bars.
- Residual remainders after partial fills are canceled to avoid unintended later fills.
- The regime strategy uses stricter breakout confirmation than earlier versions.
- Forced final flatten ensures runs finish flat cleanly.
- SHOW_PLOTS supports quiet testing and plot generation on demand.
- Generated CSV, PNG, Parquet, and backup files are ignored by git.

Reference
- The clearest final benchmark summary is in artifacts/submission/README.txt and artifacts/submission/CURRENT_BENCHMARKS.txt.
