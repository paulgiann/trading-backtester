Trading Backtester - Submission Artifacts

This folder contains the final comparison artifacts for the current local dataset used in this project.

Included files
- CURRENT_BENCHMARKS.txt
- WALK_FORWARD_SUMMARY.txt
- equity_curve_ma_current.png
- trade_sizes_ma_current.png
- equity_curve_regime_current.png
- trade_sizes_regime_current.png

What these files show
- The MA strategy is the simpler, lower-turnover baseline.
- The Regime strategy is the stronger return-oriented strategy on the current local dataset, with higher turnover.
- CURRENT_BENCHMARKS.txt summarizes the default-seed one-run comparison, the 10-seed robustness comparison, and the rolling walk-forward comparison.
- WALK_FORWARD_SUMMARY.txt gives the fold-by-fold out-of-sample walk-forward results.

Current selected regime configuration
- SHORT_W = 10
- LONG_W = 50
- SPREAD_TH = 0.0010
- BREAKOUT_Z_MIN = 0.5
- VOL_RATIO_MIN = 1.10
- RANGE_CAP = 0.020
- MAX_HOLD_HOURS = 6
- TARGET_FRAC = 0.35
- COOLDOWN_MIN = 3

Selection rationale
- Among the tested tuning levers, SPREAD_TH was the only one that materially improved the regime strategy on this dataset.
- SPREAD_TH = 0.0010 gave the best robust trade-off between return quality and stability across seeds.
- Nearby alternatives such as 0.0009, 0.00105, and 0.0011 produced at least one negative seed in robustness testing, so they were not selected.

Interpretation
- The MA strategy is preferable when simplicity, lower turnover, and smaller drawdown are prioritized.
- The Regime strategy is preferable when stronger return on the current dataset is prioritized and higher turnover is acceptable.
- The walk-forward results also favor the Regime strategy over the MA baseline on the current dataset.
- Both final compared strategies finish flat at the end of the run.

Notes
- Results in this folder are tied to the current local dataset and should be interpreted as project backtest results, not live trading expectations.
- Figures were exported at 300 DPI for cleaner presentation.
