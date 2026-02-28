from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from io_utils import save_dual


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df columns: Datetime, Open, High, Low, Close, Volume (or Datetime as index).
    Output adds engineered features for strategy + ML.
    """
    out = df.copy()

    # Ensure datetime index
    if "Datetime" in out.columns:
        out["Datetime"] = pd.to_datetime(out["Datetime"], utc=True, errors="coerce")
        out = out.dropna(subset=["Datetime"])
        out = out.set_index("Datetime")
    out = out[~out.index.duplicated(keep="first")]
    out = out.sort_index()

    # Standardize column names (common issues)
    cols = {c: c.strip().title() for c in out.columns}
    out = out.rename(columns=cols)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(out.columns)}")

    out = out.dropna(subset=required)

    # Core returns
    out["ret_1"] = out["Close"].pct_change()
    out["ret_5"] = out["Close"].pct_change(5)
    out["ret_15"] = out["Close"].pct_change(15)

    # Moving averages
    out["ma_s"] = out["Close"].rolling(12, min_periods=12).mean()
    out["ma_l"] = out["Close"].rolling(60, min_periods=60).mean()
    out["ma_spread"] = (out["ma_s"] - out["ma_l"]) / out["Close"]

    # Volatility (realized)
    out["vol_30"] = out["ret_1"].rolling(30, min_periods=30).std()
    out["vol_120"] = out["ret_1"].rolling(120, min_periods=120).std()

    # Range / ATR-like proxy (intraday)
    out["hl_range"] = (out["High"] - out["Low"]) / out["Close"].replace(0, np.nan)
    out["range_30"] = out["hl_range"].rolling(30, min_periods=30).mean()

    # Volume features
    out["volm_30"] = out["Volume"].rolling(30, min_periods=30).mean()
    out["volm_ratio"] = out["Volume"] / out["volm_30"].replace(0, np.nan)

    # Mean reversion signal ingredients
    out["px_mean_60"] = out["Close"].rolling(60, min_periods=60).mean()
    out["px_std_60"] = out["Close"].rolling(60, min_periods=60).std()
    out["z_px_60"] = (out["Close"] - out["px_mean_60"]) / out["px_std_60"].replace(0, np.nan)

    # Breakout features
    out["roll_max_60"] = out["Close"].rolling(60, min_periods=60).max()
    out["roll_min_60"] = out["Close"].rolling(60, min_periods=60).min()
    out["breakout_up"] = (out["Close"] >= out["roll_max_60"]).astype(int)
    out["breakout_dn"] = (out["Close"] <= out["roll_min_60"]).astype(int)

    # Clean NaNs at the start
    out = out.dropna()

    return out


def main():
    raw_csv = Path("data/raw/market_data.csv")
    if not raw_csv.exists():
        raise FileNotFoundError("Expected data/raw/market_data.csv. Run download_binance.py first.")

    df = pd.read_csv(raw_csv)
    feats = compute_features(df)

    save_dual(
        feats.reset_index(),
        csv_path="data/processed/features.csv",
        parquet_path="data/processed/features.parquet",
        index=False,
    )

    print(f"Saved processed features: rows={len(feats):,}, cols={feats.shape[1]}")


if __name__ == "__main__":
    main()
