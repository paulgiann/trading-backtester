from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from prepare_data import compute_features
from strategies import RegimeAwareStrategy, MACrossoverStrategy


def main():
    try:
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError:
        raise SystemExit("alpaca-py is not installed in this Python environment.")

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    symbol = os.getenv("ALPACA_SYMBOL", "SOL/USD")
    lookback_minutes = int(os.getenv("ALPACA_LOOKBACK_MINUTES", "500"))

    strategy_name = os.getenv("STRATEGY_NAME", "regime")
    short_w = int(os.getenv("SHORT_W", "10"))
    long_w = int(os.getenv("LONG_W", "50"))
    target_frac = float(os.getenv("TARGET_FRAC", "0.35"))
    breakout_z_min = float(os.getenv("BREAKOUT_Z_MIN", "0.5"))
    vol_ratio_min = float(os.getenv("VOL_RATIO_MIN", "1.10"))
    range_cap = float(os.getenv("RANGE_CAP", "0.020"))
    spread_th = float(os.getenv("SPREAD_TH", "0.0010"))

    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=lookback_minutes)

    client = CryptoHistoricalDataClient(api_key=api_key, secret_key=api_secret)

    request = CryptoBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
    )

    bars = client.get_crypto_bars(request).df.reset_index()

    if bars.empty:
        raise SystemExit("No Alpaca bars returned.")

    raw_latest = bars.iloc[-1].copy()

    bars = bars.rename(
        columns={
            "timestamp": "Datetime",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )

    keep = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    bars = bars[keep]

    feats = compute_features(bars)

    if feats.empty:
        raise SystemExit("Feature computation returned no rows.")

    latest = feats.reset_index().iloc[-1]
    close = feats["Close"]

    if strategy_name == "regime":
        st = RegimeAwareStrategy(
            short_w,
            long_w,
            target_frac,
            breakout_z_min=breakout_z_min,
            vol_ratio_min=vol_ratio_min,
            range_cap=range_cap,
        )
        signal = st.signal(
            close=close,
            z_px_60=float(latest["z_px_60"]),
            breakout_up=int(latest["breakout_up"]),
            breakout_dn=int(latest["breakout_dn"]),
            volm_ratio=float(latest["volm_ratio"]),
            range_30=float(latest["range_30"]),
        )
    else:
        st = MACrossoverStrategy(short_w, long_w, target_frac)
        signal = st.signal(close)

    spread_ok = st.spread_ok(close, float(latest["Close"]), spread_th)

    print("symbol:", symbol)
    print("bars_loaded:", len(bars))
    print("feature_rows:", len(feats))
    print("latest_datetime:", latest["Datetime"])
    print("latest_close:", latest["Close"])
    print("latest_ma_spread:", latest["ma_spread"])
    print("latest_z_px_60:", latest["z_px_60"])
    print("latest_breakout_up:", latest["breakout_up"])
    print("latest_breakout_dn:", latest["breakout_dn"])
    print("latest_volm_ratio:", latest["volm_ratio"])
    print("latest_range_30:", latest["range_30"])
    print("latest_raw_volume:", raw_latest["volume"])
    print("latest_raw_trade_count:", raw_latest["trade_count"])
    print("strategy_name:", strategy_name)
    print("signal:", signal)
    print("spread_ok:", spread_ok)

    if float(raw_latest["volume"]) == 0.0 or float(raw_latest["trade_count"]) == 0.0:
        print("warning: latest Alpaca bar has zero volume/trade_count; live execution should be blocked on this bar.")


if __name__ == "__main__":
    main()
