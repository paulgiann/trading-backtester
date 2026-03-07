from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from prepare_data import compute_features
from strategies import RegimeAwareStrategy, MACrossoverStrategy


def decide_live_action(
    *,
    signal: int,
    spread_ok: bool,
    latest_datetime,
    latest_close: float,
    raw_volume: float,
    raw_trade_count: float,
    bar_age_seconds: float,
    max_bar_age_seconds: int,
) -> str:
    if bar_age_seconds > max_bar_age_seconds:
        return (
            f"decision=BLOCK reason=stale_bar latest_datetime={latest_datetime} "
            f"bar_age_seconds={bar_age_seconds:.3f} max_bar_age_seconds={max_bar_age_seconds}"
        )

    if raw_volume == 0.0 or raw_trade_count == 0.0:
        return f"decision=BLOCK reason=zero_liquidity latest_datetime={latest_datetime}"

    if not spread_ok:
        return f"decision=HOLD reason=spread_filter latest_datetime={latest_datetime}"

    if signal > 0:
        return f"decision=BUY latest_datetime={latest_datetime} close={latest_close}"

    if signal < 0:
        return f"decision=SELL latest_datetime={latest_datetime} close={latest_close}"

    return f"decision=HOLD reason=no_signal latest_datetime={latest_datetime}"


def build_live_signal_context() -> dict:
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame

    api_key = os.environ["ALPACA_API_KEY"]
    api_secret = os.environ["ALPACA_API_SECRET"]

    symbol = os.getenv("ALPACA_SYMBOL", "SOL/USD")
    lookback_minutes = int(os.getenv("ALPACA_LOOKBACK_MINUTES", "500"))
    max_bar_age_seconds = int(os.getenv("ALPACA_MAX_BAR_AGE_SECONDS", "90"))

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
    req = CryptoBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
    )

    raw = client.get_crypto_bars(req).df.reset_index()
    if raw.empty:
        raise SystemExit("decision=BLOCK reason=no_bars")

    raw_latest = raw.iloc[-1].copy()
    latest_bar_ts = raw_latest["timestamp"]
    bar_age_seconds = (end - latest_bar_ts).total_seconds()

    bars = raw.rename(columns={
        "timestamp": "Datetime",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })

    feats = compute_features(bars[["Datetime", "Open", "High", "Low", "Close", "Volume"]]).reset_index()
    latest = feats.iloc[-1]
    close = feats.set_index("Datetime")["Close"]

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

    action_line = decide_live_action(
        signal=int(signal),
        spread_ok=bool(spread_ok),
        latest_datetime=latest["Datetime"],
        latest_close=float(latest["Close"]),
        raw_volume=float(raw_latest["volume"]),
        raw_trade_count=float(raw_latest["trade_count"]),
        bar_age_seconds=float(bar_age_seconds),
        max_bar_age_seconds=max_bar_age_seconds,
    )

    return {
        "symbol": symbol,
        "strategy_name": strategy_name,
        "latest_datetime": latest["Datetime"],
        "latest_close": float(latest["Close"]),
        "signal": int(signal),
        "spread_ok": bool(spread_ok),
        "raw_volume": float(raw_latest["volume"]),
        "raw_trade_count": float(raw_latest["trade_count"]),
        "bar_age_seconds": float(bar_age_seconds),
        "max_bar_age_seconds": int(max_bar_age_seconds),
        "action_line": action_line,
    }


def append_live_log_row(row: dict, log_path: str | Path) -> Path:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([row])
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)
    return path
