import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone

BASE = "https://api.binance.com/api/v3/klines"

def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def fetch(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000):
    r = requests.get(
        BASE,
        params={"symbol": symbol, "interval": interval, "startTime": start_ms, "endTime": end_ms, "limit": limit},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

def main():
    symbol = "SOLUSDT"
    interval = "1m"
    days = 7
    out_csv = Path("data/raw/market_data.csv")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    cur = to_ms(start)
    end_ms = to_ms(end)

    rows = []
    while cur < end_ms:
        kl = fetch(symbol, interval, cur, end_ms, limit=1000)
        if not kl:
            break

        for k in kl:
            open_time_ms = int(k[0])
            dt = datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc).isoformat()
            rows.append({
                "Datetime": dt,
                "Open": float(k[1]),
                "High": float(k[2]),
                "Low": float(k[3]),
                "Close": float(k[4]),
                "Volume": float(k[5]),
            })

        cur = int(kl[-1][0]) + 1
        time.sleep(0.15)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No data returned. Check symbol/interval.")

    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
    df = df.drop_duplicates(subset=["Datetime"]).sort_values("Datetime")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"Saved {out_csv} with {len(df)} rows for {symbol} ({interval}), ~{days} days.")

if __name__ == "__main__":
    main()
