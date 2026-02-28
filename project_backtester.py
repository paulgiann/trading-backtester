from __future__ import annotations

import csv
import os
import uuid
import random
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Part 1: Clean + features
# ----------------------------

def clean_and_engineer_features(
    csv_path: str,
    tz: Optional[str] = None,
    add_features: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"Datetime", "Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["Datetime", "Open", "High", "Low", "Close", "Volume"])
    df = df.drop_duplicates(subset=["Datetime"], keep="last")
    df = df.set_index("Datetime").sort_index()

    if tz is not None:
        df = df.tz_convert(tz)

    if add_features:
        df["ret_1"] = df["Close"].pct_change()
        df["logret_1"] = np.log(df["Close"]).diff()

        for w in (5, 20, 50):
            df[f"sma_{w}"] = df["Close"].rolling(window=w, min_periods=w).mean()

        df["vol_20"] = df["logret_1"].rolling(window=20, min_periods=20).std(ddof=1) * math.sqrt(20)
        df = df.dropna()

    return df


# ----------------------------
# Strategy
# ----------------------------

class MovingAverageCrossoverStrategy:
    def __init__(self, short_window: int = 10, long_window: int = 30, target_risk_fraction: float = 0.20):
        if short_window <= 0 or long_window <= 0:
            raise ValueError("Windows must be positive.")
        if short_window >= long_window:
            raise ValueError("Use short_window < long_window.")
        if not (0.0 < target_risk_fraction <= 1.0):
            raise ValueError("target_risk_fraction must be in (0,1].")
        self.short_window = short_window
        self.long_window = long_window
        self.target_risk_fraction = target_risk_fraction
        self.last_signal = 0

    def generate_signal(self, data_window: pd.DataFrame) -> int:
        if len(data_window) < self.long_window:
            return 0
        close = data_window["Close"]
        sma_s = close.rolling(self.short_window).mean().iloc[-1]
        sma_l = close.rolling(self.long_window).mean().iloc[-1]
        if np.isnan(sma_s) or np.isnan(sma_l):
            return 0
        if sma_s > sma_l:
            return 1
        if sma_s < sma_l:
            return -1
        return 0

    def desired_position_shares(self, equity: float, price: float, signal: int) -> int:
        if signal == 0 or price <= 0:
            return 0
        notional = self.target_risk_fraction * equity
        shares = int(notional / price)
        return signal * shares


# ----------------------------
# Part 2: Core objects
# ----------------------------

@dataclass
class MarketTick:
    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


class Gateway:
    def __init__(self, market_df: pd.DataFrame, order_log_path: str = "orders_audit.csv"):
        if market_df is None or market_df.empty:
            raise ValueError("market_df is empty.")
        self.df = market_df.copy()
        self.order_log_path = order_log_path
        self._init_audit_file()

    def _init_audit_file(self) -> None:
        if not os.path.exists(self.order_log_path):
            with open(self.order_log_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "event_time_utc",
                        "event_type",
                        "order_id",
                        "side",
                        "qty",
                        "limit_price",
                        "status",
                        "filled_qty",
                        "avg_fill_price",
                        "note",
                    ]
                )

    def log_order_event(
        self,
        event_type: str,
        order_id: str,
        side: str,
        qty: int,
        limit_price: Optional[float],
        status: str,
        filled_qty: int = 0,
        avg_fill_price: Optional[float] = None,
        note: str = "",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with open(self.order_log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([now, event_type, order_id, side, qty, limit_price, status, filled_qty, avg_fill_price, note])

    def stream(self) -> Iterator[MarketTick]:
        for ts, row in self.df.iterrows():
            yield MarketTick(
                ts=ts,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]),
            )


@dataclass
class Order:
    order_id: str
    ts: pd.Timestamp
    side: str
    qty: int
    limit_price: Optional[float]
    status: str = "NEW"
    filled_qty: int = 0
    avg_fill_price: float = 0.0


class OrderManager:
    def __init__(
        self,
        initial_cash: float = 100_000.0,
        max_orders_per_minute: int = 60,
        max_long_shares: int = 5_000,
        max_short_shares: int = 5_000,
    ):
        self.cash = float(initial_cash)
        self.position = 0
        self.max_orders_per_minute = int(max_orders_per_minute)
        self.max_long_shares = int(max_long_shares)
        self.max_short_shares = int(max_short_shares)
        self._order_timestamps: List[pd.Timestamp] = []

    def equity(self, mark_price: float) -> float:
        return self.cash + self.position * mark_price

    def _orders_in_last_minute(self, now: pd.Timestamp) -> int:
        cutoff = now - pd.Timedelta(minutes=1)
        self._order_timestamps = [t for t in self._order_timestamps if t >= cutoff]
        return len(self._order_timestamps)

    def validate(self, order: Order, mark_price: float) -> Tuple[bool, str]:
        if self._orders_in_last_minute(order.ts) >= self.max_orders_per_minute:
            return False, "RISK: max_orders_per_minute exceeded"

        if order.side == "BUY":
            if self.position + order.qty > self.max_long_shares:
                return False, "RISK: max_long_shares exceeded"
        else:
            if -(self.position - order.qty) > self.max_short_shares:
                return False, "RISK: max_short_shares exceeded"

        px = mark_price if order.limit_price is None else float(order.limit_price)
        if order.side == "BUY":
            needed = px * order.qty
            if self.cash < needed:
                return False, "CAPITAL: insufficient cash"

        return True, "OK"

    def record_order_sent(self, ts: pd.Timestamp) -> None:
        self._order_timestamps.append(ts)

    def apply_execution(self, side: str, fill_qty: int, fill_price: float) -> None:
        if fill_qty <= 0:
            return
        if side == "BUY":
            self.position += fill_qty
            self.cash -= fill_qty * fill_price
        else:
            self.position -= fill_qty
            self.cash += fill_qty * fill_price


class MatchingEngine:
    def __init__(self, rng_seed: int = 123, p_fill: float = 0.70, p_partial: float = 0.20, p_cancel: float = 0.10):
        if not np.isclose(p_fill + p_partial + p_cancel, 1.0):
            raise ValueError("Probabilities must sum to 1.")
        self.rng = random.Random(rng_seed)
        self.p_fill = p_fill
        self.p_partial = p_partial
        self.p_cancel = p_cancel

    def process_order(self, order: Order, mark_price: float) -> Tuple[str, int, float, str]:
        u = self.rng.random()
        if u < self.p_cancel:
            order.status = "CANCELED"
            return "CANCELED", 0, 0.0, "engine_random_cancel"
        elif u < self.p_cancel + self.p_partial:
            fill_qty = max(1, int(order.qty * self.rng.uniform(0.1, 0.9)))
            fill_qty = min(fill_qty, order.qty)
            fill_px = self._fill_price(order.side, mark_price)
            order.status = "PARTIALLY_FILLED"
            order.filled_qty = fill_qty
            order.avg_fill_price = fill_px
            return "PARTIALLY_FILLED", fill_qty, fill_px, "engine_random_partial"
        else:
            fill_qty = order.qty
            fill_px = self._fill_price(order.side, mark_price)
            order.status = "FILLED"
            order.filled_qty = fill_qty
            order.avg_fill_price = fill_px
            return "FILLED", fill_qty, fill_px, "engine_random_fill"

    def _fill_price(self, side: str, mark_price: float) -> float:
        slip_bps = self.rng.uniform(0, 5)
        slip = mark_price * (slip_bps / 10_000.0)
        return mark_price + slip if side == "BUY" else max(1e-10, mark_price - slip)


@dataclass
class Trade:
    ts: pd.Timestamp
    side: str
    qty: int
    price: float


class Backtester:
    def __init__(self, gateway: Gateway, strategy: MovingAverageCrossoverStrategy, om: OrderManager, engine: MatchingEngine, lookback: int = 300):
        self.gateway = gateway
        self.strategy = strategy
        self.om = om
        self.engine = engine
        self.lookback = int(lookback)
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self._hist: List[Tuple[pd.Timestamp, float]] = []

    def run(self) -> None:
        for tick in self.gateway.stream():
            self._hist.append((tick.ts, tick.close))
            if len(self._hist) > self.lookback:
                self._hist.pop(0)

            window = pd.DataFrame({"Close": [p for _, p in self._hist]}, index=[t for t, _ in self._hist])

            signal = self.strategy.generate_signal(window)
            self.strategy.last_signal = signal

            eq = self.om.equity(mark_price=tick.close)
            self.equity_curve.append((tick.ts, eq))

            target_pos = self.strategy.desired_position_shares(equity=eq, price=tick.close, signal=signal)
            delta = target_pos - self.om.position
            if delta == 0:
                continue

            side = "BUY" if delta > 0 else "SELL"
            qty = abs(int(delta))

            order = Order(order_id=str(uuid.uuid4()), ts=tick.ts, side=side, qty=qty, limit_price=None, status="NEW")

            ok, reason = self.om.validate(order, mark_price=tick.close)
            if not ok:
                order.status = "REJECTED"
                self.gateway.log_order_event("REJECT", order.order_id, order.side, order.qty, order.limit_price, order.status, note=reason)
                continue

            self.om.record_order_sent(order.ts)
            order.status = "ACCEPTED"
            self.gateway.log_order_event("SEND", order.order_id, order.side, order.qty, order.limit_price, order.status, note="validated")

            status, filled_qty, avg_px, note = self.engine.process_order(order, mark_price=tick.close)

            if filled_qty > 0:
                self.om.apply_execution(side=order.side, fill_qty=filled_qty, fill_price=avg_px)
                self.trades.append(Trade(ts=tick.ts, side=order.side, qty=filled_qty, price=avg_px))

            self.gateway.log_order_event("UPDATE", order.order_id, order.side, order.qty, order.limit_price, status, filled_qty, avg_px if filled_qty > 0 else None, note)

        if self._hist and self.om.position != 0:
            last_ts, last_px = self._hist[-1]
            side = "SELL" if self.om.position > 0 else "BUY"
            qty = abs(int(self.om.position))
            order_id = str(uuid.uuid4())

            self.gateway.log_order_event("SEND", order_id, side, qty, None, "ACCEPTED", note="forced_final_liquidation")
            self.om.apply_execution(side=side, fill_qty=qty, fill_price=last_px)
            self.trades.append(Trade(ts=last_ts, side=side, qty=qty, price=last_px))
            self.gateway.log_order_event("UPDATE", order_id, side, qty, None, "FILLED", filled_qty=qty, avg_fill_price=last_px, note="forced_final_liquidation")
            self.equity_curve.append((last_ts, self.om.equity(last_px)))

    def results(self) -> Dict[str, float]:
        if not self.equity_curve:
            return {}
        eq = pd.Series([v for _, v in self.equity_curve], index=[t for t, _ in self.equity_curve]).sort_index()
        rets = eq.pct_change().dropna()

        total_return = (eq.iloc[-1] / eq.iloc[0]) - 1.0
        sharpe_bar = float(rets.mean() / (rets.std(ddof=1) + 1e-12))

        peak = eq.cummax()
        dd = (eq - peak) / peak
        max_drawdown = float(dd.min())

        return {
            "start_equity": float(eq.iloc[0]),
            "end_equity": float(eq.iloc[-1]),
            "total_return": float(total_return),
            "sharpe_bar": float(sharpe_bar),
            "max_drawdown": float(max_drawdown),
            "num_trades": float(len(self.trades)),
            "final_cash": float(self.om.cash),
            "final_position_shares": float(self.om.position),
        }


def plot_report(equity_curve: List[Tuple[pd.Timestamp, float]], trades: List[Trade]) -> None:
    import matplotlib.pyplot as plt

    eq = pd.Series([v for _, v in equity_curve], index=[t for t, _ in equity_curve]).sort_index()

    plt.figure()
    plt.plot(eq.index, eq.values)
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.show()

    if trades:
        sizes = [tr.qty for tr in trades]
        plt.figure()
        plt.hist(sizes, bins=30)
        plt.title("Trade Size Distribution")
        plt.xlabel("Shares")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()


def main():
    df = pd.read_parquet("data/processed/features.parquet")
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["Datetime"]).set_index("Datetime").sort_index()

    gateway = Gateway(df, order_log_path="data/processed/orders_audit.csv")
    strategy = MovingAverageCrossoverStrategy(short_window=10, long_window=30, target_risk_fraction=0.20)
    om = OrderManager(initial_cash=100_000.0, max_orders_per_minute=60, max_long_shares=5_000, max_short_shares=5_000)
    engine = MatchingEngine(rng_seed=123, p_fill=0.70, p_partial=0.20, p_cancel=0.10)

    bt = Backtester(gateway=gateway, strategy=strategy, om=om, engine=engine, lookback=300)
    bt.run()

    res = bt.results()
    print("\n=== Backtest Results ===")
    for k, v in res.items():
        print(f"{k:>20s}: {v}")

    plot_report(bt.equity_curve, bt.trades)


if __name__ == "__main__":
    main()
