from __future__ import annotations

import csv
import os
import uuid
import random
import math
import time
import heapq
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from strategies import MACrossoverStrategy
SKIP_DOWNLOAD = os.getenv("SKIP_DOWNLOAD","0") == "1"


# ============================
# CONFIG
# ============================

SYMBOL = "SOLUSDT"
INTERVAL = "1m"
DAYS = 7

SHORT_W = int(os.getenv("SHORT_W", "10"))
LONG_W = int(os.getenv("LONG_W", "50"))

TARGET_FRAC = 0.35          # exposure on entry (fraction of equity)
SPREAD_TH = 0.0005          # 0.05% MA-spread filter (not too strict for 1m)
COOLDOWN_MIN = 3            # min minutes between orders
TARGET_VOL = 0.02           # vol targeting proxy

STOP_LOSS = 0.012           # 1.2% stop
TAKE_PROFIT = 0.025         # 1.8% take profit
MAX_HOLD_HOURS = 6          # time exit

ENGINE_SEED = int(os.getenv("ENGINE_SEED", "123"))
P_FILL = 0.70
P_PARTIAL = 0.20
P_CANCEL = 0.10

INIT_CASH = 100_000.0
MAX_ORDERS_PER_MIN = 12
MAX_LONG_SHARES = 5_000
MAX_SHORT_SHARES = 5_000


# ============================
# Part 1: Download (Binance)
# ============================

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def download_binance_intraday(symbol: str, interval: str, days: int, out_csv: str = "market_data.csv") -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    cur = _to_ms(start)
    end_ms = _to_ms(end)

    rows = []
    while cur < end_ms:
        r = requests.get(
            BINANCE_KLINES,
            params={"symbol": symbol, "interval": interval, "startTime": cur, "endTime": end_ms, "limit": 1000},
            timeout=30,
        )
        r.raise_for_status()
        kl = r.json()
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
        time.sleep(0.10)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No data returned. Check symbol/interval.")

    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
    df = df.drop_duplicates(subset=["Datetime"]).sort_values("Datetime")
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} with {len(df)} rows for {symbol} ({interval}), ~{days} days.")
    return df


# ============================
# Part 1: Clean + features
# ============================

def clean_and_engineer_features(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    req = {"Datetime", "Open", "High", "Low", "Close", "Volume"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"CSV missing required columns: {miss}")

    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["Datetime", "Open", "High", "Low", "Close", "Volume"])
    df = df.drop_duplicates(subset=["Datetime"], keep="last")
    df = df.set_index("Datetime").sort_index()

    df["logret_1"] = np.log(df["Close"]).diff()
    df["vol_20"] = df["logret_1"].rolling(window=20, min_periods=20).std(ddof=1) * math.sqrt(20)

    return df.dropna()


# ============================
# Strategy: crossover (event-driven)
# ============================

class Strategy:
    def __init__(self, short_w: int, long_w: int, target_frac: float):
        if short_w <= 0 or long_w <= 0 or short_w >= long_w:
            raise ValueError("Require 0 < short_w < long_w.")
        self.short_w = short_w
        self.long_w = long_w
        self.target_frac = target_frac

    def signal(self, close: pd.Series) -> int:
        if len(close) < self.long_w:
            return 0
        s = close.rolling(self.short_w).mean().iloc[-1]
        l = close.rolling(self.long_w).mean().iloc[-1]
        if not (np.isfinite(s) and np.isfinite(l)):
            return 0
        return 1 if s > l else (-1 if s < l else 0)

    def spread_ok(self, close: pd.Series, price: float, th: float) -> bool:
        s = close.rolling(self.short_w).mean().iloc[-1]
        l = close.rolling(self.long_w).mean().iloc[-1]
        if not (np.isfinite(s) and np.isfinite(l)):
            return False
        spread = abs(s - l) / max(float(price), 1e-12)
        return spread >= th

    def vol_scale(self, vol_20: float, target_vol: float) -> float:
        if not np.isfinite(vol_20):
            return 1.0
        v = max(float(vol_20), 1e-6)
        scale = target_vol / v
        return min(2.0, max(0.25, scale))

    def target_pos(self, equity: float, price: float, direction: int, vol_20: float) -> int:
        if direction == 0 or price <= 0:
            return 0
        scale = self.vol_scale(vol_20, TARGET_VOL)
        notional = self.target_frac * scale * equity
        return direction * int(notional / price)


# ============================
# Part 2: Gateway / Orders
# ============================

@dataclass
class MarketTick:
    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    vol_20: float
    ma_spread: float
    z_px_60: float
    breakout_up: int
    breakout_dn: int
    volm_ratio: float
    range_30: float

class Gateway:
    def __init__(self, df: pd.DataFrame, audit_path: str = "orders_audit.csv"):
        self.df = df.copy()
        self.audit_path = audit_path
        self._init()

    def _init(self) -> None:
        if not os.path.exists(self.audit_path):
            with open(self.audit_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["event_time_utc","event_type","order_id","side","qty","status","filled_qty","avg_fill_price","note"])

    def log(self, event_type: str, order_id: str, side: str, qty: int, status: str, filled_qty: int = 0, avg_px: Optional[float] = None, note: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat()
        with open(self.audit_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([now, event_type, order_id, side, qty, status, filled_qty, avg_px, note])

    def stream(self) -> Iterator[MarketTick]:
        for ts, r in self.df.iterrows():
            yield MarketTick(ts, float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"]), float(r["Volume"]), float(r["vol_20"]), float(r["ma_spread"]), float(r["z_px_60"]), int(r["breakout_up"]), int(r["breakout_dn"]), float(r["volm_ratio"]), float(r["range_30"]))

@dataclass
class Order:
    order_id: str
    ts: pd.Timestamp
    side: str
    qty: int
    price: float

@dataclass
class Fill:
    order_id: str
    side: str
    qty: int
    price: float

class OrderBook:
    def __init__(self):
        self._bids: List[Tuple[float, pd.Timestamp, int, str, int]] = []
        self._asks: List[Tuple[float, pd.Timestamp, int, str, int]] = []
        self._orders: Dict[str, Dict[str, object]] = {}
        self._seq = 0

    def add(self, o: Order) -> None:
        if isinstance(o, dict):
            order_id = o.get("order_id")
            side = o.get("side")
            price = o.get("price")
            qty = o.get("qty")
            ts = o.get("ts")
            o = Order(str(order_id), pd.Timestamp(ts) if ts is not None else pd.Timestamp.now("UTC"), str(side), int(qty), float(price))
        elif not isinstance(o.ts, pd.Timestamp):
            o = Order(o.order_id, pd.Timestamp(o.ts), o.side, o.qty, o.price)

        if o.qty <= 0:
            return
        if o.order_id in self._orders and self._orders[o.order_id]["active"]:
            raise ValueError(f"Order {o.order_id} already exists.")
        self._seq += 1
        rec = {
            "order_id": o.order_id,
            "side": str(o.side).upper(),
            "qty": int(o.qty),
            "price": float(o.price),
            "ts": o.ts,
            "seq": self._seq,
            "version": 1,
            "active": True,
        }
        self._orders[o.order_id] = rec
        self._push(rec)

    def add_order(self, o: Order) -> None:
        self.add(o)

    def modify(self, order_id: str, new_price: Optional[float] = None, new_qty: Optional[int] = None, ts: Optional[pd.Timestamp] = None) -> bool:
        rec = self._orders.get(order_id)
        if not rec or not rec["active"]:
            return False
        if new_price is not None:
            rec["price"] = float(new_price)
        if new_qty is not None:
            rec["qty"] = int(new_qty)
        if ts is not None:
            rec["ts"] = ts
        if rec["qty"] <= 0:
            rec["active"] = False
            rec["version"] += 1
            return True
        self._seq += 1
        rec["seq"] = self._seq
        rec["version"] += 1
        self._push(rec)
        return True

    def modify_order(self, order_id: str, new_price: Optional[float] = None, new_qty: Optional[int] = None, ts: Optional[pd.Timestamp] = None) -> bool:
        return self.modify(order_id, new_price=new_price, new_qty=new_qty, ts=ts)

    def cancel(self, order_id: str) -> bool:
        rec = self._orders.get(order_id)
        if not rec or not rec["active"]:
            return False
        rec["active"] = False
        rec["version"] += 1
        return True

    def cancel_order(self, order_id: str) -> bool:
        return self.cancel(order_id)

    def get_best_bid(self) -> Optional[float]:
        rec = self._best("BUY")
        return None if rec is None else float(rec["price"])

    def get_best_ask(self) -> Optional[float]:
        rec = self._best("SELL")
        return None if rec is None else float(rec["price"])

    @property
    def best_bid(self) -> Optional[float]:
        return self.get_best_bid()

    @property
    def best_ask(self) -> Optional[float]:
        return self.get_best_ask()

    def top_of_book(self) -> Tuple[Optional[float], Optional[float]]:
        return self.get_best_bid(), self.get_best_ask()

    def get_order(self, order_id: str) -> Optional[Dict[str, object]]:
        return self._orders.get(order_id)

    def match(self) -> List[Fill]:
        fills: List[Fill] = []
        while True:
            bid = self._best("BUY")
            ask = self._best("SELL")
            if bid is None or ask is None:
                break
            if bid["price"] < ask["price"]:
                break
            qty = min(bid["qty"], ask["qty"])
            px = float(ask["price"])
            bid["qty"] -= qty
            ask["qty"] -= qty
            fills.append(Fill(bid["order_id"], "BUY", qty, px))
            fills.append(Fill(ask["order_id"], "SELL", qty, px))
            if bid["qty"] <= 0:
                bid["active"] = False
                bid["version"] += 1
            if ask["qty"] <= 0:
                ask["active"] = False
                ask["version"] += 1
        return fills

    def _push(self, rec: Dict[str, object]) -> None:
        if rec["side"] == "BUY":
            heapq.heappush(self._bids, (-rec["price"], rec["ts"], rec["seq"], rec["order_id"], rec["version"]))
        else:
            heapq.heappush(self._asks, (rec["price"], rec["ts"], rec["seq"], rec["order_id"], rec["version"]))

    def _best(self, side: str) -> Optional[Dict[str, object]]:
        heap = self._bids if side == "BUY" else self._asks
        while heap:
            _, _, _, oid, ver = heap[0]
            rec = self._orders.get(oid)
            if rec is None or not rec["active"] or rec["version"] != ver or rec["qty"] <= 0:
                heapq.heappop(heap)
                continue
            return rec
        return None

class OrderManager:
    def __init__(self):
        self.cash = float(INIT_CASH)
        self.pos = 0
        self._sent: List[pd.Timestamp] = []

    def equity(self, px: float) -> float:
        return self.cash + self.pos * px

    def _sent_last_min(self, now: pd.Timestamp) -> int:
        cutoff = now - pd.Timedelta(minutes=1)
        self._sent = [t for t in self._sent if t >= cutoff]
        return len(self._sent)

    def validate(self, o: Order, mark_px: float) -> Tuple[bool, str]:
        if self._sent_last_min(o.ts) >= MAX_ORDERS_PER_MIN:
            return False, "RISK: max_orders_per_min exceeded"

        if o.side == "BUY":
            if self.pos + o.qty > MAX_LONG_SHARES:
                return False, "RISK: max_long_shares exceeded"
            if self.cash < mark_px * o.qty:
                return False, "CAPITAL: insufficient cash"
        else:
            if -(self.pos - o.qty) > MAX_SHORT_SHARES:
                return False, "RISK: max_short_shares exceeded"

        return True, "OK"

    def record_sent(self, ts: pd.Timestamp) -> None:
        self._sent.append(ts)

    def apply_fill(self, side: str, qty: int, px: float) -> None:
        if qty <= 0:
            return
        if side == "BUY":
            self.pos += qty
            self.cash -= qty * px
        else:
            self.pos -= qty
            self.cash += qty * px

class MatchingEngine:
    def __init__(self):
        if not np.isclose(P_FILL + P_PARTIAL + P_CANCEL, 1.0):
            raise ValueError("Probabilities must sum to 1.")
        self.rng = random.Random(ENGINE_SEED)

    def process(self, o: Order, book: OrderBook, mark_px: float) -> Tuple[str, List[Fill], str]:
        u = self.rng.random()
        if u < P_CANCEL:
            book.cancel(o.order_id)
            return "CANCELED", [], "engine_cancel"

        if u < P_CANCEL + P_PARTIAL:
            fqty = max(1, int(o.qty * self.rng.uniform(0.2, 0.8)))
            fqty = min(fqty, o.qty)
            note = "engine_partial"
        else:
            fqty = o.qty
            note = "engine_fill"

        slip_bps = self.rng.uniform(0, 5)
        slip = mark_px * (slip_bps / 10_000.0)
        exec_px = mark_px + slip if o.side == "BUY" else max(1e-10, mark_px - slip)

        if o.side == "BUY" and exec_px > o.price:
            book.modify(o.order_id, new_price=exec_px, ts=o.ts)
        if o.side == "SELL" and exec_px < o.price:
            book.modify(o.order_id, new_price=exec_px, ts=o.ts)

        contra_side = "SELL" if o.side == "BUY" else "BUY"
        contra = Order(f"LP-{uuid.uuid4()}", o.ts, contra_side, fqty, exec_px)
        book.add(contra)
        fills = book.match()

        # Remove any leftover synthetic liquidity
        c = book.get_order(contra.order_id)
        if c and c.get("active"):
            book.cancel(contra.order_id)

        return ("FILLED" if fqty == o.qty else "PARTIALLY_FILLED"), fills, note


# ============================
# Part 3: Backtest (with TP/SL + time exit)
# ============================

@dataclass
class Trade:
    ts: pd.Timestamp
    side: str
    qty: int
    price: float
    tag: str

class Backtester:
    def __init__(self, gw: Gateway, st: Strategy, om: OrderManager, book: OrderBook, eng: MatchingEngine):
        self.gw = gw
        self.st = st
        self.om = om
        self.book = book
        self.eng = eng

        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []

        self.cooldown = pd.Timedelta(minutes=COOLDOWN_MIN)
        self._last_order_ts: Optional[pd.Timestamp] = None

        self._entry_ts: Optional[pd.Timestamp] = None
        self._entry_px: Optional[float] = None

        self._prev_sig = 0
        self._hist_close: List[float] = []
        self._hist_ts: List[pd.Timestamp] = []
        self._hist_vol: List[float] = []

    def _send_order(self, ts: pd.Timestamp, side: str, qty: int, mark_px: float, tag: str) -> None:
        o = Order(str(uuid.uuid4()), ts, side, qty, mark_px)
        ok, reason = self.om.validate(o, mark_px)
        if not ok:
            self.gw.log("REJECT", o.order_id, side, qty, "REJECTED", note=reason)
            return

        self.om.record_sent(ts)
        self.gw.log("SEND", o.order_id, side, qty, "ACCEPTED", note=tag)
        self.book.add(o)
        status, fills, note = self.eng.process(o, self.book, mark_px)

        filled_qty = 0
        filled_notional = 0.0
        for f in fills:
            if f.order_id == o.order_id:
                self.om.apply_fill(f.side, f.qty, f.price)
                self.trades.append(Trade(ts, f.side, f.qty, f.price, tag))
                filled_qty += f.qty
                filled_notional += f.qty * f.price

        if filled_qty > 0:
            self._last_order_ts = ts
            if self._entry_px is None and self.om.pos != 0:
                self._entry_px = filled_notional / filled_qty
                self._entry_ts = ts
            if self.om.pos == 0:
                self._entry_px = None
                self._entry_ts = None

        avg_px = (filled_notional / filled_qty) if filled_qty > 0 else None
        self.gw.log("UPDATE", o.order_id, side, qty, status, filled_qty=filled_qty, avg_px=avg_px, note=note)

    def run(self) -> None:
        for tick in self.gw.stream():
            self._hist_close.append(tick.close)
            self._hist_ts.append(tick.ts)
            self._hist_vol.append(tick.vol_20)

            close = pd.Series(self._hist_close, index=self._hist_ts)
            vol_20 = float(self._hist_vol[-1])

            self.equity_curve.append((tick.ts, self.om.equity(tick.close)))

            # Exit checks every bar
            if self.om.pos != 0 and self._entry_px is not None and self._entry_ts is not None:
                pos_dir = 1 if self.om.pos > 0 else -1
                pnl = pos_dir * (tick.close / self._entry_px - 1.0)
                age = tick.ts - self._entry_ts

                if pnl <= -STOP_LOSS or pnl >= TAKE_PROFIT or age >= pd.Timedelta(hours=MAX_HOLD_HOURS):
                    if self._last_order_ts is None or (tick.ts - self._last_order_ts) >= self.cooldown:
                        side = "SELL" if self.om.pos > 0 else "BUY"
                        qty = abs(int(self.om.pos))
                        self._send_order(tick.ts, side, qty, tick.close, "EXIT_tp_sl_time")

            # Signal & crossover event
            sig = self.st.signal(close)
            crossover = (sig != 0) and (sig != self._prev_sig)
            self._prev_sig = sig
            if not crossover:
                continue

            # Entry filter
            if not self.st.spread_ok(close, tick.close, SPREAD_TH):
                continue

            # Cooldown gate
            if self._last_order_ts is not None and (tick.ts - self._last_order_ts) < self.cooldown:
                continue

            # Decide target position and trade delta
            eq = self.om.equity(tick.close)
            target = self.st.target_pos(eq, tick.close, sig, vol_20)
            delta = target - self.om.pos
            if delta == 0:
                continue

            side = "BUY" if delta > 0 else "SELL"
            qty = abs(int(delta))
            self._send_order(tick.ts, side, qty, tick.close, "ENTRY_crossover")

        # Liquidate at end for clean reporting
        last_ts = self._hist_ts[-1] if self._hist_ts else None
        last_px = self._hist_close[-1] if self._hist_close else None
        if last_ts is not None and last_px is not None and self.om.pos != 0:
            side = "SELL" if self.om.pos > 0 else "BUY"
            qty = abs(int(self.om.pos))
            self._send_order(last_ts, side, qty, float(last_px), "FINAL_liquidate")

    def results(self) -> Dict[str, float]:
        if not self.equity_curve:
            return {}
        eq = pd.Series([v for _, v in self.equity_curve], index=[t for t, _ in self.equity_curve]).sort_index()
        rets = eq.pct_change().dropna()

        total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
        sharpe_bar = float(rets.mean() / (rets.std(ddof=1) + 1e-12))

        peak = eq.cummax()
        dd = (eq - peak) / peak
        max_dd = float(dd.min())

        return {
            "start_equity": float(eq.iloc[0]),
            "end_equity": float(eq.iloc[-1]),
            "total_return": total_return,
            "sharpe_bar": sharpe_bar,
            "max_drawdown": max_dd,
            "num_trades": float(len(self.trades)),
            "final_cash": float(self.om.cash),
            "final_position_shares": float(self.om.pos),
        }


def plot_report(equity_curve: List[Tuple[pd.Timestamp, float]], trades: List[Trade]) -> None:
    import matplotlib.pyplot as plt

    eq = pd.Series([v for _, v in equity_curve], index=[t for t, _ in equity_curve]).sort_index()

    stamp = (
        f"SOL | SW={SHORT_W} LW={LONG_W} frac={TARGET_FRAC} "
        f"spread={SPREAD_TH} TP={TAKE_PROFIT} SL={STOP_LOSS} "
        f"seed={ENGINE_SEED}"
    )

    safe = stamp.replace(" ", "_").replace("|", "").replace("/", "_").replace("=", "")

    # Save series for auditability
    eq.to_csv(f"equity_{safe}.csv", header=["equity"])

    # Equity curve
    plt.figure()
    plt.plot(eq.index, eq.values)
    plt.title(f"Equity Curve ({stamp})")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(f"equity_curve_{safe}.png", dpi=200)
    plt.show()

    # Trade size distribution
    if trades:
        plt.figure()
        plt.hist([tr.qty for tr in trades], bins=20)
        plt.title(f"Trade Size Distribution ({stamp})")
        plt.xlabel("Shares")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"trade_sizes_{safe}.png", dpi=200)
        plt.show()
def main():
    if not (SKIP_DOWNLOAD and os.path.exists("data/raw/market_data.csv")):
        download_binance_intraday(SYMBOL, INTERVAL, DAYS, out_csv="data/raw/market_data.csv")
    else:
        print("Using existing data/raw/market_data.csv (SKIP_DOWNLOAD=1)")
    if not os.path.exists("data/processed/features.parquet"):
        raise FileNotFoundError("Expected data/processed/features.parquet. Run prepare_data.py first.")
    df = pd.read_parquet("data/processed/features.parquet")
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["Datetime"]).set_index("Datetime").sort_index()
    if "vol_20" not in df.columns:
        if "vol_30" in df.columns:
            df["vol_20"] = df["vol_30"]
        else:
            raise KeyError("Expected vol_20 or vol_30 in processed features.")


    gw = Gateway(df, audit_path="data/processed/orders_audit_run_all.csv")
    st = MACrossoverStrategy(SHORT_W, LONG_W, TARGET_FRAC)
    om = OrderManager()
    book = OrderBook()
    eng = MatchingEngine()

    bt = Backtester(gw, st, om, book, eng)
    bt.run()

    res = bt.results()
    print("\n=== Backtest Results (SOL) ===")
    for k, v in res.items():
        print(f"{k:>20s}: {v}")

    plot_report(bt.equity_curve, bt.trades)

if __name__ == "__main__":
    main()
