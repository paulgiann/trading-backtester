import pandas as pd

from run_all import Order, OrderBook


def test_price_time_priority():
    book = OrderBook()
    t1 = pd.Timestamp("2026-02-20T00:00:00Z")
    t2 = pd.Timestamp("2026-02-20T00:00:01Z")
    t3 = pd.Timestamp("2026-02-20T00:00:02Z")

    book.add(Order("b1", t1, "BUY", 5, 101.0))
    book.add(Order("b2", t2, "BUY", 5, 101.0))
    book.add(Order("s1", t3, "SELL", 5, 100.0))

    fills = book.match()
    assert fills[0].order_id == "b1"
    assert fills[0].qty == 5


def test_price_priority_over_time():
    book = OrderBook()
    t1 = pd.Timestamp("2026-02-20T00:00:00Z")
    t2 = pd.Timestamp("2026-02-20T00:00:01Z")
    t3 = pd.Timestamp("2026-02-20T00:00:02Z")

    book.add(Order("b1", t1, "BUY", 5, 101.0))
    book.add(Order("b2", t2, "BUY", 5, 102.0))
    book.add(Order("s1", t3, "SELL", 5, 100.0))

    fills = book.match()
    assert fills[0].order_id == "b2"


def test_cancel_and_modify():
    book = OrderBook()
    t1 = pd.Timestamp("2026-02-20T00:00:00Z")
    t2 = pd.Timestamp("2026-02-20T00:00:01Z")

    book.add(Order("b1", t1, "BUY", 10, 100.0))
    assert book.cancel("b1") is True
    book.add(Order("s1", t2, "SELL", 10, 99.0))
    fills = book.match()
    assert fills == []

    book.add(Order("b2", t1, "BUY", 10, 100.0))
    assert book.modify("b2", new_qty=6, new_price=105.0, ts=t2) is True
    book.add(Order("s2", t2, "SELL", 6, 104.0))
    fills = book.match()
    assert any(f.order_id == "b2" for f in fills)
    assert sum(f.qty for f in fills if f.order_id == "b2") == 6
