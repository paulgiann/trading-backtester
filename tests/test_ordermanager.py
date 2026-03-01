import pandas as pd

from run_all import Order, OrderManager


def test_order_manager_capital_check():
    om = OrderManager()
    om.cash = 100.0
    o = Order("o1", pd.Timestamp("2026-02-20T00:00:00Z"), "BUY", 2, 100.0)
    ok, reason = om.validate(o, 100.0)
    assert ok is False
    assert "CAPITAL" in reason


def test_order_manager_risk_limits():
    om = OrderManager()
    ts = pd.Timestamp("2026-02-20T00:00:00Z")
    for _ in range(12):
        om.record_sent(ts)
    o = Order("o2", ts, "BUY", 1, 1.0)
    ok, reason = om.validate(o, 1.0)
    assert ok is False
    assert "max_orders_per_min" in reason

    om = OrderManager()
    om.pos = 4999
    o = Order("o3", ts, "BUY", 10, 1.0)
    ok, reason = om.validate(o, 1.0)
    assert ok is False
    assert "max_long_shares" in reason

    om = OrderManager()
    om.pos = -4999
    o = Order("o4", ts, "SELL", 10, 1.0)
    ok, reason = om.validate(o, 1.0)
    assert ok is False
    assert "max_short_shares" in reason
