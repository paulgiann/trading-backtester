"""
Self-test suite for trading-backtester (no external deps).

Run:
    python self_test.py

Design goals:
- deterministic
- zero third-party installs (pytest not required)
- focuses on invariants + core behaviors
"""

from __future__ import annotations

import os
import sys
import time
import unittest
import importlib
from dataclasses import dataclass
from typing import Any, Optional, Tuple


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def try_import(paths):
    """
    Try importing (module_path, attr_name). Return attr, or raise ImportError with details.
    """
    errors = []
    for mod_path, attr in paths:
        try:
            m = importlib.import_module(mod_path)
            if hasattr(m, attr):
                return getattr(m, attr)
            errors.append(f"{mod_path}.{attr}: attribute not found")
        except Exception as e:
            errors.append(f"{mod_path}.{attr}: {type(e).__name__}: {e}")
    raise ImportError("Could not import target. Tried:\n  - " + "\n  - ".join(errors))


def call_first(obj: Any, names: Tuple[str, ...], *args, **kwargs):
    """
    Call the first method name found on obj; return result.
    """
    for n in names:
        if hasattr(obj, n) and callable(getattr(obj, n)):
            return getattr(obj, n)(*args, **kwargs)
    raise AttributeError(f"None of these methods exist on {type(obj).__name__}: {names}")


def get_first(obj: Any, names: Tuple[str, ...]):
    """
    Get first attribute/property/method (0-arg) found on obj.
    If it's callable, call it with no args.
    """
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            return v() if callable(v) else v
    raise AttributeError(f"None of these exist on {type(obj).__name__}: {names}")


@dataclass(frozen=True)
class SimpleOrder:
    """
    A minimal order container we can pass if your OrderBook expects a dict-like/object.
    If your OrderBook takes primitives instead, we will call those signatures too.
    """
    order_id: str
    side: str              # "buy" or "sell"
    price: float
    qty: float
    ts: int = 0            # deterministic timestamp


class TestImports(unittest.TestCase):
    def test_can_import_orderbook(self):
        OrderBook = try_import((
            ("orderbook", "OrderBook"),
            ("src.orderbook", "OrderBook"),
            ("engine.orderbook", "OrderBook"),
            ("trading_backtester.orderbook", "OrderBook"),
        ))
        self.assertTrue(OrderBook is not None)

    def test_can_import_ordermanager_if_exists(self):
        try:
            OrderManager = try_import((
                ("ordermanager", "OrderManager"),
                ("order_manager", "OrderManager"),
                ("src.ordermanager", "OrderManager"),
                ("engine.ordermanager", "OrderManager"),
                ("trading_backtester.ordermanager", "OrderManager"),
            ))
            self.assertTrue(OrderManager is not None)
        except ImportError:
            # OrderManager might not exist or might have different naming; that's ok.
            self.skipTest("OrderManager not found under common module paths.")


class TestOrderBookBasics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.OrderBook = try_import((
            ("orderbook", "OrderBook"),
            ("src.orderbook", "OrderBook"),
            ("engine.orderbook", "OrderBook"),
            ("trading_backtester.orderbook", "OrderBook"),
        ))

    def setUp(self):
        self.ob = self.OrderBook()

    def _add(self, order: SimpleOrder):
        """
        Try common add signatures.
        Supports:
          - add(order_obj_or_dict)
          - add(order_id, side, price, qty, ts=?)
          - add(side, price, qty, order_id=?)
        """
        # 1) object/dict
        try:
            return call_first(self.ob, ("add", "add_order"), vars(order))
        except Exception:
            pass
        try:
            return call_first(self.ob, ("add", "add_order"), order)
        except Exception:
            pass

        # 2) common primitive signatures
        for sig in (
            (order.order_id, order.side, order.price, order.qty, order.ts),
            (order.order_id, order.side, order.price, order.qty),
            (order.side, order.price, order.qty, order.order_id),
            (order.side, order.price, order.qty),
        ):
            try:
                return call_first(self.ob, ("add", "add_order"), *sig)
            except Exception:
                continue

        raise TypeError("Could not find a compatible OrderBook.add/add_order signature.")

    def _modify(self, order_id: str, new_price: Optional[float] = None, new_qty: Optional[float] = None):
        """
        Try common modify signatures.
        """
        # Prefer explicit kwargs if supported
        try:
            return call_first(self.ob, ("modify", "modify_order"), order_id, price=new_price, qty=new_qty)
        except Exception:
            pass
        try:
            return call_first(self.ob, ("modify", "modify_order"), order_id, new_price, new_qty)
        except Exception:
            pass
        # if only qty changes
        if new_qty is not None:
            try:
                return call_first(self.ob, ("modify", "modify_order"), order_id, new_qty)
            except Exception:
                pass
        raise TypeError("Could not find a compatible OrderBook.modify/modify_order signature.")

    def _cancel(self, order_id: str):
        return call_first(self.ob, ("cancel", "cancel_order"), order_id)

    def _best_bid_ask(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Try to read top-of-book. Supports:
          - best_bid/best_ask attributes
          - get_best_bid/get_best_ask methods
          - top_of_book() -> (bid, ask) or dict
        """
        # direct tuple method
        for m in ("top_of_book", "best_bid_ask", "get_top_of_book"):
            if hasattr(self.ob, m) and callable(getattr(self.ob, m)):
                out = getattr(self.ob, m)()
                if isinstance(out, tuple) and len(out) >= 2:
                    return out[0], out[1]
                if isinstance(out, dict):
                    return out.get("best_bid"), out.get("best_ask")

        # separate accessors
        bid = None
        ask = None
        try:
            bid = get_first(self.ob, ("best_bid", "get_best_bid", "bid", "top_bid"))
        except Exception:
            pass
        try:
            ask = get_first(self.ob, ("best_ask", "get_best_ask", "ask", "top_ask"))
        except Exception:
            pass
        return bid, ask

    def test_empty_book_has_no_top(self):
        bid, ask = self._best_bid_ask()
        # Accept None/0/Falsey depending on implementation
        self.assertTrue(bid in (None, 0) or bid is False)
        self.assertTrue(ask in (None, 0) or ask is False)

    def test_add_orders_updates_best_prices(self):
        self._add(SimpleOrder("b1", "buy", 10.0, 1.0, ts=1))
        self._add(SimpleOrder("b2", "buy", 10.5, 1.0, ts=2))
        self._add(SimpleOrder("s1", "sell", 11.0, 1.0, ts=3))
        self._add(SimpleOrder("s2", "sell", 10.8, 1.0, ts=4))

        bid, ask = self._best_bid_ask()

        # If your book matches aggressively on cross, ask might become > bid, but best bid should be around 10.5
        self.assertIsNotNone(bid)
        self.assertGreaterEqual(float(bid), 10.5 - 1e-12)

        self.assertIsNotNone(ask)
        self.assertLessEqual(float(ask), 10.8 + 1e-12)

    def test_cancel_removes_from_top(self):
        self._add(SimpleOrder("b1", "buy", 10.0, 1.0, ts=1))
        self._add(SimpleOrder("b2", "buy", 10.5, 1.0, ts=2))
        bid, _ = self._best_bid_ask()
        self.assertIsNotNone(bid)
        self.assertAlmostEqual(float(bid), 10.5, places=9)

        self._cancel("b2")
        bid2, _ = self._best_bid_ask()
        # If cancel works, best bid should drop to 10.0 (unless implementation reorders differently)
        self.assertTrue(bid2 is None or float(bid2) <= 10.0 + 1e-12)

    def test_modify_price_can_change_priority(self):
        # Price-time priority: raising price should typically move up; lowering should move down.
        self._add(SimpleOrder("b1", "buy", 10.0, 1.0, ts=1))
        self._add(SimpleOrder("b2", "buy", 10.0, 1.0, ts=2))

        # Initially best bid is 10.0
        bid, _ = self._best_bid_ask()
        self.assertIsNotNone(bid)
        self.assertAlmostEqual(float(bid), 10.0, places=9)

        # Raise b1 price above b2
        try:
            self._modify("b1", new_price=10.2, new_qty=None)
        except TypeError:
            self.skipTest("OrderBook.modify signature incompatible; adjust _modify() to your API.")

        bid2, _ = self._best_bid_ask()
        self.assertIsNotNone(bid2)
        self.assertGreaterEqual(float(bid2), 10.2 - 1e-12)


class TestOrderManagerBasics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.OrderManager = try_import((
                ("ordermanager", "OrderManager"),
                ("order_manager", "OrderManager"),
                ("src.ordermanager", "OrderManager"),
                ("engine.ordermanager", "OrderManager"),
                ("trading_backtester.ordermanager", "OrderManager"),
            ))
        except ImportError:
            cls.OrderManager = None

    def setUp(self):
        if self.OrderManager is None:
            self.skipTest("OrderManager not available under common import paths.")
        self.om = self.OrderManager()

    def test_create_modify_cancel_flow_if_supported(self):
        # This is intentionally defensive since OM APIs vary.
        # We verify that the methods exist and can be called in a basic lifecycle.
        has_place = any(hasattr(self.om, n) for n in ("place", "place_order", "submit", "submit_order", "create_order"))
        has_modify = any(hasattr(self.om, n) for n in ("modify", "modify_order", "amend", "amend_order"))
        has_cancel = any(hasattr(self.om, n) for n in ("cancel", "cancel_order"))

        if not (has_place and has_cancel):
            self.skipTest("OrderManager lifecycle methods not found (place/cancel).")

        # Place an order
        order_id = "t1"
        placed = None
        for attempt in (
            ("place", (order_id, "buy", 10.0, 1.0), {}),
            ("place_order", (order_id, "buy", 10.0, 1.0), {}),
            ("submit_order", ({"order_id": order_id, "side": "buy", "price": 10.0, "qty": 1.0},), {}),
            ("create_order", ({"order_id": order_id, "side": "buy", "price": 10.0, "qty": 1.0},), {}),
        ):
            name, args, kwargs = attempt
            if hasattr(self.om, name) and callable(getattr(self.om, name)):
                try:
                    placed = getattr(self.om, name)(*args, **kwargs)
                    break
                except Exception:
                    continue

        self.assertIsNotNone(placed, "Could not place an order via common OrderManager signatures.")

        # Optional modify
        if has_modify:
            for name in ("modify", "modify_order", "amend", "amend_order"):
                if hasattr(self.om, name) and callable(getattr(self.om, name)):
                    try:
                        getattr(self.om, name)(order_id, price=10.1)
                        break
                    except Exception:
                        try:
                            getattr(self.om, name)(order_id, 10.1)
                            break
                        except Exception:
                            pass

        # Cancel
        cancelled = False
        for name in ("cancel", "cancel_order"):
            if hasattr(self.om, name) and callable(getattr(self.om, name)):
                try:
                    getattr(self.om, name)(order_id)
                    cancelled = True
                    break
                except Exception:
                    continue
        self.assertTrue(cancelled, "Could not cancel an order via common OrderManager signatures.")


if __name__ == "__main__":
    # Make tests deterministic if your code uses randomness in default constructors.
    # This won't affect code that uses its own RNGs explicitly.
    import random
    random.seed(0)

    unittest.main(verbosity=2)
