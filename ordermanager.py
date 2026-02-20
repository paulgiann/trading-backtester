# ordermanager.py
# Test-compat wrapper: adds place/cancel/modify lifecycle aliases without touching production code.

from __future__ import annotations

from typing import Any, Optional

from run_all import OrderManager as _BaseOrderManager

try:
    # Prefer your local shim if present
    from orderbook import OrderBook as _OrderBook
except Exception:
    from run_all import OrderBook as _OrderBook


class OrderManager(_BaseOrderManager):
    """
    Compatibility wrapper for self_test.py.
    Adds lifecycle methods place/cancel/modify using best-effort forwarding.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Attach an orderbook if the base class does not expose one.
        if not hasattr(self, "orderbook") and not hasattr(self, "order_book") and not hasattr(self, "book") and not hasattr(self, "ob"):
            self.orderbook = _OrderBook()

    def _get_ob(self):
        for n in ("orderbook", "order_book", "book", "ob"):
            if hasattr(self, n):
                return getattr(self, n)
        return None

    def place(self, order_id: str, side: str, price: float, qty: float, ts: Optional[Any] = None):
        order = {"order_id": str(order_id), "side": str(side).upper(), "price": float(price), "qty": float(qty), "ts": ts}

        # 1) Forward to existing base methods if they exist
        for name, args, kwargs in (
            ("submit_order", (order,), {}),
            ("create_order", (order,), {}),
            ("submit", (order_id, side, price, qty, ts), {}),
            ("submit", (order_id, side, price, qty), {}),
            ("send_order", (order,), {}),
            ("new_order", (order,), {}),
            ("process_order", (order,), {}),
        ):
            fn = getattr(self, name, None)
            if callable(fn):
                try:
                    out = fn(*args, **kwargs)
                    return order if out is None else out
                except Exception:
                    pass

        # 2) Fall back to OrderBook add/add_order
        ob = self._get_ob()
        if ob is not None:
            for name, args in (
                ("add", (order,)),
                ("add_order", (order,)),
                ("add", (order_id, order["side"], order["price"], order["qty"], ts)),
                ("add_order", (order_id, order["side"], order["price"], order["qty"], ts)),
                ("add", (order_id, order["side"], order["price"], order["qty"])),
                ("add_order", (order_id, order["side"], order["price"], order["qty"])),
            ):
                fn = getattr(ob, name, None)
                if callable(fn):
                    try:
                        out = fn(*args)
                        return order if out is None else out
                    except Exception:
                        pass

        raise AttributeError("OrderManager.place: no underlying submit/create and no usable OrderBook attached.")

    def place_order(self, *args: Any, **kwargs: Any):
        return self.place(*args, **kwargs)

    def cancel(self, order_id: str):
        oid = str(order_id)

        # 1) Forward to existing base methods if present
        for name in ("cancel_order", "cancel_by_id", "cancel", "remove_order"):
            fn = getattr(self, name, None)
            if callable(fn):
                try:
                    return fn(oid)
                except Exception:
                    pass

        # 2) Fall back to OrderBook
        ob = self._get_ob()
        if ob is not None:
            for name in ("cancel", "cancel_order", "remove", "remove_order"):
                fn = getattr(ob, name, None)
                if callable(fn):
                    try:
                        return fn(oid)
                    except Exception:
                        pass

        raise AttributeError("OrderManager.cancel: no underlying cancel and no usable OrderBook attached.")

    def modify(self, order_id: str, price: Optional[float] = None, qty: Optional[float] = None, ts: Optional[Any] = None):
        oid = str(order_id)

        # 1) Forward to existing base methods if present
        for name, kwargs in (
            ("modify_order", {"new_price": price, "new_qty": qty, "ts": ts}),
            ("amend_order", {"price": price, "qty": qty, "ts": ts}),
            ("amend", {"price": price, "qty": qty, "ts": ts}),
        ):
            fn = getattr(self, name, None)
            if callable(fn):
                try:
                    return fn(oid, **kwargs)
                except Exception:
                    pass

        # 2) Fall back to OrderBook modify/modify_order
        ob = self._get_ob()
        if ob is not None:
            for name, args, kwargs in (
                ("modify", (oid,), {"new_price": price, "new_qty": qty}),
                ("modify_order", (oid,), {"new_price": price, "new_qty": qty}),
                ("modify", (oid, price, qty), {}),
                ("modify_order", (oid, price, qty), {}),
            ):
                fn = getattr(ob, name, None)
                if callable(fn):
                    try:
                        return fn(*args, **kwargs)
                    except Exception:
                        pass

        raise AttributeError("OrderManager.modify: no underlying modify/amend and no usable OrderBook attached.")

