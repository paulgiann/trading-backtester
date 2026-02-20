# Thin compatibility wrapper so tests can import OrderManager without guessing file names.

from run_all import OrderManager  # preferred (main entrypoint)
# If that causes issues, use:
# from project_backtester import OrderManager

__all__ = ["OrderManager"]
