from __future__ import annotations

from alpaca_live_utils import build_live_signal_context


def main():
    ctx = build_live_signal_context()
    print(ctx["action_line"])


if __name__ == "__main__":
    main()
