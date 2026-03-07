from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from alpaca_live_utils import append_live_log_row, build_live_signal_context


def main():
    iterations = int(os.getenv("ALPACA_RUNNER_ITERATIONS", "3"))
    sleep_seconds = float(os.getenv("ALPACA_RUNNER_SLEEP_SECONDS", "5"))
    log_path = os.getenv("ALPACA_RUNNER_LOG_PATH", "outputs/logs/alpaca_live_runner_log.csv")

    print(
        f"runner_start iterations={iterations} sleep_seconds={sleep_seconds} "
        f"log_path={log_path}"
    )

    for i in range(1, iterations + 1):
        now_utc = datetime.now(timezone.utc)
        ctx = build_live_signal_context()

        row = {
            "runner_time_utc": now_utc.isoformat(),
            "iteration": i,
            "symbol": ctx["symbol"],
            "strategy_name": ctx["strategy_name"],
            "latest_datetime": str(ctx["latest_datetime"]),
            "latest_close": ctx["latest_close"],
            "signal": ctx["signal"],
            "spread_ok": ctx["spread_ok"],
            "raw_volume": ctx["raw_volume"],
            "raw_trade_count": ctx["raw_trade_count"],
            "bar_age_seconds": ctx["bar_age_seconds"],
            "max_bar_age_seconds": ctx["max_bar_age_seconds"],
            "action_line": ctx["action_line"],
        }

        out_path = append_live_log_row(row, log_path)
        print(
            f"iteration={i} latest_datetime={ctx['latest_datetime']} "
            f"action_line={ctx['action_line']} log_path={out_path}"
        )

        if i < iterations:
            time.sleep(sleep_seconds)

    print("runner_done")


if __name__ == "__main__":
    main()
