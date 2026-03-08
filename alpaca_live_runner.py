from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from alpaca_live_utils import append_live_log_row, build_live_signal_context


def main():
    iterations = int(os.getenv("ALPACA_RUNNER_ITERATIONS", "3"))
    sleep_seconds = float(os.getenv("ALPACA_RUNNER_SLEEP_SECONDS", "5"))
    cooldown_seconds = float(os.getenv("ALPACA_RUNNER_COOLDOWN_SECONDS", "0"))
    max_runtime_minutes = float(os.getenv("ALPACA_RUNNER_MAX_RUNTIME_MINUTES", "0"))
    log_path = os.getenv("ALPACA_RUNNER_LOG_PATH", "outputs/logs/alpaca_live_runner_log.csv")

    start_utc = datetime.now(timezone.utc)

    print(
        f"runner_start iterations={iterations} sleep_seconds={sleep_seconds} "
        f"cooldown_seconds={cooldown_seconds} max_runtime_minutes={max_runtime_minutes} "
        f"log_path={log_path}"
    )

    last_seen_latest_datetime = None
    last_actionable_signal_time = None

    for i in range(1, iterations + 1):
        now_utc = datetime.now(timezone.utc)

        if max_runtime_minutes > 0:
            elapsed_minutes = (now_utc - start_utc).total_seconds() / 60.0
            if elapsed_minutes >= max_runtime_minutes:
                print(
                    f"runner_stop reason=max_runtime_reached elapsed_minutes={elapsed_minutes:.3f} "
                    f"max_runtime_minutes={max_runtime_minutes}"
                )
                break

        try:
            ctx = build_live_signal_context()
        except Exception as e:
            row = {
                "runner_time_utc": now_utc.isoformat(),
                "iteration": i,
                "symbol": "",
                "strategy_name": "",
                "latest_datetime": "",
                "latest_close": "",
                "signal": "",
                "spread_ok": "",
                "raw_volume": "",
                "raw_trade_count": "",
                "bar_age_seconds": "",
                "max_bar_age_seconds": "",
                "action_line": f"decision=SKIP reason=runner_exception error={type(e).__name__}",
            }
            out_path = append_live_log_row(row, log_path)
            print(
                f"iteration={i} action_line=decision=SKIP reason=runner_exception "
                f"error={type(e).__name__} log_path={out_path}"
            )
            if i < iterations:
                time.sleep(sleep_seconds)
            continue

        latest_dt_str = str(ctx["latest_datetime"])
        action_line = ctx["action_line"]

        if latest_dt_str == last_seen_latest_datetime:
            final_action_line = "decision=SKIP reason=duplicate_bar"
        else:
            if (
                ("decision=BUY" in action_line or "decision=SELL" in action_line)
                and last_actionable_signal_time is not None
                and (now_utc - last_actionable_signal_time).total_seconds() < cooldown_seconds
            ):
                final_action_line = "decision=SKIP reason=cooldown_active"
            else:
                final_action_line = action_line
                if "decision=BUY" in action_line or "decision=SELL" in action_line:
                    last_actionable_signal_time = now_utc

            last_seen_latest_datetime = latest_dt_str

        row = {
            "runner_time_utc": now_utc.isoformat(),
            "iteration": i,
            "symbol": ctx["symbol"],
            "strategy_name": ctx["strategy_name"],
            "latest_datetime": latest_dt_str,
            "latest_close": ctx["latest_close"],
            "signal": ctx["signal"],
            "spread_ok": ctx["spread_ok"],
            "raw_volume": ctx["raw_volume"],
            "raw_trade_count": ctx["raw_trade_count"],
            "bar_age_seconds": ctx["bar_age_seconds"],
            "max_bar_age_seconds": ctx["max_bar_age_seconds"],
            "action_line": final_action_line,
        }
        out_path = append_live_log_row(row, log_path)
        print(
            f"iteration={i} latest_datetime={latest_dt_str} "
            f"action_line={final_action_line} log_path={out_path}"
        )

        if i < iterations:
            time.sleep(sleep_seconds)

    print("runner_done")


if __name__ == "__main__":
    main()
