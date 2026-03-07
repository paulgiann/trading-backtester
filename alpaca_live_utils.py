from __future__ import annotations


def decide_live_action(
    *,
    signal: int,
    spread_ok: bool,
    latest_datetime,
    latest_close: float,
    raw_volume: float,
    raw_trade_count: float,
    bar_age_seconds: float,
    max_bar_age_seconds: int,
) -> str:
    if bar_age_seconds > max_bar_age_seconds:
        return (
            f"decision=BLOCK reason=stale_bar latest_datetime={latest_datetime} "
            f"bar_age_seconds={bar_age_seconds:.3f} max_bar_age_seconds={max_bar_age_seconds}"
        )

    if raw_volume == 0.0 or raw_trade_count == 0.0:
        return f"decision=BLOCK reason=zero_liquidity latest_datetime={latest_datetime}"

    if not spread_ok:
        return f"decision=HOLD reason=spread_filter latest_datetime={latest_datetime}"

    if signal > 0:
        return f"decision=BUY latest_datetime={latest_datetime} close={latest_close}"

    if signal < 0:
        return f"decision=SELL latest_datetime={latest_datetime} close={latest_close}"

    return f"decision=HOLD reason=no_signal latest_datetime={latest_datetime}"
