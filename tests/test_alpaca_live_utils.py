from alpaca_live_utils import decide_live_action


def test_decide_live_action_blocks_stale_bar():
    out = decide_live_action(
        signal=1,
        spread_ok=True,
        latest_datetime="2026-03-07 10:13:00+00:00",
        latest_close=84.5,
        raw_volume=10.0,
        raw_trade_count=3.0,
        bar_age_seconds=120.0,
        max_bar_age_seconds=90,
    )
    assert "decision=BLOCK" in out
    assert "reason=stale_bar" in out


def test_decide_live_action_blocks_zero_liquidity():
    out = decide_live_action(
        signal=1,
        spread_ok=True,
        latest_datetime="2026-03-07 10:13:00+00:00",
        latest_close=84.5,
        raw_volume=0.0,
        raw_trade_count=0.0,
        bar_age_seconds=10.0,
        max_bar_age_seconds=90,
    )
    assert out == "decision=BLOCK reason=zero_liquidity latest_datetime=2026-03-07 10:13:00+00:00"


def test_decide_live_action_holds_on_spread_filter():
    out = decide_live_action(
        signal=1,
        spread_ok=False,
        latest_datetime="2026-03-07 10:13:00+00:00",
        latest_close=84.5,
        raw_volume=10.0,
        raw_trade_count=3.0,
        bar_age_seconds=10.0,
        max_bar_age_seconds=90,
    )
    assert out == "decision=HOLD reason=spread_filter latest_datetime=2026-03-07 10:13:00+00:00"


def test_decide_live_action_buy_sell_and_flat():
    buy = decide_live_action(
        signal=1,
        spread_ok=True,
        latest_datetime="2026-03-07 10:13:00+00:00",
        latest_close=84.5,
        raw_volume=10.0,
        raw_trade_count=3.0,
        bar_age_seconds=10.0,
        max_bar_age_seconds=90,
    )
    sell = decide_live_action(
        signal=-1,
        spread_ok=True,
        latest_datetime="2026-03-07 10:13:00+00:00",
        latest_close=84.5,
        raw_volume=10.0,
        raw_trade_count=3.0,
        bar_age_seconds=10.0,
        max_bar_age_seconds=90,
    )
    flat = decide_live_action(
        signal=0,
        spread_ok=True,
        latest_datetime="2026-03-07 10:13:00+00:00",
        latest_close=84.5,
        raw_volume=10.0,
        raw_trade_count=3.0,
        bar_age_seconds=10.0,
        max_bar_age_seconds=90,
    )
    assert buy == "decision=BUY latest_datetime=2026-03-07 10:13:00+00:00 close=84.5"
    assert sell == "decision=SELL latest_datetime=2026-03-07 10:13:00+00:00 close=84.5"
    assert flat == "decision=HOLD reason=no_signal latest_datetime=2026-03-07 10:13:00+00:00"
