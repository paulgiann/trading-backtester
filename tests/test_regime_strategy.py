import pandas as pd

from strategies import RegimeAwareStrategy


def test_trend_breakout_long_signal():
    st = RegimeAwareStrategy(short_w=2, long_w=3, target_frac=0.35)
    close = pd.Series([100.0, 101.0, 102.0, 103.0])
    sig = st.signal(
        close=close,
        z_px_60=0.0,
        breakout_up=1,
        breakout_dn=0,
        volm_ratio=1.2,
        range_30=0.01,
    )
    assert sig == 1


def test_trend_breakout_short_signal():
    st = RegimeAwareStrategy(short_w=2, long_w=3, target_frac=0.35)
    close = pd.Series([103.0, 102.0, 101.0, 100.0])
    sig = st.signal(
        close=close,
        z_px_60=0.0,
        breakout_up=0,
        breakout_dn=1,
        volm_ratio=1.2,
        range_30=0.01,
    )
    assert sig == -1


def test_no_trade_when_range_too_high():
    st = RegimeAwareStrategy(short_w=2, long_w=3, target_frac=0.35, range_cap=0.02)
    close = pd.Series([100.0, 101.0, 102.0, 103.0])
    sig = st.signal(
        close=close,
        z_px_60=0.0,
        breakout_up=1,
        breakout_dn=0,
        volm_ratio=1.2,
        range_30=0.03,
    )
    assert sig == 0
