import pandas as pd

from strategies import MACrossoverStrategy


def test_signal_long_short_flat():
    st = MACrossoverStrategy(short_w=2, long_w=3, target_frac=0.35)

    up = pd.Series([1.0, 2.0, 3.0, 4.0])
    down = pd.Series([4.0, 3.0, 2.0, 1.0])
    short_hist = pd.Series([1.0, 2.0])

    assert st.signal(up) == 1
    assert st.signal(down) == -1
    assert st.signal(short_hist) == 0


def test_spread_ok_threshold():
    st = MACrossoverStrategy(short_w=2, long_w=3, target_frac=0.35)

    close = pd.Series([100.0, 100.0, 100.0, 110.0])
    assert st.spread_ok(close, price=110.0, th=0.01)
    assert not st.spread_ok(close, price=110.0, th=0.20)


def test_target_pos_direction_and_zero_cases():
    st = MACrossoverStrategy(short_w=2, long_w=3, target_frac=0.35, target_vol=0.02)

    assert st.target_pos(equity=100000.0, price=100.0, direction=0, vol_20=0.02) == 0
    assert st.target_pos(equity=100000.0, price=0.0, direction=1, vol_20=0.02) == 0

    long_pos = st.target_pos(equity=100000.0, price=100.0, direction=1, vol_20=0.02)
    short_pos = st.target_pos(equity=100000.0, price=100.0, direction=-1, vol_20=0.02)

    assert long_pos > 0
    assert short_pos < 0
    assert abs(long_pos) == abs(short_pos)
