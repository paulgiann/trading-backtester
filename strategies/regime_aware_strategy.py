from __future__ import annotations

import numpy as np
import pandas as pd


class RegimeAwareStrategy:
    def __init__(
        self,
        short_w: int,
        long_w: int,
        target_frac: float,
        target_vol: float = 0.02,
        z_enter: float = 1.75,
        breakout_z_min: float = 0.5,
        vol_ratio_min: float = 1.10,
        range_cap: float = 0.020,
    ):
        if short_w <= 0 or long_w <= 0 or short_w >= long_w:
            raise ValueError("Require 0 < short_w < long_w.")
        self.short_w = short_w
        self.long_w = long_w
        self.target_frac = target_frac
        self.target_vol = target_vol
        self.z_enter = z_enter
        self.breakout_z_min = breakout_z_min
        self.vol_ratio_min = vol_ratio_min
        self.range_cap = range_cap

    def signal(
        self,
        close: pd.Series,
        z_px_60: float,
        breakout_up: int,
        breakout_dn: int,
        volm_ratio: float,
        range_30: float,
    ) -> int:
        if len(close) < self.long_w:
            return 0

        s = close.rolling(self.short_w).mean().iloc[-1]
        l = close.rolling(self.long_w).mean().iloc[-1]
        if not (np.isfinite(s) and np.isfinite(l)):
            return 0

        if np.isfinite(range_30) and range_30 > self.range_cap:
            return 0

        trend_dir = 1 if s > l else (-1 if s < l else 0)

        if trend_dir == 1 and breakout_up == 1 and volm_ratio >= self.vol_ratio_min:
            if np.isfinite(z_px_60) and z_px_60 >= self.breakout_z_min:
                return 1
        if trend_dir == -1 and breakout_dn == 1 and volm_ratio >= self.vol_ratio_min:
            if np.isfinite(z_px_60) and z_px_60 <= -self.breakout_z_min:
                return -1

        return 0

    def spread_ok(self, close: pd.Series, price: float, th: float) -> bool:
        s = close.rolling(self.short_w).mean().iloc[-1]
        l = close.rolling(self.long_w).mean().iloc[-1]
        if not (np.isfinite(s) and np.isfinite(l)):
            return False
        spread = abs(s - l) / max(float(price), 1e-12)
        return bool(spread >= th)

    def vol_scale(self, vol_20: float) -> float:
        if not np.isfinite(vol_20):
            return 1.0
        v = max(float(vol_20), 1e-6)
        scale = self.target_vol / v
        return min(2.0, max(0.20, scale))

    def target_pos(self, equity: float, price: float, direction: int, vol_20: float) -> int:
        if direction == 0 or price <= 0:
            return 0
        scale = self.vol_scale(vol_20)
        notional = self.target_frac * scale * equity
        return direction * int(notional / price)
