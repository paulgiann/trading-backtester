from __future__ import annotations

import numpy as np
import pandas as pd


class MACrossoverStrategy:
    def __init__(self, short_w: int, long_w: int, target_frac: float, target_vol: float = 0.02):
        if short_w <= 0 or long_w <= 0 or short_w >= long_w:
            raise ValueError("Require 0 < short_w < long_w.")
        self.short_w = short_w
        self.long_w = long_w
        self.target_frac = target_frac
        self.target_vol = target_vol

    def signal(self, close: pd.Series) -> int:
        if len(close) < self.long_w:
            return 0
        s = close.rolling(self.short_w).mean().iloc[-1]
        l = close.rolling(self.long_w).mean().iloc[-1]
        if not (np.isfinite(s) and np.isfinite(l)):
            return 0
        return 1 if s > l else (-1 if s < l else 0)

    def spread_ok(self, close: pd.Series, price: float, th: float) -> bool:
        s = close.rolling(self.short_w).mean().iloc[-1]
        l = close.rolling(self.long_w).mean().iloc[-1]
        if not (np.isfinite(s) and np.isfinite(l)):
            return False
        spread = abs(s - l) / max(float(price), 1e-12)
        return spread >= th

    def vol_scale(self, vol_20: float) -> float:
        if not np.isfinite(vol_20):
            return 1.0
        v = max(float(vol_20), 1e-6)
        scale = self.target_vol / v
        return min(2.0, max(0.25, scale))

    def target_pos(self, equity: float, price: float, direction: int, vol_20: float) -> int:
        if direction == 0 or price <= 0:
            return 0
        scale = self.vol_scale(vol_20)
        notional = self.target_frac * scale * equity
        return direction * int(notional / price)
