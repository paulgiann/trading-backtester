from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AlpacaConfig:
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: str = "https://paper-api.alpaca.markets"

    @classmethod
    def from_env(cls) -> "AlpacaConfig":
        return cls(
            api_key=os.getenv("ALPACA_API_KEY"),
            api_secret=os.getenv("ALPACA_API_SECRET"),
            base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
        )


class AlpacaPaperGateway:
    def __init__(self, config: Optional[AlpacaConfig] = None):
        self.config = config or AlpacaConfig.from_env()

    def is_configured(self) -> bool:
        return bool(self.config.api_key and self.config.api_secret)

    def summary(self) -> str:
        return (
            f"Alpaca paper gateway configured={self.is_configured()} "
            f"base_url={self.config.base_url}"
        )


if __name__ == "__main__":
    gw = AlpacaPaperGateway()
    print(gw.summary())
