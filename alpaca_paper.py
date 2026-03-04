from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


PAPER_BASE_URL = "https://paper-api.alpaca.markets"


@dataclass
class AlpacaConfig:
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: str = PAPER_BASE_URL

    @classmethod
    def from_env(cls) -> "AlpacaConfig":
        return cls(
            api_key=os.getenv("ALPACA_API_KEY"),
            api_secret=os.getenv("ALPACA_API_SECRET"),
            base_url=os.getenv("ALPACA_BASE_URL", PAPER_BASE_URL),
        )


class AlpacaPaperGateway:
    """
    Paper-trading gateway placeholder for the project extension.

    This class is intentionally paper-only. It is meant to show how the
    backtester could connect to Alpaca's paper environment without enabling
    accidental live trading.
    """

    def __init__(self, config: Optional[AlpacaConfig] = None):
        self.config = config or AlpacaConfig.from_env()

    def is_configured(self) -> bool:
        return bool(self.config.api_key and self.config.api_secret)

    def is_paper_url(self) -> bool:
        return self.config.base_url.rstrip("/") == PAPER_BASE_URL

    def validate(self) -> None:
        if not self.is_paper_url():
            raise ValueError(
                "AlpacaPaperGateway is paper-only. "
                f"Expected base_url={PAPER_BASE_URL}, got {self.config.base_url}"
            )

    def summary(self) -> str:
        configured = self.is_configured()
        paper_only = self.is_paper_url()
        return (
            f"Alpaca paper gateway configured={configured} "
            f"paper_only={paper_only} "
            f"base_url={self.config.base_url}"
        )

    def account_summary(self) -> str:
        if not self.is_configured():
            return "Alpaca paper gateway not configured with API credentials."

        try:
            from alpaca.trading.client import TradingClient
        except ImportError:
            return "alpaca-py is not installed. Install it with: pip install alpaca-py"

        client = TradingClient(
            self.config.api_key,
            self.config.api_secret,
            paper=True,
        )
        acct = client.get_account()
        return (
            f"status={acct.status} "
            f"buying_power={acct.buying_power} "
            f"equity={acct.equity}"
        )


if __name__ == "__main__":
    gw = AlpacaPaperGateway()
    try:
        gw.validate()
        print(gw.summary())
        print(gw.account_summary())
    except ValueError as exc:
        print(f"Configuration error: {exc}")
