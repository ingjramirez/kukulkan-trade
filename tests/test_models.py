"""Tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from src.storage.models import OrderSide, PortfolioName, TradeSchema


class TestTradeSchema:
    def test_valid_trade(self) -> None:
        trade = TradeSchema(
            portfolio=PortfolioName.A,
            ticker="XLK",
            side=OrderSide.BUY,
            shares=100.0,
            price=200.0,
            reason="momentum signal",
        )
        assert trade.total == 20_000.0

    def test_invalid_shares_zero(self) -> None:
        with pytest.raises(ValidationError):
            TradeSchema(
                portfolio=PortfolioName.A,
                ticker="XLK",
                side=OrderSide.BUY,
                shares=0,
                price=200.0,
            )

    def test_invalid_negative_price(self) -> None:
        with pytest.raises(ValidationError):
            TradeSchema(
                portfolio=PortfolioName.A,
                ticker="XLK",
                side=OrderSide.SELL,
                shares=50.0,
                price=-10.0,
            )

    def test_ticker_max_length(self) -> None:
        with pytest.raises(ValidationError):
            TradeSchema(
                portfolio=PortfolioName.B,
                ticker="X" * 11,
                side=OrderSide.BUY,
                shares=10.0,
                price=100.0,
            )
