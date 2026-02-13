"""Tests for Portfolio A: Aggressive Momentum strategy."""

import numpy as np
import pandas as pd
import pytest

from src.strategies.portfolio_a import MomentumStrategy


@pytest.fixture
def closes() -> pd.DataFrame:
    """Synthetic close prices for Portfolio A universe tickers."""
    np.random.seed(42)
    dates = pd.bdate_range(end="2026-02-05", periods=80)
    tickers = [
        "XLK",
        "XLF",
        "XLV",
        "XLE",
        "XLI",
        "XLY",
        "XLP",
        "XLU",
        "XLB",
        "XLRE",
        "QQQ",
        "SMH",
        "XBI",
        "IWM",
        "EFA",
        "EEM",
        "TLT",
        "HYG",
        "GDX",
        "ARKK",
    ]
    data = {}
    for i, t in enumerate(tickers):
        drift = 0.5 - i * 0.05  # XLK has strongest trend, ARKK weakest
        data[t] = 100 + np.cumsum(np.random.normal(drift, 1, 80))
    return pd.DataFrame(data, index=dates)


class TestMomentumStrategy:
    def test_rank_returns_all_universe_tickers(self, closes: pd.DataFrame) -> None:
        strategy = MomentumStrategy()
        rankings = strategy.rank(closes)
        assert len(rankings) == 20

    def test_get_target_ticker_returns_string(self, closes: pd.DataFrame) -> None:
        strategy = MomentumStrategy()
        target = strategy.get_target_ticker(closes)
        assert isinstance(target, str)
        assert len(target) > 0

    def test_get_target_ticker_with_insufficient_data(self) -> None:
        short = pd.DataFrame({"XLK": [100, 101]}, index=pd.bdate_range("2026-02-04", periods=2))
        strategy = MomentumStrategy()
        assert strategy.get_target_ticker(short) is None


class TestGenerateTrades:
    def test_fresh_portfolio_generates_buy(self, closes: pd.DataFrame) -> None:
        strategy = MomentumStrategy()
        trades = strategy.generate_trades(
            closes=closes,
            current_positions={},
            cash=33_333.0,
        )
        assert len(trades) == 1
        assert trades[0].side.value == "BUY"
        assert trades[0].portfolio.value == "A"

    def test_same_target_no_trades(self, closes: pd.DataFrame) -> None:
        strategy = MomentumStrategy()
        target = strategy.get_target_ticker(closes)
        trades = strategy.generate_trades(
            closes=closes,
            current_positions={target: 100.0},
            cash=1000.0,
        )
        # No sell, no buy (already holding the target)
        assert len(trades) == 0

    def test_rotation_generates_sell_and_buy(self, closes: pd.DataFrame) -> None:
        strategy = MomentumStrategy()
        target = strategy.get_target_ticker(closes)
        # Hold a different ticker
        wrong_ticker = [t for t in closes.columns if t != target][0]
        trades = strategy.generate_trades(
            closes=closes,
            current_positions={wrong_ticker: 100.0},
            cash=1000.0,
        )
        sides = [t.side.value for t in trades]
        assert "SELL" in sides
        assert "BUY" in sides

    def test_buy_respects_concentration_limit(self, closes: pd.DataFrame) -> None:
        strategy = MomentumStrategy()
        trades = strategy.generate_trades(
            closes=closes,
            current_positions={},
            cash=33_333.0,
            portfolio_value=33_333.0,
        )
        if trades:
            buy = [t for t in trades if t.side.value == "BUY"][0]
            # Should be capped at 35% of portfolio value
            assert buy.total <= 33_333.0 * 0.35 + 1  # +1 for rounding

    def test_buy_uses_full_budget_when_small(self, closes: pd.DataFrame) -> None:
        """When cash is less than the position limit, use all cash."""
        strategy = MomentumStrategy()
        trades = strategy.generate_trades(
            closes=closes,
            current_positions={},
            cash=5_000.0,
            portfolio_value=33_333.0,
        )
        if trades:
            buy = [t for t in trades if t.side.value == "BUY"][0]
            assert buy.total <= 5_000.0
