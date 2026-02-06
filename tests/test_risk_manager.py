"""Tests for the RiskManager enforcing risk_rules.py limits."""

from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from config.universe import FULL_UNIVERSE, SECTOR_MAP
from src.analysis.risk_manager import RiskManager
from src.storage.models import OrderSide, PortfolioName, TradeSchema

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_trade(
    ticker: str, side: str, shares: float, price: float, portfolio: str = "A",
) -> TradeSchema:
    return TradeSchema(
        portfolio=PortfolioName(portfolio),
        ticker=ticker,
        side=OrderSide(side),
        shares=shares,
        price=price,
        reason="test",
    )


def _make_snapshot(snap_date: date, total_value: float):
    snap = MagicMock()
    snap.date = snap_date
    snap.total_value = total_value
    return snap


# ── TestCircuitBreakers ──────────────────────────────────────────────────


class TestCircuitBreakers:
    """Circuit breaker halts trading on large losses."""

    @pytest.mark.asyncio
    async def test_no_snapshots_no_halt(self):
        db = AsyncMock()
        db.get_snapshots = AsyncMock(return_value=[])
        rm = RiskManager()
        halted, reason = await rm.check_circuit_breakers("A", db, date.today())
        assert halted is False
        assert reason == ""

    @pytest.mark.asyncio
    async def test_daily_loss_triggers_halt(self):
        today = date(2026, 2, 6)
        db = AsyncMock()
        db.get_snapshots = AsyncMock(return_value=[
            _make_snapshot(today - timedelta(days=1), 33000.0),
            _make_snapshot(today, 31000.0),  # ~6% loss
        ])
        rm = RiskManager()
        halted, reason = await rm.check_circuit_breakers("A", db, today)
        assert halted is True
        assert "Daily loss" in reason

    @pytest.mark.asyncio
    async def test_weekly_loss_triggers_halt(self):
        today = date(2026, 2, 6)
        snapshots = [
            _make_snapshot(today - timedelta(days=5), 66000.0),
            _make_snapshot(today - timedelta(days=4), 64000.0),
            _make_snapshot(today - timedelta(days=3), 62000.0),
            _make_snapshot(today - timedelta(days=2), 60000.0),
            _make_snapshot(today - timedelta(days=1), 59500.0),
            _make_snapshot(today, 59000.0),  # ~10.6% weekly loss
        ]
        db = AsyncMock()
        db.get_snapshots = AsyncMock(return_value=snapshots)
        rm = RiskManager()
        halted, reason = await rm.check_circuit_breakers("B", db, today)
        assert halted is True
        assert "Weekly loss" in reason

    @pytest.mark.asyncio
    async def test_within_limits_no_halt(self):
        today = date(2026, 2, 6)
        db = AsyncMock()
        db.get_snapshots = AsyncMock(return_value=[
            _make_snapshot(today - timedelta(days=1), 33000.0),
            _make_snapshot(today, 32700.0),  # ~0.9% loss — within limits
        ])
        rm = RiskManager()
        halted, reason = await rm.check_circuit_breakers("A", db, today)
        assert halted is False


# ── TestPreTradeRisk ─────────────────────────────────────────────────────


class TestPreTradeRisk:
    """Pre-trade filter blocks concentration violations."""

    def test_clean_trades_pass(self):
        rm = RiskManager()
        trades = [_make_trade("XLK", "BUY", 10, 200.0)]
        verdict = rm.check_pre_trade(
            trades=trades,
            portfolio_name="A",
            current_positions={},
            latest_prices={"XLK": 200.0},
            portfolio_value=33000.0,
            cash=33000.0,
        )
        assert len(verdict.allowed) == 1
        assert len(verdict.blocked) == 0

    def test_oversized_position_blocked(self):
        rm = RiskManager()
        # Trying to buy 60 shares @ $200 = $12K of a $20K portfolio = 60%
        trades = [_make_trade("XLK", "BUY", 60, 200.0)]
        verdict = rm.check_pre_trade(
            trades=trades,
            portfolio_name="A",
            current_positions={"AAPL": 10},  # existing $2K position
            latest_prices={"XLK": 200.0, "AAPL": 200.0},
            portfolio_value=20000.0,
            cash=18000.0,
        )
        assert len(verdict.blocked) == 1
        assert "35%" in verdict.blocked[0][1]

    def test_sector_concentration_blocked(self):
        rm = RiskManager()
        # Already heavy in tech, trying to add more
        trades = [_make_trade("NVDA", "BUY", 50, 200.0)]
        verdict = rm.check_pre_trade(
            trades=trades,
            portfolio_name="A",
            current_positions={"AAPL": 50, "MSFT": 50},  # $10K+$10K tech
            latest_prices={"NVDA": 200.0, "AAPL": 200.0, "MSFT": 200.0},
            portfolio_value=30000.0,
            cash=10000.0,
        )
        assert len(verdict.blocked) == 1
        assert "Technology" in verdict.blocked[0][1]

    def test_tech_weight_blocked_portfolio_b_only(self):
        rm = RiskManager()
        # Tech ETFs (XLK, SMH, QQQ, ARKK) are checked for 40% cap in Portfolio B
        # Existing: XLK=50 shares @ $100 = $5K, SMH=50 @ $100 = $5K → $10K tech
        # Portfolio = $66K. Buying QQQ 150 shares @ $100 = $15K tech add → $25K/$66K = 37.8%
        # Still under 40% for position check, but let's push it over:
        # XLK=100 @ $100 = $10K, SMH=100 @ $100 = $10K → $20K tech.
        # Buy QQQ 80 @ $100 = $8K → total tech = $28K / $66K = 42% > 40%
        # But position check: $8K / $66K = 12% → passes
        # Sector check: Technology = ($10K + $10K + $8K) / $66K = 42% → passes (50% limit)
        # Tech ETF check: $28K / $66K = 42% → blocked!
        trades = [_make_trade("QQQ", "BUY", 80, 100.0, portfolio="B")]
        verdict_b = rm.check_pre_trade(
            trades=trades,
            portfolio_name="B",
            current_positions={"XLK": 100, "SMH": 100},
            latest_prices={"QQQ": 100.0, "XLK": 100.0, "SMH": 100.0},
            portfolio_value=66000.0,
            cash=46000.0,
        )
        assert len(verdict_b.blocked) == 1
        assert "Tech" in verdict_b.blocked[0][1]

        # Same trade should pass for Portfolio A (no tech ETF cap)
        trades_a = [_make_trade("QQQ", "BUY", 80, 100.0, portfolio="A")]
        verdict_a = rm.check_pre_trade(
            trades=trades_a,
            portfolio_name="A",
            current_positions={"XLK": 100, "SMH": 100},
            latest_prices={"QQQ": 100.0, "XLK": 100.0, "SMH": 100.0},
            portfolio_value=66000.0,
            cash=46000.0,
        )
        assert len(verdict_a.blocked) == 0

    def test_sells_always_pass(self):
        rm = RiskManager()
        trades = [_make_trade("XLK", "SELL", 100, 200.0)]
        verdict = rm.check_pre_trade(
            trades=trades,
            portfolio_name="A",
            current_positions={"XLK": 100},
            latest_prices={"XLK": 200.0},
            portfolio_value=20000.0,
            cash=0.0,
        )
        assert len(verdict.allowed) == 1
        assert len(verdict.blocked) == 0

    def test_partial_blocking(self):
        """Some trades pass, some fail."""
        rm = RiskManager()
        trades = [
            _make_trade("XLF", "BUY", 10, 40.0),   # small, should pass
            _make_trade("AAPL", "BUY", 200, 200.0),  # huge, should fail
        ]
        verdict = rm.check_pre_trade(
            trades=trades,
            portfolio_name="A",
            current_positions={},
            latest_prices={"XLF": 40.0, "AAPL": 200.0},
            portfolio_value=33000.0,
            cash=33000.0,
        )
        assert len(verdict.allowed) == 1
        assert verdict.allowed[0].ticker == "XLF"
        assert len(verdict.blocked) == 1
        assert verdict.blocked[0][0].ticker == "AAPL"


# ── TestSectorMap ────────────────────────────────────────────────────────


class TestSectorMap:
    """Verify SECTOR_MAP covers the universe."""

    def test_all_universe_tickers_mapped(self):
        unmapped = [t for t in FULL_UNIVERSE if t not in SECTOR_MAP]
        assert unmapped == [], f"Unmapped tickers: {unmapped}"

    def test_no_empty_sectors(self):
        sectors = set(SECTOR_MAP.values())
        assert len(sectors) >= 10  # At least 10 distinct sectors
        for sector in sectors:
            assert sector, "Empty sector name found"
