"""Tests for the PerformanceTracker."""

from datetime import date

import pytest

from src.analysis.performance import PerformanceTracker
from src.storage.database import Database


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


@pytest.fixture
def tracker():
    return PerformanceTracker()


class TestPerformanceTracker:
    async def test_empty_portfolio(self, db: Database, tracker) -> None:
        """No snapshots returns initial values."""
        stats = await tracker.get_portfolio_stats(db, "A", 33_000.0)

        assert stats.current_value == 33_000.0
        assert stats.inception_return_pct == 0.0
        assert stats.days_tracked == 0
        assert stats.total_trades == 0

    async def test_inception_return(self, db: Database, tracker) -> None:
        """Inception return computed from initial vs current value."""
        await db.upsert_portfolio("A", cash=33_000.0, total_value=33_000.0)
        await db.save_snapshot("A", date(2026, 1, 1), 33_000.0, 33_000.0, 0.0, None, 0.0)
        await db.save_snapshot("A", date(2026, 1, 2), 34_650.0, 34_650.0, 0.0, 5.0, 5.0)

        stats = await tracker.get_portfolio_stats(db, "A", 33_000.0)

        assert stats.current_value == 34_650.0
        assert stats.inception_return_pct == 5.0
        assert stats.days_tracked == 2

    async def test_drawdown_calculation(self, db: Database, tracker) -> None:
        """Max drawdown from peak to trough."""
        await db.upsert_portfolio("A", cash=33_000.0, total_value=33_000.0)
        await db.save_snapshot("A", date(2026, 1, 1), 33_000.0, 33_000.0, 0.0, None, 0.0)
        await db.save_snapshot("A", date(2026, 1, 2), 35_000.0, 35_000.0, 0.0, 6.06, 6.06)
        await db.save_snapshot("A", date(2026, 1, 3), 32_000.0, 32_000.0, 0.0, -8.57, -3.03)

        stats = await tracker.get_portfolio_stats(db, "A", 33_000.0)

        # Drawdown from peak (35000) to trough (32000) = 8.57%
        assert stats.peak_value == 35_000.0
        assert stats.drawdown_pct == pytest.approx(8.57, abs=0.01)

    async def test_best_worst_day(self, db: Database, tracker) -> None:
        """Best and worst daily returns identified."""
        await db.upsert_portfolio("A", cash=33_000.0, total_value=33_000.0)
        await db.save_snapshot("A", date(2026, 1, 1), 33_000.0, 33_000.0, 0.0, None, 0.0)
        await db.save_snapshot("A", date(2026, 1, 2), 34_000.0, 34_000.0, 0.0, 3.03, 3.03)
        await db.save_snapshot("A", date(2026, 1, 3), 32_500.0, 32_500.0, 0.0, -4.41, -1.52)
        await db.save_snapshot("A", date(2026, 1, 4), 33_500.0, 33_500.0, 0.0, 3.08, 1.52)

        stats = await tracker.get_portfolio_stats(db, "A", 33_000.0)

        assert stats.best_day_pct == 3.08
        assert stats.worst_day_pct == -4.41

    async def test_win_rate_from_positive_days(self, db: Database, tracker) -> None:
        """Win rate computed from daily returns when no trades."""
        await db.upsert_portfolio("B", cash=66_000.0, total_value=66_000.0)
        await db.save_snapshot("B", date(2026, 1, 1), 66_000.0, 66_000.0, 0.0, 1.0, 1.0)
        await db.save_snapshot("B", date(2026, 1, 2), 67_000.0, 67_000.0, 0.0, 1.5, 1.5)
        await db.save_snapshot("B", date(2026, 1, 3), 65_000.0, 65_000.0, 0.0, -3.0, -1.5)

        stats = await tracker.get_portfolio_stats(db, "B", 66_000.0)

        # 2 positive days, 1 negative → 66.67% win rate
        assert stats.win_rate_pct == pytest.approx(66.67, abs=0.01)

    async def test_portfolio_b(self, db: Database, tracker) -> None:
        """Stats for Portfolio B with different initial value."""
        await db.upsert_portfolio("B", cash=66_000.0, total_value=66_000.0)
        await db.save_snapshot("B", date(2026, 1, 1), 66_000.0, 66_000.0, 0.0, None, 0.0)
        await db.save_snapshot("B", date(2026, 1, 2), 69_300.0, 69_300.0, 0.0, 5.0, 5.0)

        stats = await tracker.get_portfolio_stats(db, "B", 66_000.0)

        assert stats.portfolio == "B"
        assert stats.inception_return_pct == 5.0
        assert stats.initial_value == 66_000.0

    async def test_format_for_prompt(self, db: Database, tracker) -> None:
        """Format produces readable text for the agent prompt."""
        await db.upsert_portfolio("B", cash=66_000.0, total_value=66_000.0)
        await db.save_snapshot("B", date(2026, 1, 1), 66_000.0, 66_000.0, 0.0, 2.0, 2.0)
        await db.save_snapshot("B", date(2026, 1, 2), 69_300.0, 69_300.0, 0.0, 5.0, 5.0)

        stats = await tracker.get_portfolio_stats(db, "B", 66_000.0)
        text = tracker.format_for_prompt(stats)

        assert "Portfolio B Performance:" in text
        assert "$69,300.00" in text
        assert "+5.00%" in text

    async def test_no_daily_returns(self, db: Database, tracker) -> None:
        """Handles snapshots with no daily_return_pct."""
        await db.upsert_portfolio("A", cash=33_000.0, total_value=33_000.0)
        await db.save_snapshot("A", date(2026, 1, 1), 33_000.0, 33_000.0, 0.0, None, 0.0)

        stats = await tracker.get_portfolio_stats(db, "A", 33_000.0)

        assert stats.best_day_pct is None
        assert stats.worst_day_pct is None
        assert stats.win_rate_pct is None
