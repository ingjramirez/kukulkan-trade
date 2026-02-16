"""Tests for RiskManager.check_inverse_hold_times()."""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import update

from src.analysis.risk_manager import RiskManager
from src.storage.database import Database
from src.storage.models import TradeRow


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


@pytest.fixture
def rm() -> RiskManager:
    return RiskManager()


async def _seed_inverse_position(
    db: Database,
    ticker: str = "SH",
    shares: float = 100,
    buy_days_ago: int = 0,
) -> None:
    """Seed a portfolio with an inverse position and BUY trade."""
    await db.upsert_portfolio("B", cash=50_000.0, total_value=100_000.0)
    await db.upsert_position("B", ticker, shares=shares, avg_price=15.0)
    await db.log_trade("B", ticker, "BUY", shares, 15.0, reason="test hedge")

    # Override executed_at to simulate days held
    target_time = datetime.now(timezone.utc) - timedelta(days=buy_days_ago)
    # SQLite stores naive datetimes, so strip tzinfo
    target_naive = target_time.replace(tzinfo=None)
    async with db.session() as s:
        stmt = (
            update(TradeRow).where(TradeRow.ticker == ticker, TradeRow.side == "BUY").values(executed_at=target_naive)
        )
        await s.execute(stmt)
        await s.commit()


class TestHoldTimesEmpty:
    async def test_no_inverse_positions(self, db: Database, rm: RiskManager) -> None:
        await db.upsert_portfolio("B", cash=100_000.0, total_value=100_000.0)
        await db.upsert_position("B", "XLK", shares=100, avg_price=200.0)
        alerts = await rm.check_inverse_hold_times(db, "B")
        assert alerts == []

    async def test_no_positions_at_all(self, db: Database, rm: RiskManager) -> None:
        alerts = await rm.check_inverse_hold_times(db, "B")
        assert alerts == []


class TestHoldTimeWarning:
    async def test_warning_at_3_days(self, db: Database, rm: RiskManager) -> None:
        await _seed_inverse_position(db, "SH", buy_days_ago=3)
        alerts = await rm.check_inverse_hold_times(db, "B")
        assert len(alerts) == 1
        assert alerts[0]["alert_level"] == "warning"
        assert alerts[0]["days_held"] >= 3

    async def test_warning_at_4_days(self, db: Database, rm: RiskManager) -> None:
        await _seed_inverse_position(db, "SH", buy_days_ago=4)
        alerts = await rm.check_inverse_hold_times(db, "B")
        assert len(alerts) == 1
        assert alerts[0]["alert_level"] == "warning"


class TestHoldTimeReview:
    async def test_review_at_5_days(self, db: Database, rm: RiskManager) -> None:
        await _seed_inverse_position(db, "SH", buy_days_ago=5)
        alerts = await rm.check_inverse_hold_times(db, "B")
        assert len(alerts) == 1
        assert alerts[0]["alert_level"] == "review"

    async def test_review_at_7_days(self, db: Database, rm: RiskManager) -> None:
        await _seed_inverse_position(db, "SH", buy_days_ago=7)
        alerts = await rm.check_inverse_hold_times(db, "B")
        assert len(alerts) == 1
        assert alerts[0]["alert_level"] == "review"


class TestHoldTimeNoAlert:
    async def test_no_alert_at_0_days(self, db: Database, rm: RiskManager) -> None:
        await _seed_inverse_position(db, "SH", buy_days_ago=0)
        alerts = await rm.check_inverse_hold_times(db, "B")
        assert alerts == []

    async def test_no_alert_at_2_days(self, db: Database, rm: RiskManager) -> None:
        await _seed_inverse_position(db, "SH", buy_days_ago=2)
        alerts = await rm.check_inverse_hold_times(db, "B")
        assert alerts == []
