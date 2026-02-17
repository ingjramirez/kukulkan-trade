"""Tests for BudgetTracker — daily/monthly budget enforcement."""

from datetime import date

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.agent.budget_tracker import BudgetStatus, BudgetTracker
from src.agent.token_tracker import TokenTracker
from src.storage.database import Database
from src.storage.models import Base


@pytest.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    db_instance = Database.__new__(Database)
    db_instance._engine = engine
    db_instance._session_factory = session_factory
    db_instance._url = "sqlite+aiosqlite:///:memory:"
    yield db_instance
    await engine.dispose()


# ── BudgetStatus ─────────────────────────────────────────────────────────────


class TestBudgetStatus:
    def test_daily_remaining(self):
        status = BudgetStatus(daily_spent=1.0, daily_limit=3.0, monthly_spent=10.0, monthly_limit=75.0)
        assert status.daily_remaining == 2.0

    def test_daily_exhausted(self):
        status = BudgetStatus(daily_spent=3.0, daily_limit=3.0, monthly_spent=10.0, monthly_limit=75.0)
        assert status.daily_exhausted is True

    def test_daily_not_exhausted(self):
        status = BudgetStatus(daily_spent=1.0, daily_limit=3.0, monthly_spent=10.0, monthly_limit=75.0)
        assert status.daily_exhausted is False

    def test_monthly_exhausted(self):
        status = BudgetStatus(daily_spent=0.0, daily_limit=3.0, monthly_spent=75.0, monthly_limit=75.0)
        assert status.monthly_exhausted is True

    def test_haiku_only_at_80_pct(self):
        status = BudgetStatus(daily_spent=0.0, daily_limit=3.0, monthly_spent=60.0, monthly_limit=75.0)
        assert status.haiku_only is True  # 60 >= 75 * 0.80 = 60

    def test_not_haiku_only_below_80_pct(self):
        status = BudgetStatus(daily_spent=0.0, daily_limit=3.0, monthly_spent=50.0, monthly_limit=75.0)
        assert status.haiku_only is False

    def test_remaining_never_negative(self):
        status = BudgetStatus(daily_spent=5.0, daily_limit=3.0, monthly_spent=100.0, monthly_limit=75.0)
        assert status.daily_remaining == 0.0
        assert status.monthly_remaining == 0.0


# ── BudgetTracker ────────────────────────────────────────────────────────────


class TestBudgetTracker:
    @pytest.mark.asyncio
    async def test_check_budget_no_spend(self, db):
        tracker = BudgetTracker(db, daily_limit=3.0, monthly_limit=75.0)
        status = await tracker.check_budget("default", today=date(2026, 2, 15))
        assert status.daily_spent == 0.0
        assert status.monthly_spent == 0.0
        assert status.daily_exhausted is False

    @pytest.mark.asyncio
    async def test_record_and_check(self, db):
        tracker = BudgetTracker(db, daily_limit=3.0, monthly_limit=75.0)

        # Record a session
        tt = TokenTracker()
        tt.record("claude-sonnet-4-6", input_tokens=1000, output_tokens=500, turn=1)

        await tracker.record_session(
            tenant_id="default",
            session_date=date(2026, 2, 15),
            session_label="morning",
            session_id="test-session-1",
            token_tracker=tt,
            session_profile="full",
        )

        # Check budget
        status = await tracker.check_budget("default", today=date(2026, 2, 15))
        assert status.daily_spent > 0.0
        assert status.monthly_spent > 0.0

    @pytest.mark.asyncio
    async def test_daily_budget_exhausted(self, db):
        tracker = BudgetTracker(db, daily_limit=0.005, monthly_limit=75.0)

        # Record a session that exceeds daily limit
        tt = TokenTracker()
        tt.record("claude-sonnet-4-6", input_tokens=1000, output_tokens=500, turn=1)

        await tracker.record_session(
            tenant_id="default",
            session_date=date(2026, 2, 15),
            session_label="morning",
            session_id="test-session-1",
            token_tracker=tt,
        )

        status = await tracker.check_budget("default", today=date(2026, 2, 15))
        assert status.daily_exhausted is True

    @pytest.mark.asyncio
    async def test_different_dates_isolated(self, db):
        tracker = BudgetTracker(db, daily_limit=3.0, monthly_limit=75.0)

        tt = TokenTracker()
        tt.record("claude-sonnet-4-6", input_tokens=1000, output_tokens=500, turn=1)

        await tracker.record_session(
            tenant_id="default",
            session_date=date(2026, 2, 14),
            session_label="morning",
            session_id="session-14",
            token_tracker=tt,
        )

        # Check for Feb 15 — should be zero
        status = await tracker.check_budget("default", today=date(2026, 2, 15))
        assert status.daily_spent == 0.0
        # But monthly should have the Feb 14 spend
        assert status.monthly_spent > 0.0

    @pytest.mark.asyncio
    async def test_different_tenants_isolated(self, db):
        tracker = BudgetTracker(db, daily_limit=3.0, monthly_limit=75.0)

        tt = TokenTracker()
        tt.record("claude-sonnet-4-6", input_tokens=1000, output_tokens=500, turn=1)

        await tracker.record_session(
            tenant_id="tenant-a",
            session_date=date(2026, 2, 15),
            session_label="morning",
            session_id="session-a",
            token_tracker=tt,
        )

        # Check for tenant-b — should be zero
        status = await tracker.check_budget("tenant-b", today=date(2026, 2, 15))
        assert status.daily_spent == 0.0
        assert status.monthly_spent == 0.0
