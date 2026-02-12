"""Tests for earnings calendar: DB CRUD, cache staleness, upcoming filtering."""

from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from src.data.earnings_calendar import EarningsCalendar
from src.storage.database import Database


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


# ── DB CRUD ───────────────────────────────────────────────────────────


async def test_upsert_earnings(db: Database):
    """Upsert creates a new row."""
    future = date.today() + timedelta(days=7)
    await db.upsert_earnings("AAPL", future)

    rows = await db.get_upcoming_earnings(["AAPL"], days_ahead=14)
    assert len(rows) == 1
    assert rows[0].ticker == "AAPL"
    assert rows[0].earnings_date == future


async def test_upsert_earnings_updates_existing(db: Database):
    """Upsert with same ticker+date updates instead of duplicating."""
    future = date.today() + timedelta(days=7)
    await db.upsert_earnings("AAPL", future, source="yfinance")
    await db.upsert_earnings("AAPL", future, source="manual")

    rows = await db.get_upcoming_earnings(["AAPL"], days_ahead=14)
    assert len(rows) == 1
    assert rows[0].source == "manual"


async def test_get_upcoming_earnings_filters_by_range(db: Database):
    """Only earnings within days_ahead window are returned."""
    soon = date.today() + timedelta(days=3)
    far = date.today() + timedelta(days=30)
    await db.upsert_earnings("AAPL", soon)
    await db.upsert_earnings("MSFT", far)

    rows = await db.get_upcoming_earnings(["AAPL", "MSFT"], days_ahead=14)
    assert len(rows) == 1
    assert rows[0].ticker == "AAPL"


async def test_get_upcoming_earnings_filters_by_tickers(db: Database):
    """Only earnings for requested tickers are returned."""
    future = date.today() + timedelta(days=5)
    await db.upsert_earnings("AAPL", future)
    await db.upsert_earnings("MSFT", future)

    rows = await db.get_upcoming_earnings(["AAPL"], days_ahead=14)
    assert len(rows) == 1
    assert rows[0].ticker == "AAPL"


async def test_cleanup_past_earnings(db: Database):
    """cleanup_past_earnings removes rows with past dates."""
    past = date.today() - timedelta(days=1)
    future = date.today() + timedelta(days=5)
    await db.upsert_earnings("OLD", past)
    await db.upsert_earnings("NEW", future)

    count = await db.cleanup_past_earnings()
    assert count == 1

    # Only future remains
    rows = await db.get_upcoming_earnings(["OLD", "NEW"], days_ahead=14)
    assert len(rows) == 1
    assert rows[0].ticker == "NEW"


async def test_get_latest_earnings_fetch(db: Database):
    """get_latest_earnings_fetch returns the most recent fetched_at."""
    future = date.today() + timedelta(days=5)
    await db.upsert_earnings("AAPL", future)

    latest = await db.get_latest_earnings_fetch()
    assert latest is not None
    # SQLite returns naive datetimes, compare with naive now
    assert (datetime.utcnow() - latest).total_seconds() < 5


async def test_get_latest_earnings_fetch_empty(db: Database):
    """Returns None when table is empty."""
    latest = await db.get_latest_earnings_fetch()
    assert latest is None


# ── EarningsCalendar class ────────────────────────────────────────────


async def test_refresh_skips_when_cache_fresh(db: Database):
    """refresh_earnings skips fetch if cache is less than TTL hours old."""
    future = date.today() + timedelta(days=5)
    await db.upsert_earnings("AAPL", future)

    cal = EarningsCalendar()
    count = await cal.refresh_earnings(db, ["AAPL", "MSFT"])
    assert count == 0  # Skipped because recent fetch exists


async def test_refresh_fetches_when_cache_stale(db: Database):
    """refresh_earnings fetches when cache is older than TTL."""
    cal = EarningsCalendar()

    future = date.today() + timedelta(days=10)
    mock_fetch = AsyncMock(return_value=future)

    with patch.object(cal, "_fetch_earnings_date", mock_fetch):
        count = await cal.refresh_earnings(db, ["AAPL"])

    assert count == 1
    rows = await db.get_upcoming_earnings(["AAPL"], days_ahead=14)
    assert len(rows) == 1


async def test_refresh_handles_fetch_failure(db: Database):
    """Individual ticker failures don't crash the batch."""
    cal = EarningsCalendar()

    async def _flaky(ticker: str) -> date | None:
        if ticker == "BAD":
            raise ValueError("yfinance error")
        return date.today() + timedelta(days=5)

    with patch.object(cal, "_fetch_earnings_date", side_effect=_flaky):
        count = await cal.refresh_earnings(db, ["AAPL", "BAD", "MSFT"])

    assert count == 2  # AAPL + MSFT, BAD failed gracefully


async def test_get_upcoming_delegates_to_db(db: Database):
    """get_upcoming delegates to db.get_upcoming_earnings."""
    future = date.today() + timedelta(days=3)
    await db.upsert_earnings("AAPL", future)

    cal = EarningsCalendar()
    rows = await cal.get_upcoming(db, ["AAPL"], days_ahead=7)
    assert len(rows) == 1
    assert rows[0].ticker == "AAPL"
