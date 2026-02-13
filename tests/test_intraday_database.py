"""Tests for intraday snapshot CRUD methods in Database."""

from datetime import datetime, timedelta, timezone

import pytest

from src.storage.database import Database


@pytest.fixture
async def db():
    test_db = Database(url="sqlite+aiosqlite:///:memory:")
    await test_db.init_db()
    yield test_db
    await test_db.close()


# ── save_intraday_snapshot ─────────────────────────────────────────


async def test_save_intraday_snapshot(db: Database):
    ts = datetime(2026, 2, 13, 14, 30, tzinfo=timezone.utc)
    await db.save_intraday_snapshot(
        tenant_id="t-1",
        portfolio="A",
        timestamp=ts,
        total_value=35000.0,
        cash=30000.0,
        positions_value=5000.0,
    )
    rows = await db.get_intraday_snapshots("t-1", portfolio="A")
    assert len(rows) == 1
    assert rows[0].total_value == 35000.0
    assert rows[0].cash == 30000.0
    assert rows[0].positions_value == 5000.0
    assert rows[0].portfolio == "A"


async def test_save_intraday_snapshot_replaces_on_conflict(db: Database):
    ts = datetime(2026, 2, 13, 14, 30, tzinfo=timezone.utc)
    await db.save_intraday_snapshot(
        tenant_id="t-1",
        portfolio="A",
        timestamp=ts,
        total_value=35000.0,
        cash=30000.0,
        positions_value=5000.0,
    )
    # Save again with updated values — should replace
    await db.save_intraday_snapshot(
        tenant_id="t-1",
        portfolio="A",
        timestamp=ts,
        total_value=36000.0,
        cash=31000.0,
        positions_value=5000.0,
    )
    rows = await db.get_intraday_snapshots("t-1", portfolio="A")
    assert len(rows) == 1
    assert rows[0].total_value == 36000.0


async def test_save_multiple_portfolios(db: Database):
    ts = datetime(2026, 2, 13, 14, 30, tzinfo=timezone.utc)
    await db.save_intraday_snapshot(
        tenant_id="t-1",
        portfolio="A",
        timestamp=ts,
        total_value=35000.0,
        cash=30000.0,
        positions_value=5000.0,
    )
    await db.save_intraday_snapshot(
        tenant_id="t-1",
        portfolio="B",
        timestamp=ts,
        total_value=68000.0,
        cash=60000.0,
        positions_value=8000.0,
    )
    all_rows = await db.get_intraday_snapshots("t-1")
    assert len(all_rows) == 2
    a_rows = await db.get_intraday_snapshots("t-1", portfolio="A")
    assert len(a_rows) == 1
    b_rows = await db.get_intraday_snapshots("t-1", portfolio="B")
    assert len(b_rows) == 1


# ── get_intraday_snapshots ─────────────────────────────────────────


async def test_get_intraday_snapshots_ordered_by_timestamp(db: Database):
    base = datetime(2026, 2, 13, 10, 0, tzinfo=timezone.utc)
    for i in range(5):
        ts = base + timedelta(minutes=15 * i)
        await db.save_intraday_snapshot(
            tenant_id="t-1",
            portfolio="B",
            timestamp=ts,
            total_value=68000.0 + i * 100,
            cash=60000.0,
            positions_value=8000.0 + i * 100,
        )
    rows = await db.get_intraday_snapshots("t-1", portfolio="B")
    assert len(rows) == 5
    # Check ascending order
    for i in range(len(rows) - 1):
        assert rows[i].timestamp <= rows[i + 1].timestamp


async def test_get_intraday_snapshots_filter_since(db: Database):
    base = datetime(2026, 2, 13, 10, 0, tzinfo=timezone.utc)
    for i in range(4):
        ts = base + timedelta(hours=i)
        await db.save_intraday_snapshot(
            tenant_id="t-1",
            portfolio="A",
            timestamp=ts,
            total_value=35000.0,
            cash=30000.0,
            positions_value=5000.0,
        )
    since = base + timedelta(hours=2)
    rows = await db.get_intraday_snapshots("t-1", portfolio="A", since=since)
    assert len(rows) == 2


async def test_get_intraday_snapshots_filter_until(db: Database):
    base = datetime(2026, 2, 13, 10, 0, tzinfo=timezone.utc)
    for i in range(4):
        ts = base + timedelta(hours=i)
        await db.save_intraday_snapshot(
            tenant_id="t-1",
            portfolio="A",
            timestamp=ts,
            total_value=35000.0,
            cash=30000.0,
            positions_value=5000.0,
        )
    until = base + timedelta(hours=1)
    rows = await db.get_intraday_snapshots("t-1", portfolio="A", until=until)
    assert len(rows) == 2


async def test_get_intraday_snapshots_tenant_isolation(db: Database):
    ts = datetime(2026, 2, 13, 14, 30, tzinfo=timezone.utc)
    await db.save_intraday_snapshot(
        tenant_id="t-1",
        portfolio="A",
        timestamp=ts,
        total_value=35000.0,
        cash=30000.0,
        positions_value=5000.0,
    )
    await db.save_intraday_snapshot(
        tenant_id="t-2",
        portfolio="A",
        timestamp=ts,
        total_value=50000.0,
        cash=45000.0,
        positions_value=5000.0,
    )
    t1_rows = await db.get_intraday_snapshots("t-1")
    t2_rows = await db.get_intraday_snapshots("t-2")
    assert len(t1_rows) == 1
    assert len(t2_rows) == 1
    assert t1_rows[0].total_value == 35000.0
    assert t2_rows[0].total_value == 50000.0


async def test_get_intraday_snapshots_empty(db: Database):
    rows = await db.get_intraday_snapshots("nonexistent")
    assert rows == []


# ── purge_old_intraday_snapshots ───────────────────────────────────


async def test_purge_old_intraday_snapshots(db: Database):
    now = datetime.now(timezone.utc)
    old_ts = now - timedelta(days=100)
    recent_ts = now - timedelta(days=10)

    await db.save_intraday_snapshot(
        tenant_id="t-1",
        portfolio="A",
        timestamp=old_ts,
        total_value=35000.0,
        cash=30000.0,
        positions_value=5000.0,
    )
    await db.save_intraday_snapshot(
        tenant_id="t-1",
        portfolio="A",
        timestamp=recent_ts,
        total_value=36000.0,
        cash=31000.0,
        positions_value=5000.0,
    )

    deleted = await db.purge_old_intraday_snapshots(days=90)
    assert deleted == 1

    rows = await db.get_intraday_snapshots("t-1")
    assert len(rows) == 1
    assert rows[0].total_value == 36000.0


async def test_purge_old_intraday_snapshots_none_old(db: Database):
    now = datetime.now(timezone.utc)
    recent_ts = now - timedelta(days=10)

    await db.save_intraday_snapshot(
        tenant_id="t-1",
        portfolio="B",
        timestamp=recent_ts,
        total_value=68000.0,
        cash=60000.0,
        positions_value=8000.0,
    )

    deleted = await db.purge_old_intraday_snapshots(days=90)
    assert deleted == 0
