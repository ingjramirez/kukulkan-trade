"""Tests for dynamic watchlist: CRUD, expiry, AI parsing, tenant isolation."""

from datetime import date, timedelta

import pytest

from src.storage.database import Database


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


# ── CRUD ──────────────────────────────────────────────────────────────


async def test_upsert_watchlist_item(db: Database):
    """upsert_watchlist_item creates a new row."""
    await db.upsert_watchlist_item(
        tenant_id="t1",
        ticker="PLTR",
        reason="AI rotation",
        conviction="high",
        target_entry=22.50,
        expires_at=date.today() + timedelta(days=14),
    )
    items = await db.get_watchlist("t1")
    assert len(items) == 1
    assert items[0].ticker == "PLTR"
    assert items[0].conviction == "high"
    assert items[0].target_entry == 22.50


async def test_upsert_updates_existing(db: Database):
    """Upserting same ticker updates instead of duplicating."""
    expires = date.today() + timedelta(days=14)
    await db.upsert_watchlist_item(
        "t1",
        "PLTR",
        "old reason",
        "low",
        expires_at=expires,
    )
    await db.upsert_watchlist_item(
        "t1",
        "PLTR",
        "new reason",
        "high",
        target_entry=25.0,
        expires_at=expires,
    )
    items = await db.get_watchlist("t1")
    assert len(items) == 1
    assert items[0].reason == "new reason"
    assert items[0].conviction == "high"
    assert items[0].target_entry == 25.0


async def test_get_watchlist_filters_by_portfolio(db: Database):
    """get_watchlist filters by portfolio."""
    expires = date.today() + timedelta(days=14)
    await db.upsert_watchlist_item(
        "t1",
        "PLTR",
        "reason",
        portfolio="B",
        expires_at=expires,
    )
    await db.upsert_watchlist_item(
        "t1",
        "XLK",
        "reason",
        portfolio="A",
        expires_at=expires,
    )
    b_items = await db.get_watchlist("t1", "B")
    assert len(b_items) == 1
    assert b_items[0].ticker == "PLTR"

    a_items = await db.get_watchlist("t1", "A")
    assert len(a_items) == 1
    assert a_items[0].ticker == "XLK"


async def test_remove_watchlist_item(db: Database):
    """remove_watchlist_item deletes the row."""
    expires = date.today() + timedelta(days=14)
    await db.upsert_watchlist_item("t1", "PLTR", "reason", expires_at=expires)
    await db.remove_watchlist_item("t1", "PLTR")
    items = await db.get_watchlist("t1")
    assert len(items) == 0


async def test_remove_nonexistent_item_is_noop(db: Database):
    """Removing a non-existent item doesn't raise."""
    await db.remove_watchlist_item("t1", "DOESNTEXIST")


# ── Expiry ────────────────────────────────────────────────────────────


async def test_cleanup_expired_watchlist(db: Database):
    """cleanup_expired_watchlist removes items past expiry."""
    await db.upsert_watchlist_item(
        "t1",
        "OLD",
        "expired",
        expires_at=date.today() - timedelta(days=1),
    )
    await db.upsert_watchlist_item(
        "t1",
        "NEW",
        "fresh",
        expires_at=date.today() + timedelta(days=7),
    )
    count = await db.cleanup_expired_watchlist("t1")
    assert count == 1

    items = await db.get_watchlist("t1")
    assert len(items) == 1
    assert items[0].ticker == "NEW"


async def test_cleanup_scoped_to_tenant(db: Database):
    """cleanup_expired_watchlist only affects the specified tenant."""
    past = date.today() - timedelta(days=1)
    await db.upsert_watchlist_item("t1", "OLD1", "reason", expires_at=past)
    await db.upsert_watchlist_item("t2", "OLD2", "reason", expires_at=past)

    count = await db.cleanup_expired_watchlist("t1")
    assert count == 1

    # t2 still has its expired item
    items_t2 = await db.get_watchlist("t2")
    assert len(items_t2) == 1


# ── Auto-promote (trade removes from watchlist) ──────────────────────


async def test_remove_watchlist_if_traded(db: Database):
    """Trading a watchlist ticker removes it."""
    expires = date.today() + timedelta(days=14)
    await db.upsert_watchlist_item("t1", "PLTR", "watching", expires_at=expires)
    await db.remove_watchlist_if_traded("t1", "PLTR")
    items = await db.get_watchlist("t1")
    assert len(items) == 0


# ── Tenant isolation ────────────────────────────────────────────────


async def test_tenant_isolation(db: Database):
    """Watchlist items are scoped per tenant."""
    expires = date.today() + timedelta(days=14)
    await db.upsert_watchlist_item("t1", "PLTR", "reason1", expires_at=expires)
    await db.upsert_watchlist_item("t2", "MSFT", "reason2", expires_at=expires)

    t1 = await db.get_watchlist("t1")
    t2 = await db.get_watchlist("t2")
    assert len(t1) == 1
    assert t1[0].ticker == "PLTR"
    assert len(t2) == 1
    assert t2[0].ticker == "MSFT"


# ── AI response parsing (orchestrator) ──────────────────────────────


async def test_process_watchlist_updates_add(db: Database):
    """_process_watchlist_updates handles add action."""
    from src.notifications.telegram_bot import TelegramNotifier
    from src.orchestrator import Orchestrator

    orch = Orchestrator(db, notifier=TelegramNotifier("fake", "fake"))
    today = date.today()
    updates = [
        {
            "action": "add",
            "ticker": "PLTR",
            "reason": "AI rotation",
            "conviction": "high",
            "target_entry": 22.50,
        },
    ]
    await orch._process_watchlist_updates(updates, "t1", today)

    items = await db.get_watchlist("t1")
    assert len(items) == 1
    assert items[0].ticker == "PLTR"
    assert items[0].conviction == "high"
    assert items[0].expires_at == today + timedelta(days=14)


async def test_process_watchlist_updates_remove(db: Database):
    """_process_watchlist_updates handles remove action."""
    from src.notifications.telegram_bot import TelegramNotifier
    from src.orchestrator import Orchestrator

    expires = date.today() + timedelta(days=14)
    await db.upsert_watchlist_item("t1", "COIN", "reason", expires_at=expires)

    orch = Orchestrator(db, notifier=TelegramNotifier("fake", "fake"))
    updates = [{"action": "remove", "ticker": "COIN"}]
    await orch._process_watchlist_updates(updates, "t1", date.today())

    items = await db.get_watchlist("t1")
    assert len(items) == 0


async def test_process_watchlist_updates_empty_is_noop(db: Database):
    """Empty updates list is a no-op."""
    from src.notifications.telegram_bot import TelegramNotifier
    from src.orchestrator import Orchestrator

    orch = Orchestrator(db, notifier=TelegramNotifier("fake", "fake"))
    await orch._process_watchlist_updates([], "t1", date.today())
