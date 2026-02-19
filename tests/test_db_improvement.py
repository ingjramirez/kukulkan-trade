"""Tests for improvement snapshot and parameter changelog CRUD + trailing_stop_multiplier."""

from datetime import date

import pytest

from src.storage.database import Database


@pytest.fixture
async def db():
    d = Database(url="sqlite+aiosqlite:///:memory:")
    await d.init_db()
    yield d
    await d.close()


# ── Improvement Snapshot CRUD ─────────────────────────────────────


async def test_save_and_get_improvement_snapshot(db: Database):
    snap_id = await db.save_improvement_snapshot(
        tenant_id="default",
        week_start=date(2026, 2, 10),
        week_end=date(2026, 2, 17),
        total_trades=15,
        win_rate_pct=60.0,
        avg_pnl_pct=1.5,
        avg_alpha_vs_spy=0.8,
        total_cost_usd=2.50,
        strategy_mode="conservative",
        trailing_stop_multiplier=1.0,
        proposal_json='{"changes": []}',
        applied_changes="[]",
        report_text="Good week",
    )
    assert snap_id > 0

    row = await db.get_improvement_snapshot(snap_id, tenant_id="default")
    assert row is not None
    assert row.total_trades == 15
    assert row.win_rate_pct == 60.0
    assert row.strategy_mode == "conservative"
    assert row.report_text == "Good week"


async def test_get_improvement_snapshots_list(db: Database):
    for i in range(3):
        await db.save_improvement_snapshot(
            tenant_id="default",
            week_start=date(2026, 2, 3 + i * 7),
            week_end=date(2026, 2, 10 + i * 7),
            total_trades=10 + i,
            win_rate_pct=50.0 + i,
            avg_pnl_pct=1.0,
            avg_alpha_vs_spy=None,
            total_cost_usd=1.0,
            strategy_mode="conservative",
            trailing_stop_multiplier=1.0,
            proposal_json=None,
            applied_changes=None,
            report_text=None,
        )

    snapshots = await db.get_improvement_snapshots(tenant_id="default")
    assert len(snapshots) == 3
    # Newest first
    assert snapshots[0].total_trades == 12
    assert snapshots[2].total_trades == 10


async def test_get_improvement_snapshots_limit(db: Database):
    from datetime import timedelta

    base = date(2026, 1, 6)
    for i in range(5):
        ws = base + timedelta(weeks=i)
        we = ws + timedelta(days=7)
        await db.save_improvement_snapshot(
            tenant_id="default",
            week_start=ws,
            week_end=we,
            total_trades=i,
            win_rate_pct=50.0,
            avg_pnl_pct=0.0,
            avg_alpha_vs_spy=None,
            total_cost_usd=0.0,
            strategy_mode="conservative",
            trailing_stop_multiplier=1.0,
            proposal_json=None,
            applied_changes=None,
            report_text=None,
        )

    snapshots = await db.get_improvement_snapshots(tenant_id="default", limit=2)
    assert len(snapshots) == 2


async def test_get_improvement_snapshot_not_found(db: Database):
    row = await db.get_improvement_snapshot(999, tenant_id="default")
    assert row is None


async def test_improvement_snapshot_tenant_isolation(db: Database):
    await db.ensure_tenant("t1")
    await db.save_improvement_snapshot(
        tenant_id="default",
        week_start=date(2026, 2, 10),
        week_end=date(2026, 2, 17),
        total_trades=5,
        win_rate_pct=50.0,
        avg_pnl_pct=0.0,
        avg_alpha_vs_spy=None,
        total_cost_usd=0.0,
        strategy_mode="conservative",
        trailing_stop_multiplier=1.0,
        proposal_json=None,
        applied_changes=None,
        report_text=None,
    )

    # t1 should see nothing
    snapshots = await db.get_improvement_snapshots(tenant_id="t1")
    assert len(snapshots) == 0


# ── Parameter Changelog CRUD ─────────────────────────────────────


async def test_insert_and_get_parameter_changelog(db: Database):
    await db.insert_parameter_changelog(
        tenant_id="default",
        parameter="strategy_mode",
        old_value="conservative",
        new_value="standard",
        reason="Higher win rate supports standard mode",
    )

    entries = await db.get_parameter_changelog(tenant_id="default")
    assert len(entries) == 1
    assert entries[0].parameter == "strategy_mode"
    assert entries[0].old_value == "conservative"
    assert entries[0].new_value == "standard"


async def test_parameter_changelog_with_snapshot_id(db: Database):
    snap_id = await db.save_improvement_snapshot(
        tenant_id="default",
        week_start=date(2026, 2, 10),
        week_end=date(2026, 2, 17),
        total_trades=10,
        win_rate_pct=55.0,
        avg_pnl_pct=1.0,
        avg_alpha_vs_spy=0.5,
        total_cost_usd=1.0,
        strategy_mode="conservative",
        trailing_stop_multiplier=1.0,
        proposal_json=None,
        applied_changes=None,
        report_text=None,
    )

    await db.insert_parameter_changelog(
        tenant_id="default",
        parameter="trailing_stop_multiplier",
        old_value="1.0",
        new_value="0.8",
        reason="Tighter stops needed",
        snapshot_id=snap_id,
    )

    entries = await db.get_parameter_changelog(tenant_id="default")
    assert entries[0].snapshot_id == snap_id


async def test_get_parameter_changes_for(db: Database):
    await db.insert_parameter_changelog(
        tenant_id="default",
        parameter="strategy_mode",
        old_value="conservative",
        new_value="standard",
    )
    await db.insert_parameter_changelog(
        tenant_id="default",
        parameter="trailing_stop_multiplier",
        old_value="1.0",
        new_value="0.8",
    )
    await db.insert_parameter_changelog(
        tenant_id="default",
        parameter="strategy_mode",
        old_value="standard",
        new_value="aggressive",
    )

    strategy_changes = await db.get_parameter_changes_for("default", "strategy_mode")
    assert len(strategy_changes) == 2

    trail_changes = await db.get_parameter_changes_for("default", "trailing_stop_multiplier")
    assert len(trail_changes) == 1


async def test_parameter_changelog_tenant_isolation(db: Database):
    await db.ensure_tenant("t2")
    await db.insert_parameter_changelog(
        tenant_id="default",
        parameter="strategy_mode",
        old_value="conservative",
        new_value="standard",
    )

    entries = await db.get_parameter_changelog(tenant_id="t2")
    assert len(entries) == 0


# ── Trailing Stop Multiplier on TenantRow ────────────────────────


async def test_trailing_stop_multiplier_default(db: Database):
    tenant = await db.get_tenant("default")
    assert tenant is not None
    assert tenant.trailing_stop_multiplier == 1.0


async def test_trailing_stop_multiplier_update(db: Database):
    await db.update_tenant("default", {"trailing_stop_multiplier": 0.8})
    tenant = await db.get_tenant("default")
    assert tenant.trailing_stop_multiplier == 0.8
