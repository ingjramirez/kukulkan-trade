"""Tests for DB CRUD methods: posture, playbook snapshots, conviction calibration."""

import asyncio
from datetime import date

import pytest

from src.storage.database import Database


@pytest.fixture
async def db():
    test_db = Database(url="sqlite+aiosqlite:///:memory:")
    await test_db.init_db()
    yield test_db
    await test_db.close()


# ── Posture CRUD ─────────────────────────────────────────────────────────────


class TestPostureCRUD:
    async def test_save_and_get_current_posture(self, db: Database) -> None:
        await db.save_posture(
            tenant_id="default",
            session_date=date(2026, 2, 15),
            session_label="Morning",
            posture="defensive",
            effective_posture="defensive",
            reason="VIX elevated above 25",
        )

        row = await db.get_current_posture(tenant_id="default")
        assert row is not None
        assert row.session_date == date(2026, 2, 15)
        assert row.session_label == "Morning"
        assert row.posture == "defensive"
        assert row.effective_posture == "defensive"
        assert row.reason == "VIX elevated above 25"
        assert row.tenant_id == "default"

    async def test_get_current_posture_empty(self, db: Database) -> None:
        row = await db.get_current_posture(tenant_id="default")
        assert row is None

    async def test_get_posture_history(self, db: Database) -> None:
        # Save 3 postures with slightly different data
        await db.save_posture(
            tenant_id="default",
            session_date=date(2026, 2, 13),
            session_label="Morning",
            posture="balanced",
            effective_posture="balanced",
            reason="Normal conditions",
        )
        await db.save_posture(
            tenant_id="default",
            session_date=date(2026, 2, 14),
            session_label="Midday",
            posture="defensive",
            effective_posture="defensive",
            reason="VIX spike",
        )
        await db.save_posture(
            tenant_id="default",
            session_date=date(2026, 2, 15),
            session_label="Morning",
            posture="aggressive",
            effective_posture="balanced",
            reason="Strong momentum but regime gate",
        )

        rows = await db.get_posture_history(tenant_id="default")
        assert len(rows) == 3
        # Most recent first (ordered by created_at desc)
        assert rows[0].posture == "aggressive"
        assert rows[1].posture == "defensive"
        assert rows[2].posture == "balanced"


# ── Playbook Snapshots CRUD ──────────────────────────────────────────────────


class TestPlaybookCRUD:
    async def test_save_and_get_playbook(self, db: Database) -> None:
        cells = [
            {
                "regime": "BULL",
                "sector": "Technology",
                "total_trades": 15,
                "wins": 10,
                "losses": 4,
                "win_rate_pct": 71.4,
                "avg_pnl_pct": 2.5,
                "recommendation": "sweet_spot",
            },
            {
                "regime": "BEAR",
                "sector": "Healthcare",
                "total_trades": 8,
                "wins": 3,
                "losses": 5,
                "win_rate_pct": 37.5,
                "avg_pnl_pct": -1.2,
                "recommendation": "avoid",
            },
        ]
        await db.save_playbook_snapshot(cells, tenant_id="default")

        rows = await db.get_latest_playbook(tenant_id="default")
        assert len(rows) == 2

        regimes = {r.regime for r in rows}
        assert regimes == {"BULL", "BEAR"}

        bull = next(r for r in rows if r.regime == "BULL")
        assert bull.sector == "Technology"
        assert bull.total_trades == 15
        assert bull.wins == 10
        assert bull.losses == 4
        assert bull.win_rate_pct == pytest.approx(71.4)
        assert bull.avg_pnl_pct == pytest.approx(2.5)
        assert bull.recommendation == "sweet_spot"
        assert bull.tenant_id == "default"
        assert bull.generated_at is not None

    async def test_get_latest_playbook_empty(self, db: Database) -> None:
        rows = await db.get_latest_playbook(tenant_id="default")
        assert rows == []

    async def test_playbook_latest_only(self, db: Database) -> None:
        """Saving two snapshots at different times, get_latest returns only the newest."""
        first_cells = [
            {
                "regime": "BULL",
                "sector": "Technology",
                "total_trades": 10,
                "wins": 7,
                "losses": 3,
                "win_rate_pct": 70.0,
                "avg_pnl_pct": 1.5,
                "recommendation": "sweet_spot",
            },
        ]
        await db.save_playbook_snapshot(first_cells, tenant_id="default")

        # Small delay to ensure different generated_at timestamps
        await asyncio.sleep(0.05)

        second_cells = [
            {
                "regime": "SIDEWAYS",
                "sector": "Energy",
                "total_trades": 5,
                "wins": 2,
                "losses": 3,
                "win_rate_pct": 40.0,
                "avg_pnl_pct": -0.8,
                "recommendation": "reduce_size",
            },
            {
                "regime": "BEAR",
                "sector": "Financials",
                "total_trades": 12,
                "wins": 4,
                "losses": 7,
                "win_rate_pct": 36.4,
                "avg_pnl_pct": -2.1,
                "recommendation": "avoid",
            },
        ]
        await db.save_playbook_snapshot(second_cells, tenant_id="default")

        rows = await db.get_latest_playbook(tenant_id="default")
        # Should return only the second snapshot (2 cells), not the first (1 cell)
        assert len(rows) == 2
        regimes = {r.regime for r in rows}
        assert regimes == {"SIDEWAYS", "BEAR"}


# ── Conviction Calibration CRUD ──────────────────────────────────────────────


class TestCalibrationCRUD:
    async def test_save_and_get_calibration(self, db: Database) -> None:
        buckets = [
            {
                "conviction_level": "high",
                "total_trades": 20,
                "wins": 14,
                "losses": 4,
                "win_rate_pct": 77.8,
                "avg_pnl_pct": 4.21,
                "assessment": "validated",
                "suggested_multiplier": 1.2,
            },
            {
                "conviction_level": "low",
                "total_trades": 10,
                "wins": 4,
                "losses": 5,
                "win_rate_pct": 44.4,
                "avg_pnl_pct": -0.5,
                "assessment": "over_confident",
                "suggested_multiplier": 0.7,
            },
        ]
        await db.save_conviction_calibration(buckets, tenant_id="default")

        rows = await db.get_latest_calibration(tenant_id="default")
        assert len(rows) == 2

        levels = {r.conviction_level for r in rows}
        assert levels == {"high", "low"}

        high = next(r for r in rows if r.conviction_level == "high")
        assert high.total_trades == 20
        assert high.wins == 14
        assert high.losses == 4
        assert high.win_rate_pct == pytest.approx(77.8)
        assert high.avg_pnl_pct == pytest.approx(4.21)
        assert high.assessment == "validated"
        assert high.suggested_multiplier == pytest.approx(1.2)
        assert high.tenant_id == "default"
        assert high.generated_at is not None

    async def test_get_latest_calibration_empty(self, db: Database) -> None:
        rows = await db.get_latest_calibration(tenant_id="default")
        assert rows == []
