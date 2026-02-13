"""Tests for the missed day recovery feature in the orchestrator."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.orchestrator import Orchestrator
from src.storage.database import Database


def _make_closes(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Generate synthetic closes aligned to business days."""
    dates = pd.bdate_range(start=start, end=end)
    rng = np.random.default_rng(42)
    data = {}
    for t in tickers:
        base = rng.uniform(100, 300)
        returns = rng.normal(0.0005, 0.01, len(dates))
        data[t] = base * np.cumprod(1 + returns)
    return pd.DataFrame(data, index=dates)


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


@pytest.fixture
async def orchestrator(db):
    orch = Orchestrator(db)
    return orch


class TestRecoveryCheck:
    async def test_no_recovery_needed(
        self,
        orchestrator: Orchestrator,
        db: Database,
    ) -> None:
        """No missed days → empty recovery list."""
        await db.upsert_portfolio("A", cash=33_000.0, total_value=33_000.0)
        await db.upsert_portfolio("B", cash=66_000.0, total_value=66_000.0)

        today = date(2026, 2, 6)
        # Snapshot for yesterday (Feb 5)
        await db.save_snapshot(
            "A",
            date(2026, 2, 5),
            33_000.0,
            33_000.0,
            0.0,
            None,
            0.0,
        )
        await db.save_snapshot(
            "B",
            date(2026, 2, 5),
            66_000.0,
            66_000.0,
            0.0,
            None,
            0.0,
        )

        closes = _make_closes(["XLK"], "2026-01-01", "2026-02-06")
        recovered = await orchestrator.recovery_check(today, closes)

        assert recovered == []

    async def test_recovers_missed_days(
        self,
        orchestrator: Orchestrator,
        db: Database,
    ) -> None:
        """One missed day → one snapshot backfilled."""
        await db.upsert_portfolio("A", cash=33_000.0, total_value=33_000.0)
        await db.upsert_portfolio("B", cash=66_000.0, total_value=66_000.0)

        # Snapshot for Monday Feb 2, skip Feb 3-5, today is Feb 6
        await db.save_snapshot(
            "A",
            date(2026, 2, 2),
            33_000.0,
            33_000.0,
            0.0,
            None,
            0.0,
        )
        await db.save_snapshot(
            "B",
            date(2026, 2, 2),
            66_000.0,
            66_000.0,
            0.0,
            None,
            0.0,
        )

        today = date(2026, 2, 6)
        closes = _make_closes(["XLK"], "2026-01-01", "2026-02-06")

        recovered = await orchestrator.recovery_check(today, closes)

        # Should have recovered Feb 3, 4, 5 for both portfolios
        assert len(recovered) > 0
        # All recovered dates should be between Feb 2 and Feb 6
        for d in recovered:
            assert "2026-02-0" in d

    async def test_no_snapshots_skips(
        self,
        orchestrator: Orchestrator,
        db: Database,
    ) -> None:
        """No existing snapshots → nothing to recover from."""
        await db.upsert_portfolio("A", cash=33_000.0, total_value=33_000.0)

        today = date(2026, 2, 6)
        closes = _make_closes(["XLK"], "2026-01-01", "2026-02-06")

        recovered = await orchestrator.recovery_check(today, closes)
        assert recovered == []

    async def test_backfilled_snapshots_in_db(
        self,
        orchestrator: Orchestrator,
        db: Database,
    ) -> None:
        """Backfilled snapshots are actually persisted."""
        await db.upsert_portfolio("A", cash=33_000.0, total_value=33_000.0)
        await db.upsert_portfolio("B", cash=66_000.0, total_value=66_000.0)

        # Only have snapshot for Feb 3
        await db.save_snapshot(
            "A",
            date(2026, 2, 3),
            33_000.0,
            33_000.0,
            0.0,
            None,
            0.0,
        )
        await db.save_snapshot(
            "B",
            date(2026, 2, 3),
            66_000.0,
            66_000.0,
            0.0,
            None,
            0.0,
        )

        today = date(2026, 2, 6)
        closes = _make_closes(["XLK"], "2026-01-01", "2026-02-06")

        await orchestrator.recovery_check(today, closes)

        # Check snapshots for Feb 4 and 5 exist now
        a_snaps = await db.get_snapshots("A")
        a_dates = {s.date for s in a_snaps}
        assert date(2026, 2, 4) in a_dates
        assert date(2026, 2, 5) in a_dates

    async def test_recovery_with_positions(
        self,
        orchestrator: Orchestrator,
        db: Database,
    ) -> None:
        """Backfilled snapshots reflect position values."""
        await db.upsert_portfolio("A", cash=31_000.0, total_value=33_000.0)
        await db.upsert_position("A", "XLK", 10, 200.0)

        await db.save_snapshot(
            "A",
            date(2026, 2, 3),
            33_000.0,
            31_000.0,
            2_000.0,
            None,
            0.0,
        )

        today = date(2026, 2, 6)
        closes = _make_closes(["XLK"], "2026-01-01", "2026-02-06")

        await orchestrator.recovery_check(today, closes)

        a_snaps = await db.get_snapshots("A")
        # Should have more than just the initial snapshot
        assert len(a_snaps) > 1
        # Last backfilled should have non-zero positions_value
        last_backfilled = [s for s in a_snaps if s.date > date(2026, 2, 3)]
        if last_backfilled:
            assert last_backfilled[0].positions_value > 0
