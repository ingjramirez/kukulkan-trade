"""Tests for the list_discovered_tickers agent tool.

Validates listing, filtering, summary counts, and source tracking.
"""

from datetime import date

import pytest

from src.agent.tools.portfolio import _list_discovered_tickers
from src.storage.database import Database
from src.storage.models import DiscoveredTickerRow


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


async def _seed_discoveries(db: Database, tenant_id: str = "default") -> None:
    """Seed the DB with a variety of discovered tickers."""
    tickers = [
        ("ANET", "approved", "agent_tool", "Semiconductor strength"),
        ("PLTR", "rejected", "agent", "Government AI contracts"),
        ("SMCI", "proposed", "agent_tool", "AI infrastructure demand"),
        ("MRVL", "expired", "agent", "Was extended"),
        ("CRWD", "approved", "agent_tool", "Cybersecurity leader"),
    ]
    for ticker, status, source, rationale in tickers:
        await db.save_discovered_ticker(
            DiscoveredTickerRow(
                tenant_id=tenant_id,
                ticker=ticker,
                source=source,
                rationale=rationale,
                status=status,
                proposed_at=date(2026, 2, 10),
                expires_at=date(2026, 3, 10),
                sector="Technology",
                market_cap=45e9,
            )
        )


class TestListAllDiscoveries:
    async def test_list_all(self, db: Database) -> None:
        await _seed_discoveries(db)

        result = await _list_discovered_tickers(db, "default")

        assert len(result["discoveries"]) == 5
        assert result["summary"]["total"] == 5

    async def test_summary_counts(self, db: Database) -> None:
        await _seed_discoveries(db)

        result = await _list_discovered_tickers(db, "default")

        summary = result["summary"]
        assert summary["approved"] == 2
        assert summary["rejected"] == 1
        assert summary["pending"] == 1
        assert summary["expired"] == 1

    async def test_approval_rate(self, db: Database) -> None:
        await _seed_discoveries(db)

        result = await _list_discovered_tickers(db, "default")

        # 2 approved / 3 resolved (2 approved + 1 rejected)
        assert result["summary"]["approval_rate"] == round(2 / 3, 2)


class TestListFilteredByStatus:
    async def test_filter_approved(self, db: Database) -> None:
        await _seed_discoveries(db)

        result = await _list_discovered_tickers(db, "default", status="approved")

        assert len(result["discoveries"]) == 2
        assert all(d["status"] == "approved" for d in result["discoveries"])

    async def test_filter_pending(self, db: Database) -> None:
        await _seed_discoveries(db)

        result = await _list_discovered_tickers(db, "default", status="proposed")

        assert len(result["discoveries"]) == 1
        assert result["discoveries"][0]["ticker"] == "SMCI"

    async def test_filter_rejected(self, db: Database) -> None:
        await _seed_discoveries(db)

        result = await _list_discovered_tickers(db, "default", status="rejected")

        assert len(result["discoveries"]) == 1
        assert result["discoveries"][0]["ticker"] == "PLTR"


class TestListDiscoveryFields:
    async def test_includes_source_field(self, db: Database) -> None:
        await _seed_discoveries(db)

        result = await _list_discovered_tickers(db, "default")

        sources = {d["source"] for d in result["discoveries"]}
        assert "agent_tool" in sources
        assert "agent" in sources

    async def test_includes_in_active_universe(self, db: Database) -> None:
        await _seed_discoveries(db)

        result = await _list_discovered_tickers(db, "default")

        approved_entries = [d for d in result["discoveries"] if d["status"] == "approved"]
        for entry in approved_entries:
            assert entry["in_active_universe"] is True

        rejected_entries = [d for d in result["discoveries"] if d["status"] == "rejected"]
        for entry in rejected_entries:
            assert entry["in_active_universe"] is False

    async def test_includes_market_cap_display(self, db: Database) -> None:
        await _seed_discoveries(db)

        result = await _list_discovered_tickers(db, "default")

        for d in result["discoveries"]:
            assert "market_cap" in d
            assert "$" in d["market_cap"]

    async def test_includes_proposed_date(self, db: Database) -> None:
        await _seed_discoveries(db)

        result = await _list_discovered_tickers(db, "default")

        for d in result["discoveries"]:
            assert d["proposed_date"] == "2026-02-10"


class TestListEdgeCases:
    async def test_empty_no_discoveries(self, db: Database) -> None:
        result = await _list_discovered_tickers(db, "default")

        assert result["discoveries"] == []
        assert result["summary"]["total"] == 0
        assert result["summary"]["approval_rate"] is None

    async def test_tenant_isolation(self, db: Database) -> None:
        await db.ensure_tenant("t-a")
        await _seed_discoveries(db, tenant_id="t-a")

        # Default tenant should see nothing
        result = await _list_discovered_tickers(db, "default")
        assert result["summary"]["total"] == 0

        # Tenant t-a should see all
        result_ta = await _list_discovered_tickers(db, "t-a")
        assert result_ta["summary"]["total"] == 5
