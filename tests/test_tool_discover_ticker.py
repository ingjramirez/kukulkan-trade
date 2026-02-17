"""Tests for the discover_ticker agent tool.

Validates proposal flow, duplicate detection, validation failures,
source tracking, and ActionState accumulation.
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.agent.ticker_discovery import TickerDiscovery
from src.agent.tools.actions import ActionState, _discover_ticker
from src.storage.database import Database
from src.storage.models import DiscoveredTickerRow


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


@pytest.fixture
def discovery(db: Database) -> TickerDiscovery:
    return TickerDiscovery(db)


@pytest.fixture
def state() -> ActionState:
    return ActionState()


def _make_yf_info(
    market_cap: float = 45e9,
    avg_volume: float = 2_100_000,
    sector: str = "Technology",
    name: str = "Test Corp",
) -> dict:
    return {
        "regularMarketPrice": 285.0,
        "marketCap": market_cap,
        "averageVolume": avg_volume,
        "sector": sector,
        "shortName": name,
    }


class TestDiscoverTickerSuccess:
    @patch("src.agent.ticker_discovery.yf.Ticker")
    async def test_discover_valid_ticker(
        self, mock_yf_cls, state: ActionState, discovery: TickerDiscovery, db: Database
    ) -> None:
        mock_ticker = MagicMock()
        mock_ticker.info = _make_yf_info()
        mock_yf_cls.return_value = mock_ticker

        result = await _discover_ticker(state, discovery, "default", "ANET", "Semiconductor strength", "high")

        assert result["success"] is True
        assert result["ticker"] == "ANET"
        assert result["status"] == "pending_approval"
        assert result["validation"]["market_cap_ok"] is True
        assert result["validation"]["sector"] == "Technology"

        # Verify saved to DB
        row = await db.get_discovered_ticker("ANET")
        assert row is not None
        assert row.status == "proposed"
        assert row.source == "agent_tool"

    @patch("src.agent.ticker_discovery.yf.Ticker")
    async def test_tracks_in_action_state(self, mock_yf_cls, state: ActionState, discovery: TickerDiscovery) -> None:
        mock_ticker = MagicMock()
        mock_ticker.info = _make_yf_info()
        mock_yf_cls.return_value = mock_ticker

        await _discover_ticker(state, discovery, "default", "ANET", "AI infrastructure", "high")

        assert len(state.discovery_proposals) == 1
        assert state.discovery_proposals[0]["ticker"] == "ANET"
        assert state.discovery_proposals[0]["source"] == "agent_tool"

    @patch("src.agent.ticker_discovery.yf.Ticker")
    async def test_includes_conviction_in_rationale(
        self, mock_yf_cls, state: ActionState, discovery: TickerDiscovery, db: Database
    ) -> None:
        mock_ticker = MagicMock()
        mock_ticker.info = _make_yf_info()
        mock_yf_cls.return_value = mock_ticker

        await _discover_ticker(state, discovery, "default", "ANET", "Strong momentum", "high", "Fits tech thesis")

        row = await db.get_discovered_ticker("ANET")
        assert "[high]" in row.rationale
        assert "Sector:" in row.rationale

    @patch("src.agent.ticker_discovery.yf.Ticker")
    async def test_source_is_agent_tool(
        self, mock_yf_cls, state: ActionState, discovery: TickerDiscovery, db: Database
    ) -> None:
        mock_ticker = MagicMock()
        mock_ticker.info = _make_yf_info()
        mock_yf_cls.return_value = mock_ticker

        await _discover_ticker(state, discovery, "default", "ANET", "test reason")

        row = await db.get_discovered_ticker("ANET")
        assert row.source == "agent_tool"


class TestDiscoverTickerRejections:
    @patch("src.agent.ticker_discovery.yf.Ticker")
    async def test_rejects_low_market_cap(self, mock_yf_cls, state: ActionState, discovery: TickerDiscovery) -> None:
        mock_ticker = MagicMock()
        mock_ticker.info = _make_yf_info(market_cap=500e6)
        mock_yf_cls.return_value = mock_ticker

        result = await _discover_ticker(state, discovery, "default", "PENNY", "cheap stock")

        assert result["success"] is False
        assert result["status"] == "rejected"
        assert "Market cap" in result["message"]
        assert len(state.discovery_proposals) == 0

    @patch("src.agent.ticker_discovery.yf.Ticker")
    async def test_rejects_low_volume(self, mock_yf_cls, state: ActionState, discovery: TickerDiscovery) -> None:
        mock_ticker = MagicMock()
        mock_ticker.info = _make_yf_info(avg_volume=50_000)
        mock_yf_cls.return_value = mock_ticker

        result = await _discover_ticker(state, discovery, "default", "LOWVOL", "illiquid")

        assert result["success"] is False
        assert result["status"] == "rejected"
        assert len(state.discovery_proposals) == 0

    async def test_rejects_ticker_in_universe(self, state: ActionState, discovery: TickerDiscovery) -> None:
        result = await _discover_ticker(state, discovery, "default", "AAPL", "already there")

        assert result["success"] is False
        assert result["status"] == "already_in_universe"

    @patch("src.agent.ticker_discovery.yf.Ticker")
    async def test_rejects_already_pending(
        self, mock_yf_cls, state: ActionState, discovery: TickerDiscovery, db: Database
    ) -> None:
        await db.save_discovered_ticker(
            DiscoveredTickerRow(
                tenant_id="default",
                ticker="ANET",
                source="agent_tool",
                rationale="prior proposal",
                status="proposed",
                proposed_at=date(2026, 2, 15),
                expires_at=date(2026, 3, 15),
            )
        )

        result = await _discover_ticker(state, discovery, "default", "ANET", "try again")

        assert result["success"] is False
        assert result["status"] == "already_pending"

    @patch("src.agent.ticker_discovery.yf.Ticker")
    async def test_rejects_already_approved(
        self, mock_yf_cls, state: ActionState, discovery: TickerDiscovery, db: Database
    ) -> None:
        await db.save_discovered_ticker(
            DiscoveredTickerRow(
                tenant_id="default",
                ticker="ANET",
                source="agent",
                rationale="prior",
                status="approved",
                proposed_at=date(2026, 2, 10),
                expires_at=date(2026, 3, 10),
            )
        )

        result = await _discover_ticker(state, discovery, "default", "ANET", "already approved")

        assert result["success"] is False
        assert result["status"] == "already_approved"

    async def test_rejects_empty_reason(self, state: ActionState, discovery: TickerDiscovery) -> None:
        result = await _discover_ticker(state, discovery, "default", "ANET", "")

        assert result["success"] is False
        assert "Reason is required" in result["message"]

    async def test_rejects_empty_ticker(self, state: ActionState, discovery: TickerDiscovery) -> None:
        result = await _discover_ticker(state, discovery, "default", "", "some reason")

        assert result["success"] is False

    async def test_returns_error_without_discovery(self, state: ActionState) -> None:
        result = await _discover_ticker(state, None, "default", "ANET", "test")

        assert result["success"] is False
        assert "not available" in result["message"]


class TestDiscoverTickerActionState:
    @patch("src.agent.ticker_discovery.yf.Ticker")
    async def test_accumulated_state_includes_discoveries(
        self, mock_yf_cls, state: ActionState, discovery: TickerDiscovery
    ) -> None:
        mock_ticker = MagicMock()
        mock_ticker.info = _make_yf_info()
        mock_yf_cls.return_value = mock_ticker

        await _discover_ticker(state, discovery, "default", "ANET", "test reason")

        accumulated = state.get_accumulated_state()
        assert len(accumulated["discovery_proposals"]) == 1

    def test_reset_clears_discoveries(self) -> None:
        state = ActionState()
        state.discovery_proposals.append({"ticker": "TEST"})
        state.reset()
        assert len(state.discovery_proposals) == 0
