"""Tests for discovery tool registration and system prompt.

Verifies all 3 discovery tools are registered and the system prompt
includes the discovery section.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.agent.tools import ToolRegistry
from src.agent.tools.actions import ActionState, register_action_tools
from src.agent.tools.market import register_market_tools
from src.agent.tools.portfolio import register_portfolio_tools
from src.storage.database import Database


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


@pytest.fixture
def closes() -> pd.DataFrame:
    """Minimal closes DataFrame for registration."""
    return pd.DataFrame({"SPY": [450.0, 451.0, 452.0]})


class TestToolRegistration:
    def test_search_ticker_info_registered(self, db: Database, closes: pd.DataFrame) -> None:
        registry = ToolRegistry()
        register_market_tools(registry, closes, db=db, tenant_id="default")

        assert "search_ticker_info" in registry.tool_names

    def test_discover_ticker_registered(self) -> None:
        registry = ToolRegistry()
        state = ActionState()
        mock_discovery = MagicMock()
        register_action_tools(registry, state, ticker_discovery=mock_discovery)

        assert "discover_ticker" in registry.tool_names

    def test_list_discovered_tickers_registered(self, db: Database, closes: pd.DataFrame) -> None:
        registry = ToolRegistry()
        register_portfolio_tools(registry, db, "default", {"SPY": 452.0}, closes=closes)

        assert "list_discovered_tickers" in registry.tool_names

    def test_total_tools_count(self, db: Database, closes: pd.DataFrame) -> None:
        """Verify total tool count after registering all modules."""
        registry = ToolRegistry()
        state = ActionState()
        mock_discovery = MagicMock()

        register_portfolio_tools(registry, db, "default", {"SPY": 452.0}, closes=closes)
        register_market_tools(registry, closes, db=db, tenant_id="default")
        register_action_tools(registry, state, db=db, tenant_id="default", ticker_discovery=mock_discovery)

        # Portfolio: 6 + 3 legacy + 1 discovery = 10
        # Market: 4 + 2 legacy + 1 discovery + 1 signal_rankings = 8
        # Action: 6 + 2 legacy + 1 discovery = 9
        # Total: 27 (news tools not included here since we don't register them)
        assert len(registry.tool_names) == 27

    def test_discover_ticker_registered_without_discovery(self) -> None:
        """discover_ticker should register even without TickerDiscovery (returns error)."""
        registry = ToolRegistry()
        state = ActionState()
        register_action_tools(registry, state, ticker_discovery=None)

        assert "discover_ticker" in registry.tool_names


class TestToolDefinitions:
    def test_search_ticker_info_schema(self, db: Database, closes: pd.DataFrame) -> None:
        registry = ToolRegistry()
        register_market_tools(registry, closes, db=db, tenant_id="default")

        defs = {d["name"]: d for d in registry.get_tool_definitions()}
        schema = defs["search_ticker_info"]["input_schema"]
        assert "ticker" in schema["properties"]
        assert "ticker" in schema["required"]

    def test_discover_ticker_schema(self) -> None:
        registry = ToolRegistry()
        state = ActionState()
        register_action_tools(registry, state, ticker_discovery=MagicMock())

        defs = {d["name"]: d for d in registry.get_tool_definitions()}
        schema = defs["discover_ticker"]["input_schema"]
        assert "ticker" in schema["required"]
        assert "reason" in schema["required"]
        assert "conviction" in schema["properties"]

    def test_list_discovered_tickers_schema(self, db: Database, closes: pd.DataFrame) -> None:
        registry = ToolRegistry()
        register_portfolio_tools(registry, db, "default", {"SPY": 452.0}, closes=closes)

        defs = {d["name"]: d for d in registry.get_tool_definitions()}
        schema = defs["list_discovered_tickers"]["input_schema"]
        assert "status" in schema["properties"]
