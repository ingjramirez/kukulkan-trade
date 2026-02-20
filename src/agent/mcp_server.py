"""MCP tool server for Kukulkan trading agent.

Exposes all 24 agent tools over the Model Context Protocol (stdio transport).
Claude Code spawns this as a subprocess and calls tools during its investigation loop.

The orchestrator writes session state to a JSON file before invoking Claude Code.
This server reads that state on startup to initialize tools with pre-fetched data.

Usage:
    python -m src.agent.mcp_server

Environment variables:
    KUKULKAN_SESSION_STATE: Path to session-state.json (written by orchestrator)
    DATABASE_URL: SQLite connection string (default: sqlite+aiosqlite:///data/kukulkan.db)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import structlog

# Ensure project root is on sys.path for imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from mcp.server import Server  # noqa: E402
from mcp.server.stdio import stdio_server  # noqa: E402
from mcp.types import TextContent, Tool  # noqa: E402

from src.agent.tools import ToolRegistry  # noqa: E402

log = structlog.get_logger()

# ── Global state ──────────────────────────────────────────────────────────────

_registry: ToolRegistry | None = None
_action_state = None  # ActionState instance, set during init


def _load_session_state(state_path: str) -> dict:
    """Load session state written by the orchestrator."""
    with open(state_path) as f:
        return json.load(f)


async def _init_registry(state: dict) -> ToolRegistry:
    """Initialize the tool registry with pre-fetched session state.

    Mirrors the registration logic in orchestrator._run_portfolio_b_persistent().
    """
    import pandas as pd

    from src.agent.ticker_discovery import TickerDiscovery
    from src.agent.tools.actions import ActionState, register_action_tools
    from src.agent.tools.market import register_market_tools
    from src.agent.tools.news import register_news_tools
    from src.agent.tools.portfolio import register_portfolio_tools
    from src.storage.database import Database

    # Connect to database
    db_url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///data/kukulkan.db")
    db = Database(db_url)
    await db.init_db()

    tenant_id = state.get("tenant_id", "default")

    # Reconstruct DataFrames from serialized state
    closes = pd.DataFrame(state.get("closes", {}))
    if "closes_index" in state:
        closes.index = pd.to_datetime(state["closes_index"])

    # Current prices
    current_prices: dict[str, float] = state.get("current_prices", {})

    # Build registry
    registry = ToolRegistry()

    # Portfolio tools
    register_portfolio_tools(registry, db, tenant_id, current_prices, closes=closes)

    # Market tools
    held_tickers: list[str] = state.get("held_tickers", [])
    fear_greed_data: dict | None = state.get("fear_greed")
    register_market_tools(
        registry,
        closes,
        vix=state.get("vix"),
        yield_curve=state.get("yield_curve"),
        regime=state.get("regime"),
        db=db,
        held_tickers=held_tickers,
        tenant_id=tenant_id,
        fear_greed=fear_greed_data,
    )

    # News tools
    news_context: str = state.get("news_context", "")
    news_fetcher = None
    try:
        from src.data.news_aggregator import NewsAggregator

        news_fetcher = NewsAggregator(db=db)
    except Exception:
        pass

    register_news_tools(
        registry,
        news_context,
        news_fetcher=news_fetcher,
        db=db,
        tenant_id=tenant_id,
        current_prices=current_prices,
    )

    # Action tools — store globally so we can persist results on exit
    global _action_state
    _action_state = ActionState()
    ticker_discovery = TickerDiscovery(db)
    register_action_tools(
        registry,
        _action_state,
        db=db,
        tenant_id=tenant_id,
        ticker_discovery=ticker_discovery,
    )

    log.info("mcp_tools_registered", count=len(registry.tool_names), tools=registry.tool_names)
    return registry


# ── MCP Server ────────────────────────────────────────────────────────────────

app = Server("kukulkan")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """Return all registered tools as MCP Tool objects."""
    if _registry is None:
        return []
    return [
        Tool(
            name=td["name"],
            description=td["description"],
            inputSchema=td["input_schema"],
        )
        for td in _registry.get_tool_definitions()
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch a tool call to the existing handler."""
    if _registry is None:
        return [TextContent(type="text", text="Error: MCP server not initialized")]

    try:
        result = await _registry.execute(name, arguments)
        if isinstance(result, str):
            text = result
        else:
            text = json.dumps(result, default=str)
        # Truncate large results to keep context manageable
        max_chars = int(os.environ.get("TOOL_RESULT_MAX_CHARS", "3000"))
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... (truncated, {len(text)} total chars)"
        return [TextContent(type="text", text=text)]
    except KeyError:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        log.error("mcp_tool_error", tool=name, error=str(e))
        return [TextContent(type="text", text=f"Error executing {name}: {e}")]


def _write_session_results(results_path: Path) -> None:
    """Persist accumulated ActionState to disk for the invoker to read."""
    if _action_state is None:
        return
    try:
        accumulated = _action_state.get_accumulated_state()
        tmp = results_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(accumulated, default=str))
        tmp.rename(results_path)
        log.info("session_results_written", path=str(results_path))
    except Exception as e:
        log.warning("session_results_write_failed", error=str(e))


async def main() -> None:
    """Initialize registry and run MCP server over stdio."""
    global _registry

    state_path = os.environ.get("KUKULKAN_SESSION_STATE", "data/agent-workspace/session-state.json")
    if not Path(state_path).exists():
        log.error("session_state_not_found", path=state_path)
        sys.exit(1)

    state = _load_session_state(state_path)
    _registry = await _init_registry(state)

    results_path = Path(state_path).parent / "session-results.json"

    try:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())
    finally:
        _write_session_results(results_path)


if __name__ == "__main__":
    asyncio.run(main())
