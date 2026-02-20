"""Integration tests for the invoker → MCP → tool data flow.

Tests the data handoff between components without requiring Claude Code CLI.
Exercises: session-state → ToolRegistry init → tool call → session-results round-trip.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Thread

import pytest


class TestSessionStateToRegistry:
    """Verify session-state.json → _init_registry → tool call works end-to-end."""

    @pytest.mark.asyncio
    async def test_init_registry_from_state(self, tmp_path: Path):
        """Write session-state.json, init registry, call a tool."""
        from src.agent.claude_invoker import write_session_state
        from src.agent.mcp_server import _init_registry

        write_session_state(
            workspace=tmp_path,
            tenant_id="default",
            closes_dict={"SPY": {"2024-01-01": 470.0, "2024-01-02": 475.0}},
            closes_index=["2024-01-01", "2024-01-02"],
            current_prices={"SPY": 475.0},
            held_tickers=["SPY"],
            vix=15.0,
            yield_curve=0.5,
            regime="bull",
        )

        state = json.loads((tmp_path / "session-state.json").read_text())
        registry = await _init_registry(state)

        # Registry should have tools registered
        assert len(registry.tool_names) > 0
        assert "get_market_overview" in registry.tool_names

        # Call a tool that uses the session state
        result = await registry.execute("get_market_overview", {})
        if isinstance(result, str):
            data = json.loads(result)
        else:
            data = result
        assert data["vix"] == 15.0
        assert data["regime"] == "bull"


class TestSessionResultsRoundTrip:
    """Verify MCP _write_session_results → invoker _read_session_results."""

    def test_write_then_read(self, tmp_path: Path):
        """Session results written by MCP should be readable by invoker."""
        import src.agent.mcp_server as mcp_mod
        from src.agent.claude_invoker import ClaudeInvoker
        from src.agent.mcp_server import _write_session_results
        from src.agent.tools.actions import ActionState

        # Simulate MCP writing results
        mcp_mod._action_state = ActionState()
        mcp_mod._action_state.declared_posture = "aggressive"
        mcp_mod._action_state.trailing_stop_requests.append({"ticker": "NVDA", "trail_pct": 0.07})

        results_path = tmp_path / "default" / "session-results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        _write_session_results(results_path)
        mcp_mod._action_state = None

        # Simulate invoker reading results
        invoker = ClaudeInvoker(workspace=tmp_path)
        data = invoker._read_session_results(results_path)

        assert data["declared_posture"] == "aggressive"
        assert len(data["trailing_stop_requests"]) == 1
        assert data["trailing_stop_requests"][0]["ticker"] == "NVDA"
        # File should be cleaned up
        assert not results_path.exists()

    def test_delayed_write_read_with_retry(self, tmp_path: Path):
        """Invoker should retry and find file written after a short delay (simulates MCP flush race)."""
        from src.agent.claude_invoker import ClaudeInvoker

        invoker = ClaudeInvoker(workspace=tmp_path)
        results_path = tmp_path / "default" / "session-results.json"

        # Write file after a short delay in another thread
        def delayed_write():
            time.sleep(0.15)
            results_path.write_text(json.dumps({"posture": "defensive", "trailing_stop_requests": []}))

        t = Thread(target=delayed_write)
        t.start()

        data = invoker._read_session_results(results_path, retries=6, delay=0.05)
        t.join()

        assert data["posture"] == "defensive"


class TestMCPConfigGeneration:
    """Verify mcp.json generation resolves paths correctly."""

    def test_paths_are_absolute(self, tmp_path: Path):
        from src.agent.claude_invoker import ClaudeInvoker

        invoker = ClaudeInvoker(workspace=tmp_path, tenant_id="prod")
        path = invoker._write_mcp_config()

        data = json.loads(path.read_text())
        server = data["mcpServers"]["kukulkan"]

        # All paths should be absolute
        assert Path(server["args"][0]).is_absolute()
        assert Path(server["env"]["KUKULKAN_SESSION_STATE"]).is_absolute()
        db_url = server["env"]["DATABASE_URL"]
        # Extract path from sqlite URL
        db_path = db_url.split("///")[-1]
        assert Path(db_path).is_absolute()

    def test_tenant_isolation(self, tmp_path: Path):
        """Different tenants should get different session state paths."""
        from src.agent.claude_invoker import ClaudeInvoker

        inv_a = ClaudeInvoker(workspace=tmp_path, tenant_id="alice")
        inv_b = ClaudeInvoker(workspace=tmp_path, tenant_id="bob")

        path_a = inv_a._write_mcp_config()
        path_b = inv_b._write_mcp_config()

        data_a = json.loads(path_a.read_text())
        data_b = json.loads(path_b.read_text())

        state_a = data_a["mcpServers"]["kukulkan"]["env"]["KUKULKAN_SESSION_STATE"]
        state_b = data_b["mcpServers"]["kukulkan"]["env"]["KUKULKAN_SESSION_STATE"]

        assert "alice" in state_a
        assert "bob" in state_b
        assert state_a != state_b

    def test_venv_python_preferred(self, tmp_path: Path):
        """If .venv/bin/python exists, it should be used as the command."""
        from src.agent.claude_invoker import ClaudeInvoker, _project_root

        invoker = ClaudeInvoker(workspace=tmp_path, tenant_id="test")
        path = invoker._write_mcp_config()

        data = json.loads(path.read_text())
        cmd = data["mcpServers"]["kukulkan"]["command"]

        venv_python = _project_root / ".venv" / "bin" / "python"
        if venv_python.exists():
            assert cmd == str(venv_python)
        else:
            assert cmd == "python"
