"""Tests for the MCP tool server adapter."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestMCPServerModule:
    """Test mcp_server.py module functions without requiring the MCP SDK."""

    def test_load_session_state(self, tmp_path: Path):
        from src.agent.mcp_server import _load_session_state

        state = {"tenant_id": "default", "vix": 15.0, "closes": {}}
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(state))

        loaded = _load_session_state(str(state_file))
        assert loaded["tenant_id"] == "default"
        assert loaded["vix"] == 15.0

    def test_write_session_results(self, tmp_path: Path):
        import src.agent.mcp_server as mcp_mod
        from src.agent.mcp_server import _write_session_results

        # Set up module-level _action_state
        from src.agent.tools.actions import ActionState

        mcp_mod._action_state = ActionState()
        mcp_mod._action_state.declared_posture = "aggressive"
        mcp_mod._action_state.trailing_stop_requests.append({"ticker": "NVDA", "trail_pct": 0.07})

        results_path = tmp_path / "session-results.json"
        _write_session_results(results_path)

        assert results_path.exists()
        data = json.loads(results_path.read_text())
        assert data["declared_posture"] == "aggressive"
        assert len(data["trailing_stop_requests"]) == 1

        # Clean up module state
        mcp_mod._action_state = None

    def test_write_session_results_no_state(self, tmp_path: Path):
        import src.agent.mcp_server as mcp_mod
        from src.agent.mcp_server import _write_session_results

        mcp_mod._action_state = None
        results_path = tmp_path / "session-results.json"
        _write_session_results(results_path)
        assert not results_path.exists()


class TestMCPServerToolDispatch:
    """Test the call_tool handler with a mocked registry."""

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        import src.agent.mcp_server as mcp_mod

        mock_registry = MagicMock()
        mock_registry.execute = AsyncMock(return_value={"cash": 50000, "equity": 100000})
        mcp_mod._registry = mock_registry

        try:
            result = await mcp_mod.call_tool("get_portfolio_state", {})
            assert len(result) == 1
            text = result[0].text
            data = json.loads(text)
            assert data["cash"] == 50000
        finally:
            mcp_mod._registry = None

    @pytest.mark.asyncio
    async def test_call_tool_unknown(self):
        import src.agent.mcp_server as mcp_mod

        mock_registry = MagicMock()
        mock_registry.execute = AsyncMock(side_effect=KeyError("no_such_tool"))
        mcp_mod._registry = mock_registry

        try:
            result = await mcp_mod.call_tool("no_such_tool", {})
            assert "Unknown tool" in result[0].text
        finally:
            mcp_mod._registry = None

    @pytest.mark.asyncio
    async def test_call_tool_exception(self):
        import src.agent.mcp_server as mcp_mod

        mock_registry = MagicMock()
        mock_registry.execute = AsyncMock(side_effect=ValueError("bad input"))
        mcp_mod._registry = mock_registry

        try:
            result = await mcp_mod.call_tool("get_portfolio_state", {})
            assert "Error executing" in result[0].text
            assert "bad input" in result[0].text
        finally:
            mcp_mod._registry = None

    @pytest.mark.asyncio
    async def test_call_tool_no_registry(self):
        import src.agent.mcp_server as mcp_mod

        mcp_mod._registry = None
        result = await mcp_mod.call_tool("anything", {})
        assert "not initialized" in result[0].text

    @pytest.mark.asyncio
    async def test_call_tool_truncates_large_result(self):
        import src.agent.mcp_server as mcp_mod

        mock_registry = MagicMock()
        mock_registry.execute = AsyncMock(return_value="x" * 5000)
        mcp_mod._registry = mock_registry

        try:
            with patch.dict("os.environ", {"TOOL_RESULT_MAX_CHARS": "100"}):
                result = await mcp_mod.call_tool("big_tool", {})
            text = result[0].text
            assert "truncated" in text
            assert len(text) < 5000
        finally:
            mcp_mod._registry = None

    @pytest.mark.asyncio
    async def test_list_tools_empty_when_no_registry(self):
        import src.agent.mcp_server as mcp_mod

        mcp_mod._registry = None
        result = await mcp_mod.list_tools()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_tools_returns_tool_objects(self):
        import src.agent.mcp_server as mcp_mod

        mock_registry = MagicMock()
        mock_registry.get_tool_definitions.return_value = [
            {
                "name": "get_portfolio_state",
                "description": "Get portfolio",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]
        mcp_mod._registry = mock_registry

        try:
            tools = await mcp_mod.list_tools()
            assert len(tools) == 1
            assert tools[0].name == "get_portfolio_state"
        finally:
            mcp_mod._registry = None


class TestWorkspaceFiles:
    """Verify workspace configuration files are valid."""

    def test_claude_md_exists(self):
        path = Path(__file__).parent.parent / "data" / "agent-workspace" / "CLAUDE.md"
        assert path.exists()
        content = path.read_text()
        assert "Kukulkan" in content
        assert "Hard Rules" in content
        assert "Output Format" in content

    def test_claude_md_has_tool_guidance(self):
        path = Path(__file__).parent.parent / "data" / "agent-workspace" / "CLAUDE.md"
        content = path.read_text()
        assert "get_signal_rankings" in content
        assert "get_portfolio_state" in content
        assert "execute_trade" in content

    def test_settings_json_whitelists_mcp(self):
        path = Path(__file__).parent.parent / "data" / "agent-workspace" / ".claude" / "settings.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert "mcp__kukulkan__*" in data["permissions"]["allow"]

    def test_settings_json_denies_bash(self):
        path = Path(__file__).parent.parent / "data" / "agent-workspace" / ".claude" / "settings.json"
        data = json.loads(path.read_text())
        assert "Bash(*)" in data["permissions"]["deny"]

    def test_mcp_json_valid(self):
        path = Path(__file__).parent.parent / "data" / "agent-workspace" / "mcp.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert "kukulkan" in data["mcpServers"]
        server = data["mcpServers"]["kukulkan"]
        assert server["type"] == "stdio"
        assert "python" in server["command"]

    def test_gitignore_covers_ephemeral(self):
        path = Path(__file__).parent.parent / "data" / "agent-workspace" / ".gitignore"
        content = path.read_text()
        assert "session-state.json" in content
        assert "context.md" in content
