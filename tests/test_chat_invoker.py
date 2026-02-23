"""Tests for ClaudeInvoker chat methods (chat, chat_stream, helpers)."""

from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.claude_invoker import CHAT_SYSTEM_PROMPT, ChatResult, ClaudeInvoker


@pytest.fixture
def invoker(tmp_path: Path) -> ClaudeInvoker:
    """ClaudeInvoker using a temporary workspace."""
    inv = ClaudeInvoker(workspace=tmp_path, timeout=10, max_turns=5, tenant_id="default")
    return inv


# ── CHAT_SYSTEM_PROMPT ───────────────────────────────────────────────────────


class TestEnsureSessionState:
    def test_writes_minimal_state_when_missing(self, invoker: ClaudeInvoker, tmp_path: Path):
        invoker._ensure_session_state()
        state_file = invoker._workspace / "session-state.json"
        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert data["tenant_id"] == "default"
        assert data["closes"] == {}
        assert data["current_prices"] == {}

    def test_does_not_overwrite_existing_state(self, invoker: ClaudeInvoker, tmp_path: Path):
        state_file = invoker._workspace / "session-state.json"
        existing = {"tenant_id": "default", "closes": {"SPY": {"2026-01-01": 500.0}}, "current_prices": {"SPY": 510.0}}
        state_file.write_text(json.dumps(existing))

        invoker._ensure_session_state()

        data = json.loads(state_file.read_text())
        assert data["closes"] == {"SPY": {"2026-01-01": 500.0}}  # unchanged


def test_chat_system_prompt_is_defined():
    assert CHAT_SYSTEM_PROMPT
    assert "chat mode" in CHAT_SYSTEM_PROMPT.lower()


# ── ChatResult ───────────────────────────────────────────────────────────────


def test_chat_result_defaults():
    r = ChatResult()
    assert r.content == ""
    assert r.session_id is None
    assert r.tool_calls == []
    assert r.error is None


def test_chat_result_with_values():
    r = ChatResult(content="Hello!", session_id="sess_1", num_turns=3, duration_ms=1200)
    assert r.content == "Hello!"
    assert r.num_turns == 3


# ── _build_chat_cmd ──────────────────────────────────────────────────────────


def test_build_chat_cmd_no_resume(invoker: ClaudeInvoker, tmp_path: Path):
    cmd = invoker._build_chat_cmd("What's my portfolio?", session_id=None)
    assert "claude" in cmd
    assert "-p" in cmd
    assert "What's my portfolio?" in cmd
    assert "--output-format" in cmd
    assert "json" in cmd
    assert "--append-system-prompt" in cmd
    assert CHAT_SYSTEM_PROMPT in cmd
    assert "--resume" not in cmd


def test_build_chat_cmd_with_resume(invoker: ClaudeInvoker):
    cmd = invoker._build_chat_cmd("Hello", session_id="sess_abc")
    assert "--resume" in cmd
    assert "sess_abc" in cmd


def test_build_chat_stream_cmd(invoker: ClaudeInvoker):
    cmd = invoker._build_chat_stream_cmd("Tell me about NVDA", session_id=None)
    assert "--output-format" in cmd
    idx = cmd.index("--output-format")
    assert cmd[idx + 1] == "stream-json"


def test_chat_cmd_caps_max_turns(tmp_path: Path):
    """Chat sessions cap at 10 turns even if invoker has higher max."""
    inv = ClaudeInvoker(workspace=tmp_path, max_turns=25, tenant_id="default")
    cmd = inv._build_chat_cmd("Hi", None)
    idx = cmd.index("--max-turns")
    assert int(cmd[idx + 1]) <= 10


# ── _parse_stream_event ───────────────────────────────────────────────────────


class TestParseStreamEvent:
    def test_parses_text_event(self, invoker: ClaudeInvoker):
        raw = {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]}}
        parsed = invoker._parse_stream_event(raw)
        assert parsed == {"type": "text", "text": "Hello!"}

    def test_parses_tool_use_event(self, invoker: ClaudeInvoker):
        raw = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "tu_1", "name": "get_portfolio_state", "input": {}}],
            },
        }
        parsed = invoker._parse_stream_event(raw)
        assert parsed == {"type": "tool_use", "id": "tu_1", "name": "get_portfolio_state", "input": {}}

    def test_parses_tool_result(self, invoker: ClaudeInvoker):
        raw = {"type": "tool", "tool_use_id": "tu_1", "content": "portfolio data"}
        parsed = invoker._parse_stream_event(raw)
        assert parsed == {"type": "tool_result", "tool_use_id": "tu_1", "content": "portfolio data"}

    def test_parses_result_event(self, invoker: ClaudeInvoker):
        raw = {"type": "result", "result": "Done", "session_id": "sess_x", "num_turns": 3, "duration_ms": 900}
        parsed = invoker._parse_stream_event(raw)
        assert parsed is not None
        assert parsed["type"] == "done"
        assert parsed["session_id"] == "sess_x"
        assert parsed["num_turns"] == 3

    def test_unknown_type_returns_none(self, invoker: ClaudeInvoker):
        assert invoker._parse_stream_event({"type": "unknown_event"}) is None

    def test_empty_text_skipped(self, invoker: ClaudeInvoker):
        raw = {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": ""}]}}
        assert invoker._parse_stream_event(raw) is None

    def test_tool_result_truncated(self, invoker: ClaudeInvoker):
        long_content = "x" * 600
        raw = {"type": "tool", "tool_use_id": "tu_2", "content": long_content}
        parsed = invoker._parse_stream_event(raw)
        assert len(parsed["content"]) <= 500


# ── chat() ────────────────────────────────────────────────────────────────────


class TestChat:
    def _make_proc(self, stdout: str, returncode: int = 0):
        return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")

    async def test_chat_returns_content(self, invoker: ClaudeInvoker, tmp_path: Path):
        response_json = json.dumps({
            "result": "The portfolio is doing well with 5 positions.",
            "session_id": "sess_new",
            "num_turns": 3,
            "duration_ms": 800,
        })
        with (
            patch.object(invoker, "_write_mcp_config"),
            patch("src.agent.claude_invoker.asyncio.to_thread", return_value=self._make_proc(response_json)),
        ):
            result = await invoker.chat("How is the portfolio?", today=date(2026, 1, 15))

        assert result.error is None
        assert result.content == "The portfolio is doing well with 5 positions."
        assert result.session_id == "sess_new"
        assert result.num_turns == 3

    async def test_chat_saves_session_id(self, invoker: ClaudeInvoker, tmp_path: Path):
        response_json = json.dumps({"result": "Hi", "session_id": "sess_saved", "num_turns": 1, "duration_ms": 100})
        today = date(2026, 1, 15)
        with (
            patch.object(invoker, "_write_mcp_config"),
            patch("src.agent.claude_invoker.asyncio.to_thread", return_value=self._make_proc(response_json)),
        ):
            await invoker.chat("Hello", today=today)

        assert invoker._get_daily_session_id(today) == "sess_saved"

    async def test_chat_error_on_nonzero_exit(self, invoker: ClaudeInvoker):
        with (
            patch.object(invoker, "_write_mcp_config"),
            patch("src.agent.claude_invoker.asyncio.to_thread", return_value=self._make_proc("", returncode=1)),
        ):
            result = await invoker.chat("Hello")

        assert result.error is not None
        assert "Exit code 1" in result.error

    async def test_chat_timeout_returns_error(self, invoker: ClaudeInvoker):
        with (
            patch.object(invoker, "_write_mcp_config"),
            patch("src.agent.claude_invoker.asyncio.to_thread", side_effect=subprocess.TimeoutExpired("claude", 10)),
        ):
            result = await invoker.chat("Hello")

        assert result.error is not None
        assert "timed out" in result.error.lower()

    async def test_chat_resumes_existing_session(self, invoker: ClaudeInvoker):
        today = date(2026, 1, 15)
        invoker._save_daily_session_id(today, "existing_sess")

        mock_run = MagicMock(return_value=subprocess.CompletedProcess([], 0, '{"result": "ok"}', ""))
        with (
            patch.object(invoker, "_write_mcp_config"),
            patch("src.agent.claude_invoker.asyncio.to_thread", side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
            patch("src.agent.claude_invoker._run_with_kill", mock_run),
        ):
            await invoker.chat("Hello", today=today)

        call_args = mock_run.call_args
        assert call_args is not None
        cmd_used = call_args[0][0]
        assert "--resume" in cmd_used
        assert "existing_sess" in cmd_used


# ── chat_stream() ─────────────────────────────────────────────────────────────


class TestChatStream:
    async def test_streams_text_events(self, invoker: ClaudeInvoker):
        asst_event = {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
        }
        result_event = {"type": "result", "result": "Hello", "session_id": "s1", "num_turns": 1, "duration_ms": 200}
        ndjson_lines = [json.dumps(asst_event) + "\n", json.dumps(result_event) + "\n"]

        mock_proc = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        async def _aiter():
            for line in ndjson_lines:
                yield line.encode()

        mock_proc.stdout.__aiter__ = lambda self: _aiter()

        with patch("src.agent.claude_invoker.asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch.object(invoker, "_write_mcp_config"):
                events = []
                async for event in invoker.chat_stream("Hello"):
                    events.append(event)

        text_events = [e for e in events if e.get("type") == "text"]
        done_events = [e for e in events if e.get("type") == "done"]
        assert len(text_events) == 1
        assert text_events[0]["text"] == "Hello"
        assert len(done_events) == 1
        assert done_events[0]["session_id"] == "s1"

    async def test_stream_saves_session_id(self, invoker: ClaudeInvoker):
        today = date(2026, 2, 10)
        result_event = {"type": "result", "result": "ok", "session_id": "stream_sess", "num_turns": 1, "duration_ms": 0}
        ndjson_lines = [json.dumps(result_event) + "\n"]
        mock_proc = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        async def _aiter():
            for line in ndjson_lines:
                yield line.encode()

        mock_proc.stdout.__aiter__ = lambda self: _aiter()

        with patch("src.agent.claude_invoker.asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch.object(invoker, "_write_mcp_config"):
                async for _ in invoker.chat_stream("Hi", today=today):
                    pass

        assert invoker._get_daily_session_id(today) == "stream_sess"

    async def test_stream_skips_invalid_json(self, invoker: ClaudeInvoker):
        ndjson_lines = [b"not json\n", b'{"type": "result", "result": "ok"}\n']

        mock_proc = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        async def _aiter():
            for line in ndjson_lines:
                yield line

        mock_proc.stdout.__aiter__ = lambda self: _aiter()

        with patch("src.agent.claude_invoker.asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch.object(invoker, "_write_mcp_config"):
                events = []
                async for event in invoker.chat_stream("Hi"):
                    events.append(event)

        # Invalid JSON line produces no event; result line may produce a done event
        error_events = [e for e in events if e.get("type") == "error"]
        assert len(error_events) == 0

    async def test_stream_subprocess_error_yields_error_event(self, invoker: ClaudeInvoker):
        with patch(
            "src.agent.claude_invoker.asyncio.create_subprocess_exec",
            side_effect=OSError("executable not found"),
        ):
            with patch.object(invoker, "_write_mcp_config"):
                events = []
                async for event in invoker.chat_stream("Hi"):
                    events.append(event)

        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert "executable not found" in events[0]["message"]
