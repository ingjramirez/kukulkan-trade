"""Tests for Claude Code CLI invoker and context builders."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from src.agent.claude_invoker import (
    VALID_SESSION_TYPES,
    ClaudeInvoker,
    InvokeResult,
    _run_with_kill,
    write_context_file,
    write_session_state,
)

# ── write_session_state tests ────────────────────────────────────────────────


class TestWriteSessionState:
    def test_writes_json_file(self, tmp_path: Path):
        out = write_session_state(
            workspace=tmp_path,
            tenant_id="default",
            closes_dict={"SPY": {"2024-01-01": 470.0}},
            closes_index=["2024-01-01"],
            current_prices={"SPY": 470.0},
            held_tickers=["SPY"],
            vix=15.0,
            yield_curve=0.5,
            regime="bull",
        )
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["tenant_id"] == "default"
        assert data["current_prices"]["SPY"] == 470.0
        assert data["vix"] == 15.0
        assert data["regime"] == "bull"
        assert data["held_tickers"] == ["SPY"]

    def test_includes_fear_greed(self, tmp_path: Path):
        out = write_session_state(
            workspace=tmp_path,
            tenant_id="t1",
            closes_dict={},
            closes_index=[],
            current_prices={},
            held_tickers=[],
            fear_greed={"value": 72, "classification": "Greed"},
        )
        data = json.loads(out.read_text())
        assert data["fear_greed"]["value"] == 72

    def test_atomic_write(self, tmp_path: Path):
        """Verify no .tmp file remains after write."""
        write_session_state(
            workspace=tmp_path,
            tenant_id="default",
            closes_dict={},
            closes_index=[],
            current_prices={},
            held_tickers=[],
        )
        assert not (tmp_path / "session-state.tmp").exists()
        assert (tmp_path / "session-state.json").exists()


# ── write_context_file tests ─────────────────────────────────────────────────


class TestWriteContextFile:
    def test_writes_markdown(self, tmp_path: Path):
        out = write_context_file(
            workspace=tmp_path,
            session_type="morning",
            today=date(2024, 6, 15),
            regime="bull",
            vix=14.5,
            yield_curve=0.3,
            cash=50000.0,
            total_value=100000.0,
            positions=[],
        )
        content = out.read_text()
        assert "# Trading Session: Morning" in content
        assert "2024-06-15" in content
        assert "bull" in content
        assert "14.5" in content
        assert "$50,000.00" in content

    def test_includes_positions(self, tmp_path: Path):
        out = write_context_file(
            workspace=tmp_path,
            session_type="midday",
            today=date(2024, 6, 15),
            regime="bull",
            vix=14.0,
            yield_curve=0.2,
            cash=30000.0,
            total_value=80000.0,
            positions=[
                {"ticker": "NVDA", "shares": 10, "avg_price": 120.0, "market_value": 1300.0},
            ],
        )
        content = out.read_text()
        assert "NVDA" in content
        assert "10 shares" in content

    def test_includes_signal_text(self, tmp_path: Path):
        out = write_context_file(
            workspace=tmp_path,
            session_type="morning",
            today=date(2024, 6, 15),
            regime="bull",
            vix=14.0,
            yield_curve=0.2,
            cash=50000.0,
            total_value=100000.0,
            positions=[],
            signal_text="NVDA: score 85, RSI 62",
        )
        content = out.read_text()
        assert "Signal Rankings" in content
        assert "NVDA: score 85" in content

    def test_includes_sentinel_alerts(self, tmp_path: Path):
        out = write_context_file(
            workspace=tmp_path,
            session_type="morning",
            today=date(2024, 6, 15),
            regime="correction",
            vix=28.0,
            yield_curve=-0.1,
            cash=50000.0,
            total_value=95000.0,
            positions=[],
            sentinel_alerts=["NVDA stop triggered at $118"],
        )
        content = out.read_text()
        assert "Sentinel Alerts" in content
        assert "NVDA stop triggered" in content

    def test_includes_pinned_context(self, tmp_path: Path):
        out = write_context_file(
            workspace=tmp_path,
            session_type="morning",
            today=date(2024, 6, 15),
            regime="bull",
            vix=14.0,
            yield_curve=0.2,
            cash=50000.0,
            total_value=100000.0,
            positions=[],
            pinned_context="## Benchmark: Portfolio A\nReturn: +12.3%",
        )
        content = out.read_text()
        assert "Benchmark: Portfolio A" in content
        assert "+12.3%" in content

    def test_no_yield_curve(self, tmp_path: Path):
        out = write_context_file(
            workspace=tmp_path,
            session_type="morning",
            today=date(2024, 6, 15),
            regime=None,
            vix=None,
            yield_curve=None,
            cash=50000.0,
            total_value=100000.0,
            positions=[],
        )
        content = out.read_text()
        assert "Yield Curve: N/A" in content


# ── InvokeResult tests ───────────────────────────────────────────────────────


class TestInvokeResult:
    def test_trades_from_response(self):
        r = InvokeResult(response={"trades": [{"ticker": "NVDA", "side": "BUY"}]})
        assert len(r.trades) == 1
        assert r.trades[0]["ticker"] == "NVDA"

    def test_empty_trades_on_error(self):
        r = InvokeResult(error="timeout")
        assert r.trades == []

    def test_posture_from_response(self):
        r = InvokeResult(response={"posture": "aggressive"})
        assert r.posture == "aggressive"

    def test_posture_from_accumulated(self):
        r = InvokeResult(response={}, accumulated={"declared_posture": "defensive"})
        assert r.posture == "defensive"

    def test_trailing_stops_from_response(self):
        r = InvokeResult(
            response={"trailing_stops": [{"ticker": "NVDA", "trail_pct": 0.07}]},
        )
        assert len(r.trailing_stop_requests) == 1

    def test_trailing_stops_from_accumulated(self):
        r = InvokeResult(
            response={},
            accumulated={"trailing_stop_requests": [{"ticker": "NVDA", "trail_pct": 0.05}]},
        )
        assert len(r.trailing_stop_requests) == 1

    def test_tool_summary_shape(self):
        r = InvokeResult(
            response={"posture": "neutral"},
            accumulated={"trailing_stop_requests": [], "declared_posture": None, "tool_call_count": 5},
            num_turns=3,
            duration_ms=45000,
        )
        summary = r.tool_summary
        assert "trailing_stop_requests" in summary
        assert "declared_posture" in summary
        assert summary["source"] == "claude_code"
        assert summary["tools_used"] == 5
        assert summary["turns"] == 3
        assert summary["duration_ms"] == 45000

    def test_tools_used_from_accumulated(self):
        r = InvokeResult(accumulated={"tool_call_count": 8})
        assert r.tools_used == 8

    def test_tools_used_defaults_zero(self):
        r = InvokeResult()
        assert r.tools_used == 0

    def test_turns_and_duration_default_zero(self):
        r = InvokeResult()
        assert r.num_turns == 0
        assert r.duration_ms == 0


# ── ClaudeInvoker tests ─────────────────────────────────────────────────────


class TestClaudeInvoker:
    def test_tenant_workspace_created(self, tmp_path: Path):
        invoker = ClaudeInvoker(workspace=tmp_path, tenant_id="tenant-abc")
        assert invoker._workspace == tmp_path / "tenant-abc"
        assert invoker._workspace.exists()

    def test_default_tenant_workspace(self, tmp_path: Path):
        invoker = ClaudeInvoker(workspace=tmp_path)
        assert invoker._workspace == tmp_path / "default"
        assert invoker._workspace.exists()

    def test_build_cmd_new_session(self, tmp_path: Path):
        invoker = ClaudeInvoker(workspace=tmp_path)
        cmd = invoker._build_cmd("morning", session_id=None)
        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "--output-format" in cmd
        assert "--resume" not in cmd

    def test_build_cmd_resume(self, tmp_path: Path):
        invoker = ClaudeInvoker(workspace=tmp_path)
        cmd = invoker._build_cmd("midday", session_id="abc-123")
        assert "--resume" in cmd
        idx = cmd.index("--resume")
        assert cmd[idx + 1] == "abc-123"

    def test_build_cmd_mcp_config_in_tenant_dir(self, tmp_path: Path):
        invoker = ClaudeInvoker(workspace=tmp_path, tenant_id="t1")
        cmd = invoker._build_cmd("morning", session_id=None)
        idx = cmd.index("--mcp-config")
        assert "t1/mcp.json" in cmd[idx + 1]

    def test_parse_response_direct_json(self, tmp_path: Path):
        invoker = ClaudeInvoker(workspace=tmp_path)
        stdout = json.dumps({"regime_assessment": "bullish", "trades": []})
        result = invoker._parse_response(stdout)
        assert result["regime_assessment"] == "bullish"

    def test_parse_response_wrapped_json(self, tmp_path: Path):
        invoker = ClaudeInvoker(workspace=tmp_path)
        inner = json.dumps({"regime_assessment": "bearish", "trades": [{"ticker": "GLD"}]})
        stdout = json.dumps({"result": inner, "session_id": "sess-1"})
        result = invoker._parse_response(stdout)
        assert result["regime_assessment"] == "bearish"
        assert len(result["trades"]) == 1

    def test_parse_response_markdown_block(self, tmp_path: Path):
        invoker = ClaudeInvoker(workspace=tmp_path)
        text = 'Here is my analysis:\n```json\n{"trades": [{"ticker": "SPY"}], "reasoning": "test"}\n```\nDone.'
        result = invoker._extract_json_from_text(text)
        assert result["trades"][0]["ticker"] == "SPY"

    def test_parse_response_fallback(self, tmp_path: Path):
        invoker = ClaudeInvoker(workspace=tmp_path)
        result = invoker._parse_response("not json at all")
        assert result["trades"] == []
        assert "not json" in result["reasoning"]

    def test_extract_json_non_greedy(self, tmp_path: Path):
        """Non-greedy regex should extract first valid JSON, not greedy match."""
        invoker = ClaudeInvoker(workspace=tmp_path)
        text = 'First {"a": 1} then {"b": 2} end'
        result = invoker._extract_json_from_text(text)
        assert result == {"a": 1}

    def test_extract_json_skips_invalid(self, tmp_path: Path):
        """Should skip invalid JSON fragments and find valid one."""
        invoker = ClaudeInvoker(workspace=tmp_path)
        text = 'Bad {invalid} then {"valid": true} end'
        result = invoker._extract_json_from_text(text)
        assert result == {"valid": True}

    def test_extract_cli_metadata(self, tmp_path: Path):
        invoker = ClaudeInvoker(workspace=tmp_path)
        stdout = json.dumps({
            "result": "hello",
            "session_id": "abc-def",
            "num_turns": 4,
            "duration_ms": 30000,
        })
        meta = invoker._extract_cli_metadata(stdout)
        assert meta["session_id"] == "abc-def"
        assert meta["num_turns"] == 4
        assert meta["duration_ms"] == 30000

    def test_extract_cli_metadata_missing_fields(self, tmp_path: Path):
        invoker = ClaudeInvoker(workspace=tmp_path)
        stdout = json.dumps({"result": "hello", "session_id": "abc"})
        meta = invoker._extract_cli_metadata(stdout)
        assert meta["session_id"] == "abc"
        assert meta["num_turns"] == 0
        assert meta["duration_ms"] == 0

    def test_extract_cli_metadata_invalid_json(self, tmp_path: Path):
        invoker = ClaudeInvoker(workspace=tmp_path)
        meta = invoker._extract_cli_metadata("not json")
        assert meta == {}

    def test_extract_cli_metadata_null_values(self, tmp_path: Path):
        """Null values in CLI output should default to 0, not None."""
        invoker = ClaudeInvoker(workspace=tmp_path)
        stdout = json.dumps({
            "result": "hello",
            "session_id": "s1",
            "num_turns": None,
            "duration_ms": None,
        })
        meta = invoker._extract_cli_metadata(stdout)
        assert meta["num_turns"] == 0
        assert meta["duration_ms"] == 0

    def test_session_id_persistence(self, tmp_path: Path):
        invoker = ClaudeInvoker(workspace=tmp_path)
        today = date(2024, 6, 15)

        # No session ID initially
        assert invoker._get_daily_session_id(today) is None

        # Save and retrieve
        invoker._save_daily_session_id(today, "sess-abc")
        assert invoker._get_daily_session_id(today) == "sess-abc"

        # Different day returns None
        assert invoker._get_daily_session_id(date(2024, 6, 16)) is None

    def test_read_session_results(self, tmp_path: Path):
        invoker = ClaudeInvoker(workspace=tmp_path)
        results_path = tmp_path / "default" / "session-results.json"
        results_path.write_text(
            json.dumps(
                {
                    "trades": [{"ticker": "NVDA"}],
                    "declared_posture": "aggressive",
                    "trailing_stop_requests": [{"ticker": "NVDA", "trail_pct": 0.07}],
                }
            )
        )
        data = invoker._read_session_results(results_path)
        assert data["declared_posture"] == "aggressive"
        assert not results_path.exists()  # Cleaned up

    def test_read_session_results_missing(self, tmp_path: Path):
        invoker = ClaudeInvoker(workspace=tmp_path)
        data = invoker._read_session_results(tmp_path / "nonexistent.json", retries=1, delay=0.01)
        assert data == {}

    def test_read_session_results_retries_on_missing(self, tmp_path: Path):
        """Should retry when file doesn't exist yet (MCP grandchild flush race)."""
        invoker = ClaudeInvoker(workspace=tmp_path)
        results_path = tmp_path / "default" / "session-results.json"

        # File appears on 3rd attempt
        call_count = 0

        def delayed_exists():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return False
            results_path.write_text(json.dumps({"posture": "defensive"}))
            return True

        with patch.object(type(results_path), "exists", side_effect=delayed_exists):
            data = invoker._read_session_results(results_path, retries=5, delay=0.01)

        assert data == {"posture": "defensive"}

    def test_write_mcp_config(self, tmp_path: Path):
        invoker = ClaudeInvoker(workspace=tmp_path, tenant_id="t1")
        path = invoker._write_mcp_config()
        assert path.exists()
        data = json.loads(path.read_text())
        server = data["mcpServers"]["kukulkan"]
        assert server["type"] == "stdio"
        assert "python" in server["command"] or ".venv" in server["command"]
        assert "mcp_server.py" in server["args"][0]
        # Session state points to tenant workspace
        assert "t1" in server["env"]["KUKULKAN_SESSION_STATE"]

    @pytest.mark.asyncio
    async def test_invoke_validates_session_type(self, tmp_path: Path):
        invoker = ClaudeInvoker(workspace=tmp_path)
        with pytest.raises(ValueError, match="Invalid session_type"):
            await invoker.invoke("bogus")

    @pytest.mark.asyncio
    async def test_invoke_accepts_all_valid_types(self, tmp_path: Path):
        """All VALID_SESSION_TYPES should pass validation (test early exit via mock)."""
        invoker = ClaudeInvoker(workspace=tmp_path)
        for st in VALID_SESSION_TYPES:
            mock_result = subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps({"result": "{}", "session_id": "s1"}),
                stderr="",
            )
            with patch("src.agent.claude_invoker._run_with_kill", return_value=mock_result):
                result = await invoker.invoke(st, today=date(2024, 6, 15))
            assert result.error is None

    @pytest.mark.asyncio
    async def test_invoke_subprocess_success(self, tmp_path: Path):
        invoker = ClaudeInvoker(workspace=tmp_path, timeout=30)

        cli_output = json.dumps(
            {
                "result": json.dumps({"regime_assessment": "bull", "trades": []}),
                "session_id": "sess-new",
                "num_turns": 5,
                "duration_ms": 25000,
            }
        )

        mock_result = subprocess.CompletedProcess(
            args=["claude"],
            returncode=0,
            stdout=cli_output,
            stderr="",
        )

        with patch("src.agent.claude_invoker._run_with_kill", return_value=mock_result):
            result = await invoker.invoke("morning", today=date(2024, 6, 15))

        assert result.error is None
        assert result.session_id == "sess-new"
        assert result.response["regime_assessment"] == "bull"
        assert result.num_turns == 5
        assert result.duration_ms == 25000

    @pytest.mark.asyncio
    async def test_invoke_subprocess_failure(self, tmp_path: Path):
        invoker = ClaudeInvoker(workspace=tmp_path)

        mock_result = subprocess.CompletedProcess(
            args=["claude"],
            returncode=1,
            stdout="",
            stderr="claude: command not found",
        )

        with patch("src.agent.claude_invoker._run_with_kill", return_value=mock_result):
            result = await invoker.invoke("morning", today=date(2024, 6, 15))

        assert result.error is not None
        assert "Exit code 1" in result.error

    @pytest.mark.asyncio
    async def test_invoke_timeout(self, tmp_path: Path):
        invoker = ClaudeInvoker(workspace=tmp_path, timeout=1)

        with patch(
            "src.agent.claude_invoker._run_with_kill",
            side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=1),
        ):
            result = await invoker.invoke("morning", today=date(2024, 6, 15))

        assert result.error == "Session timed out"

    @pytest.mark.asyncio
    async def test_invoke_resumes_midday(self, tmp_path: Path):
        """Midday session should use --resume with morning's session ID."""
        invoker = ClaudeInvoker(workspace=tmp_path)
        today = date(2024, 6, 15)

        # Simulate morning session saving an ID
        invoker._save_daily_session_id(today, "morning-sess-id")

        cli_output = json.dumps(
            {
                "result": json.dumps({"trades": [], "reasoning": "no changes"}),
                "session_id": "morning-sess-id",
            }
        )

        mock_result = subprocess.CompletedProcess(
            args=["claude"],
            returncode=0,
            stdout=cli_output,
            stderr="",
        )

        with patch("src.agent.claude_invoker._run_with_kill", return_value=mock_result) as mock_run:
            await invoker.invoke("midday", today=today)

        # Verify --resume was passed — first positional arg is the cmd list
        call_args = mock_run.call_args[0][0]
        assert "--resume" in call_args
        idx = call_args.index("--resume")
        assert call_args[idx + 1] == "morning-sess-id"

    @pytest.mark.asyncio
    async def test_invoke_env_strips_api_key(self, tmp_path: Path):
        """Ensure ANTHROPIC_API_KEY is never passed to Claude Code (use Max sub)."""
        invoker = ClaudeInvoker(workspace=tmp_path)

        mock_result = subprocess.CompletedProcess(
            args=["claude"],
            returncode=0,
            stdout=json.dumps({"result": "{}", "session_id": "s1"}),
            stderr="",
        )

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=False):
            with patch("src.agent.claude_invoker._run_with_kill", return_value=mock_result) as mock_run:
                await invoker.invoke("morning", today=date(2024, 6, 15))

        # Second positional arg is the env dict
        call_env = mock_run.call_args[0][1]
        assert "ANTHROPIC_API_KEY" not in call_env


# ── _run_with_kill tests ─────────────────────────────────────────────────────


class TestRunWithKill:
    def test_successful_run(self):
        result = _run_with_kill(["echo", "hello"], env=dict(os.environ), timeout=10)
        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_timeout_kills_process(self):
        with pytest.raises(subprocess.TimeoutExpired):
            _run_with_kill(["sleep", "60"], env=dict(os.environ), timeout=1)

    def test_captures_stderr(self):
        result = _run_with_kill(
            ["python", "-c", "import sys; sys.stderr.write('err')"],
            env=dict(os.environ),
            timeout=10,
        )
        assert "err" in result.stderr


# ── VALID_SESSION_TYPES tests ───────────────────────────────────────────────


class TestValidSessionTypes:
    def test_contains_expected_types(self):
        assert "morning" in VALID_SESSION_TYPES
        assert "midday" in VALID_SESSION_TYPES
        assert "closing" in VALID_SESSION_TYPES
        assert "manual" in VALID_SESSION_TYPES
        assert "event" in VALID_SESSION_TYPES
        assert "sentinel-crisis" in VALID_SESSION_TYPES

    def test_is_frozen(self):
        assert isinstance(VALID_SESSION_TYPES, frozenset)
