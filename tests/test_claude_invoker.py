"""Tests for Claude Code CLI invoker and context builders."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent.claude_invoker import ClaudeInvoker, InvokeResult, write_context_file, write_session_state

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
            accumulated={"trailing_stop_requests": [], "declared_posture": None},
        )
        summary = r.tool_summary
        assert "trailing_stop_requests" in summary
        assert "declared_posture" in summary
        assert summary["source"] == "claude_code"


# ── ClaudeInvoker tests ─────────────────────────────────────────────────────


class TestClaudeInvoker:
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

    def test_extract_session_id(self, tmp_path: Path):
        invoker = ClaudeInvoker(workspace=tmp_path)
        stdout = json.dumps({"result": "hello", "session_id": "abc-def"})
        assert invoker._extract_session_id(stdout) == "abc-def"

    def test_extract_session_id_missing(self, tmp_path: Path):
        invoker = ClaudeInvoker(workspace=tmp_path)
        assert invoker._extract_session_id("not json") is None

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
        results_path = tmp_path / "session-results.json"
        results_path.write_text(json.dumps({
            "trades": [{"ticker": "NVDA"}],
            "declared_posture": "aggressive",
            "trailing_stop_requests": [{"ticker": "NVDA", "trail_pct": 0.07}],
        }))
        data = invoker._read_session_results(results_path)
        assert data["declared_posture"] == "aggressive"
        assert not results_path.exists()  # Cleaned up

    def test_read_session_results_missing(self, tmp_path: Path):
        invoker = ClaudeInvoker(workspace=tmp_path)
        data = invoker._read_session_results(tmp_path / "nonexistent.json")
        assert data == {}

    @pytest.mark.asyncio
    async def test_invoke_subprocess_success(self, tmp_path: Path):
        invoker = ClaudeInvoker(workspace=tmp_path, timeout=30)

        cli_output = json.dumps({
            "result": json.dumps({"regime_assessment": "bull", "trades": []}),
            "session_id": "sess-new",
        })

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = cli_output
        mock_result.stderr = ""

        with patch("src.agent.claude_invoker.subprocess.run", return_value=mock_result):
            result = await invoker.invoke("morning", today=date(2024, 6, 15))

        assert result.error is None
        assert result.session_id == "sess-new"
        assert result.response["regime_assessment"] == "bull"

    @pytest.mark.asyncio
    async def test_invoke_subprocess_failure(self, tmp_path: Path):
        invoker = ClaudeInvoker(workspace=tmp_path)

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "claude: command not found"

        with patch("src.agent.claude_invoker.subprocess.run", return_value=mock_result):
            result = await invoker.invoke("morning", today=date(2024, 6, 15))

        assert result.error is not None
        assert "Exit code 1" in result.error

    @pytest.mark.asyncio
    async def test_invoke_timeout(self, tmp_path: Path):
        import subprocess

        invoker = ClaudeInvoker(workspace=tmp_path, timeout=1)

        with patch(
            "src.agent.claude_invoker.subprocess.run",
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

        cli_output = json.dumps({
            "result": json.dumps({"trades": [], "reasoning": "no changes"}),
            "session_id": "morning-sess-id",
        })

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = cli_output
        mock_result.stderr = ""

        with patch("src.agent.claude_invoker.subprocess.run", return_value=mock_result) as mock_run:
            await invoker.invoke("midday", today=today)

        # Verify --resume was passed
        call_args = mock_run.call_args[0][0]
        assert "--resume" in call_args
        idx = call_args.index("--resume")
        assert call_args[idx + 1] == "morning-sess-id"

    @pytest.mark.asyncio
    async def test_invoke_env_strips_api_key(self, tmp_path: Path):
        """Ensure ANTHROPIC_API_KEY is never passed to Claude Code (use Max sub)."""
        invoker = ClaudeInvoker(workspace=tmp_path)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"result": "{}", "session_id": "s1"})
        mock_result.stderr = ""

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=False):
            with patch("src.agent.claude_invoker.subprocess.run", return_value=mock_result) as mock_run:
                await invoker.invoke("morning", today=date(2024, 6, 15))

        # Check env passed to subprocess
        call_kwargs = mock_run.call_args[1]
        assert "ANTHROPIC_API_KEY" not in call_kwargs["env"]
