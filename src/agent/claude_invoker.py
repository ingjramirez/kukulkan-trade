"""Claude Code CLI invoker for Portfolio B trading sessions.

Replaces AgentRunner + PersistentAgent with subprocess calls to `claude -p`.
Uses Claude Max subscription (no API key, no rate limits, no budget tracking).

Architecture:
    Orchestrator → writes session-state.json + context.md
                 → ClaudeInvoker.invoke() → subprocess `claude -p`
                 → Claude Code spawns MCP server (reads session-state.json)
                 → Claude reads context.md, calls MCP tools, returns JSON
                 → Invoker reads JSON + session-results.json → returns result
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import structlog

log = structlog.get_logger()

# Default workspace (resolved relative to project root)
WORKSPACE = Path(__file__).resolve().parent.parent.parent / "data" / "agent-workspace"


# ── Result types ─────────────────────────────────────────────────────────────


@dataclass
class InvokeResult:
    """Result of a Claude Code CLI invocation."""

    response: dict = field(default_factory=dict)
    session_id: str | None = None
    accumulated: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def trades(self) -> list[dict]:
        return self.response.get("trades", [])

    @property
    def reasoning(self) -> str:
        return self.response.get("reasoning", "")

    @property
    def posture(self) -> str | None:
        return self.response.get("posture") or self.accumulated.get("declared_posture")

    @property
    def trailing_stop_requests(self) -> list[dict]:
        """Merge trailing stops from JSON response and MCP ActionState."""
        from_response = self.response.get("trailing_stops", [])
        from_accumulated = self.accumulated.get("trailing_stop_requests", [])
        return from_response or from_accumulated

    @property
    def tool_summary(self) -> dict:
        return {
            "trailing_stop_requests": self.trailing_stop_requests,
            "declared_posture": self.posture,
            "source": "claude_code",
        }


# ── Session state writer ────────────────────────────────────────────────────


def write_session_state(
    workspace: Path,
    tenant_id: str,
    closes_dict: dict,
    closes_index: list[str],
    current_prices: dict[str, float],
    held_tickers: list[str],
    vix: float | None = None,
    yield_curve: float | None = None,
    regime: str | None = None,
    news_context: str = "",
    fear_greed: dict | None = None,
) -> Path:
    """Write session-state.json for MCP server tool initialization.

    This file is read by mcp_server.py on startup to reconstruct
    DataFrames and initialize tools with market data.
    """
    state = {
        "tenant_id": tenant_id,
        "closes": closes_dict,
        "closes_index": closes_index,
        "current_prices": current_prices,
        "held_tickers": held_tickers,
        "vix": vix,
        "yield_curve": yield_curve,
        "regime": regime,
        "news_context": news_context,
        "fear_greed": fear_greed,
    }
    out = workspace / "session-state.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, default=str))
    tmp.rename(out)
    return out


# ── Context file writer ─────────────────────────────────────────────────────


def write_context_file(
    workspace: Path,
    session_type: str,
    today: date,
    regime: str | None,
    vix: float | None,
    yield_curve: float | None,
    cash: float,
    total_value: float,
    positions: list[dict],
    signal_text: str | None = None,
    fear_greed: dict | None = None,
    sentinel_alerts: list[str] | None = None,
    earnings_context: str | None = None,
    news_context: str = "",
    pinned_context: str | None = None,
    trailing_stops_context: str | None = None,
    watchlist_context: str | None = None,
) -> Path:
    """Write context.md for Claude Code to read as prompt context.

    This is human-readable market state — the agent's briefing document.
    """
    now = datetime.now()
    lines = [
        f"# Trading Session: {session_type.title()}",
        f"**Date**: {today.isoformat()}  **Time**: {now.strftime('%H:%M')} CT",
        f"**Regime**: {regime or 'unknown'}",
        "",
        "## Macro",
        f"- VIX: {vix or 'N/A'}",
        f"- Yield Curve: {yield_curve:+.2f}%" if yield_curve is not None else "- Yield Curve: N/A",
    ]

    if fear_greed:
        lines.append(f"- Fear & Greed: {fear_greed.get('value', 'N/A')} ({fear_greed.get('classification', '')})")

    lines.extend(
        [
            "",
            "## Portfolio",
            f"- Cash: ${cash:,.2f}",
            f"- Total Value: ${total_value:,.2f}",
            f"- Positions: {len(positions)}",
        ]
    )

    if positions:
        lines.append("")
        lines.append("### Current Positions")
        for pos in positions:
            mv = pos.get("market_value", 0)
            pnl = mv - (pos.get("shares", 0) * pos.get("avg_price", 0)) if mv else 0
            lines.append(
                f"- {pos['ticker']}: {pos.get('shares', 0)} shares "
                f"@ ${pos.get('avg_price', 0):.2f} (MV ${mv:,.2f}, P&L ${pnl:+,.2f})"
            )

    if signal_text:
        lines.extend(["", "## Signal Rankings", signal_text])

    if sentinel_alerts:
        lines.extend(["", "## Sentinel Alerts"])
        for alert in sentinel_alerts:
            lines.append(f"- {alert}")

    if earnings_context:
        lines.extend(["", "## Upcoming Earnings", earnings_context])

    if news_context:
        lines.extend(["", "## News Summary", news_context])

    if trailing_stops_context:
        lines.extend(["", "## Active Trailing Stops", trailing_stops_context])

    if watchlist_context:
        lines.extend(["", "## Your Watchlist", watchlist_context])

    if pinned_context:
        lines.extend(["", pinned_context])

    out = workspace / "context.md"
    tmp = out.with_suffix(".tmp")
    content = "\n".join(lines)
    tmp.write_text(content)
    tmp.rename(out)
    return out


# ── CLI Invoker ──────────────────────────────────────────────────────────────


class ClaudeInvoker:
    """Invoke Claude Code CLI as a subprocess for trading sessions.

    Replaces AgentRunner + PersistentAgent + RequestPacer + TokenTracker.
    Uses Claude Max subscription — no API key, no rate limits, no budget tracking.
    """

    def __init__(
        self,
        workspace: Path = WORKSPACE,
        timeout: int = 600,
        max_turns: int = 25,
        model: str = "claude-sonnet-4-6",
    ):
        self._workspace = workspace
        self._timeout = timeout
        self._max_turns = max_turns
        self._model = model
        self._session_id_file = workspace / ".session-id"

    def _get_daily_session_id(self, today: date) -> str | None:
        """Read today's session ID from file (survives process restarts)."""
        if not self._session_id_file.exists():
            return None
        try:
            data = json.loads(self._session_id_file.read_text())
            if data.get("date") == today.isoformat():
                return data.get("session_id")
        except (json.JSONDecodeError, KeyError):
            pass
        return None

    def _save_daily_session_id(self, today: date, session_id: str) -> None:
        """Persist session ID for --resume across invocations."""
        data = {"date": today.isoformat(), "session_id": session_id}
        self._session_id_file.write_text(json.dumps(data))

    async def invoke(
        self,
        session_type: str,
        today: date | None = None,
    ) -> InvokeResult:
        """Run a trading session via Claude Code CLI.

        Args:
            session_type: "morning", "midday", "closing", "manual", "event"
            today: Current date (default: today)

        Returns:
            InvokeResult with parsed response, session_id, and accumulated actions.
        """
        today = today or date.today()

        # Session strategy: morning starts new, midday/close resume
        is_new = session_type == "morning" or self._get_daily_session_id(today) is None
        session_id = None if is_new else self._get_daily_session_id(today)

        cmd = self._build_cmd(session_type, session_id)

        # Ensure ANTHROPIC_API_KEY is NOT set (use Max subscription auth)
        env = {**os.environ}
        env.pop("ANTHROPIC_API_KEY", None)

        log.info(
            "claude_invoke_start",
            session_type=session_type,
            is_new=is_new,
            resume_id=session_id,
        )

        # Clean previous session results
        results_path = self._workspace / "session-results.json"
        if results_path.exists():
            results_path.unlink()

        try:
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                cwd=str(self._workspace),
                env=env,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )

            if result.returncode != 0:
                log.error(
                    "claude_invoke_failed",
                    returncode=result.returncode,
                    stderr=result.stderr[:500] if result.stderr else "",
                )
                return InvokeResult(error=f"Exit code {result.returncode}: {(result.stderr or '')[:200]}")

            # Parse CLI JSON output
            response = self._parse_response(result.stdout)

            # Extract and persist session ID for future --resume
            new_session_id = self._extract_session_id(result.stdout)
            if new_session_id:
                self._save_daily_session_id(today, new_session_id)

            # Read accumulated ActionState from MCP server
            accumulated = self._read_session_results(results_path)

            log.info(
                "claude_invoke_complete",
                session_type=session_type,
                trades=len(response.get("trades", [])),
                session_id=new_session_id,
            )

            return InvokeResult(
                response=response,
                session_id=new_session_id or session_id,
                accumulated=accumulated,
            )

        except subprocess.TimeoutExpired:
            log.error("claude_invoke_timeout", timeout=self._timeout)
            accumulated = self._read_session_results(results_path)
            return InvokeResult(error="Session timed out", accumulated=accumulated)

        except Exception as e:
            log.error("claude_invoke_exception", error=str(e))
            return InvokeResult(error=str(e))

    def _build_cmd(self, session_type: str, session_id: str | None) -> list[str]:
        """Build the claude CLI command."""
        prompt = (
            f"Session type: {session_type}. "
            "Read context.md for current market state, then analyze and trade. "
            "Return your final analysis as JSON matching the Output Format in CLAUDE.md."
        )

        cmd = [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "json",
            "--mcp-config",
            str(self._workspace / "mcp.json"),
            "--allowedTools",
            "mcp__kukulkan__*",
            "--max-turns",
            str(self._max_turns),
            "--model",
            self._model,
        ]

        if session_id:
            cmd.extend(["--resume", session_id])

        return cmd

    def _parse_response(self, stdout: str) -> dict:
        """Parse Claude Code JSON output into a trading response dict."""
        if not stdout.strip():
            return {}

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            # Try to extract JSON from mixed output
            return self._extract_json_from_text(stdout)

        # Claude Code --output-format json wraps in {"result": "...", "session_id": "..."}
        if "result" in data and isinstance(data["result"], str):
            return self._extract_json_from_text(data["result"])

        # Direct JSON response
        return data

    def _extract_json_from_text(self, text: str) -> dict:
        """Extract a JSON object from text that may contain markdown or other content."""
        # Try to find JSON block (possibly in ```json ... ```)
        code_block = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        if code_block:
            try:
                return json.loads(code_block.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find bare JSON object
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: return text as reasoning
        return {"reasoning": text[:1000], "trades": []}

    def _extract_session_id(self, stdout: str) -> str | None:
        """Extract session ID from Claude Code JSON output."""
        try:
            data = json.loads(stdout)
            return data.get("session_id")
        except (json.JSONDecodeError, TypeError):
            return None

    def _read_session_results(self, results_path: Path) -> dict:
        """Read accumulated ActionState written by MCP server on exit."""
        if not results_path.exists():
            return {}
        try:
            data = json.loads(results_path.read_text())
            results_path.unlink()  # Clean up
            return data
        except (json.JSONDecodeError, OSError) as e:
            log.warning("session_results_read_failed", error=str(e))
            return {}
