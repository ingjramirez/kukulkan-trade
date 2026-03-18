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
import signal
import subprocess
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import structlog

log = structlog.get_logger()

# Project root (resolved once)
_project_root = Path(__file__).resolve().parent.parent.parent

# Default workspace (resolved relative to project root)
WORKSPACE = _project_root / "data" / "agent-workspace"

# Valid session types for invoke()
VALID_SESSION_TYPES = frozenset({"morning", "midday", "closing", "manual", "event", "sentinel-crisis"})

# System prompt appended for interactive chat sessions (via --append-system-prompt).
# Overrides the default JSON-output expectation so Claude responds conversationally.
CHAT_SYSTEM_PROMPT = """\
## Chat Mode
You are now in direct interactive chat with the portfolio owner via the Kukulkan dashboard.

- Respond conversationally. Do NOT output the trading JSON summary format.
- Use MCP tools when you need live data (portfolio state, prices, news, signals).
- You CAN execute trades if the user explicitly asks — confirm first, then use execute_trade.
- Keep answers concise (under 400 words) unless the user asks for detail.
- Reference today's context.md if asked about the current session.
- When recommending a trade, explain your reasoning before executing.
"""


# ── Result types ─────────────────────────────────────────────────────────────


@dataclass
class InvokeResult:
    """Result of a Claude Code CLI invocation."""

    response: dict = field(default_factory=dict)
    session_id: str | None = None
    accumulated: dict = field(default_factory=dict)
    error: str | None = None
    num_turns: int = 0
    duration_ms: int = 0

    @property
    def trades(self) -> list[dict]:
        return self.response.get("trades", [])

    @property
    def mcp_executed_trades(self) -> list[dict]:
        """Trades that were directly filled via MCP execute_trade tool."""
        return [t for t in self.accumulated.get("executed_trades", []) if t.get("status") == "filled"]

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
    def tools_used(self) -> int:
        """Tool call count from MCP session-results.json."""
        return self.accumulated.get("tool_call_count", 0)

    @property
    def tool_call_logs(self) -> list[dict]:
        """Per-call tool logs from MCP session-results.json."""
        return self.accumulated.get("tool_call_logs", [])

    @property
    def tool_summary(self) -> dict:
        return {
            "trailing_stop_requests": self.trailing_stop_requests,
            "declared_posture": self.posture,
            "source": "claude_code",
            "tools_used": self.tools_used,
            "turns": self.num_turns,
            "duration_ms": self.duration_ms,
            "mcp_executed_trades": self.mcp_executed_trades,
        }


@dataclass
class ChatResult:
    """Result of an interactive chat invocation."""

    content: str = ""
    session_id: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    num_turns: int = 0
    duration_ms: int = 0
    error: str | None = None
    accumulated: dict = field(default_factory=dict)


# ── Process management ─────────────────────────────────────────────────────


def _run_with_kill(
    cmd: list[str],
    env: dict[str, str],
    timeout: int,
    cwd: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run subprocess with process-group kill on timeout.

    Uses os.setsid to create a process group so we can kill the entire tree
    (Claude Code + MCP server grandchild) on timeout instead of leaving zombies.
    """
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=os.setsid,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        # Kill entire process group (child + grandchildren)
        pgid = os.getpgid(proc.pid)
        try:
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                proc.wait(timeout=5)
        except ProcessLookupError:
            pass  # Already exited
        raise


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
    sync_metadata: dict | None = None,
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
        "sync_metadata": sync_metadata,
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
    sync_warning: str | None = None,
) -> Path:
    """Write context.md for Claude Code to read as prompt context.

    This is human-readable market state — the agent's briefing document.
    """
    now = datetime.now()
    lines = [
        f"# Trading Session: {session_type.title()}",
        f"**Date**: {today.isoformat()}  **Time**: {now.strftime('%H:%M')} CT",
        f"**Regime**: {regime or 'unknown'}",
    ]

    if sync_warning:
        lines.extend(["", f"> {sync_warning}"])

    lines.extend(
        [
            "",
            "## Macro",
            f"- VIX: {vix or 'N/A'}",
            f"- Yield Curve: {yield_curve:+.2f}%" if yield_curve is not None else "- Yield Curve: N/A",
        ]
    )

    if fear_greed:
        lines.append(f"- Fear & Greed: {fear_greed.get('value', 'N/A')} ({fear_greed.get('classification', '')})")

    cash_pct = (cash / total_value * 100) if total_value > 0 else 0
    cash_alert = ""
    if cash_pct >= 40:
        cash_alert = f" ⚠️ HIGH CASH ({cash_pct:.0f}%) — DEPLOY into top-ranked commodities/discretionary"
    elif cash_pct >= 25:
        cash_alert = f" (cash {cash_pct:.0f}% — consider deploying)"

    lines.extend(
        [
            "",
            "## Portfolio",
            f"- Cash: ${cash:,.2f}{cash_alert}",
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
        tenant_id: str = "default",
    ):
        self._root_workspace = workspace
        self._workspace = workspace / tenant_id
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._timeout = timeout
        self._max_turns = max_turns
        self._model = model
        self._tenant_id = tenant_id
        self._session_id_file = self._workspace / ".session-id"
        self._chat_session_id_file = self._session_id_file  # unified: chat + trading share one session

    def _get_daily_session_id(self, today: date) -> str | None:
        """Read the persistent session ID (no date expiry — sessions live forever)."""
        if not self._session_id_file.exists():
            return None
        try:
            data = json.loads(self._session_id_file.read_text())
            return data.get("session_id")
        except (json.JSONDecodeError, KeyError):
            return None

    def _save_daily_session_id(self, today: date, session_id: str) -> None:
        """Persist session ID for --resume across invocations."""
        self._workspace.mkdir(parents=True, exist_ok=True)
        data = {"date": today.isoformat(), "session_id": session_id}
        self._session_id_file.write_text(json.dumps(data))

    def _clear_daily_session_id(self, today: date) -> None:
        """Remove session ID so the next invocation starts fresh."""
        if self._session_id_file.exists():
            self._session_id_file.unlink()
            log.info("session_id_cleared", tenant_id=self._tenant_id)

    def _get_chat_session_id(self) -> str | None:
        """Read the persistent chat session ID (no date expiry)."""
        if not self._chat_session_id_file.exists():
            return None
        try:
            data = json.loads(self._chat_session_id_file.read_text())
            return data.get("session_id")
        except (json.JSONDecodeError, KeyError):
            return None

    def _save_chat_session_id(self, session_id: str) -> None:
        """Persist chat session ID for continuous conversation."""
        self._chat_session_id_file.write_text(json.dumps({"session_id": session_id}))

    def _clear_chat_session_id(self) -> None:
        """Remove stale chat session ID so the next message starts fresh."""
        if self._chat_session_id_file.exists():
            self._chat_session_id_file.unlink()
            log.info("chat_session_id_cleared_persistent", tenant_id=self._tenant_id)

    @staticmethod
    def _database_url() -> str:
        """Resolve database URL from settings (lazy import to avoid circular deps).

        For SQLite URLs, ensures the path is absolute (relative to project root).
        PostgreSQL URLs are passed through as-is.
        """
        from config.settings import settings

        url = settings.database_url
        if url.startswith("sqlite"):
            # Convert relative SQLite path to absolute
            path = url.split("///", 1)[-1]
            if not Path(path).is_absolute():
                url = f"sqlite+aiosqlite:///{_project_root / path}"
        return url

    def _write_mcp_config(self) -> Path:
        """Generate mcp.json dynamically with resolved paths for this tenant workspace."""
        venv_python = _project_root / ".venv" / "bin" / "python"
        python_cmd = str(venv_python) if venv_python.exists() else "python"

        config = {
            "mcpServers": {
                "kukulkan": {
                    "type": "stdio",
                    "command": python_cmd,
                    "args": [str(_project_root / "src" / "agent" / "mcp_server.py")],
                    "env": {
                        "DATABASE_URL": self._database_url(),
                        "KUKULKAN_SESSION_STATE": str(self._workspace / "session-state.json"),
                        "TOOL_RESULT_MAX_CHARS": "3000",
                    },
                }
            }
        }
        out = self._workspace / "mcp.json"
        out.write_text(json.dumps(config, indent=2))
        return out

    async def invoke(
        self,
        session_type: str,
        today: date | None = None,
    ) -> InvokeResult:
        """Run a trading session via Claude Code CLI.

        Args:
            session_type: "morning", "midday", "closing", "manual", "event", "sentinel-crisis"
            today: Current date (default: today)

        Returns:
            InvokeResult with parsed response, session_id, and accumulated actions.
        """
        if session_type not in VALID_SESSION_TYPES:
            raise ValueError(f"Invalid session_type {session_type!r}. Must be one of {sorted(VALID_SESSION_TYPES)}")

        today = today or date.today()

        # Always resume the existing session to preserve full conversation history.
        # Only start new if no session exists yet.
        session_id = self._get_daily_session_id(today)
        is_new = session_id is None

        # Generate mcp.json with resolved paths for this tenant
        self._write_mcp_config()

        cmd = self._build_cmd(session_type, session_id)

        env = {**os.environ}

        log.info(
            "claude_invoke_start",
            session_type=session_type,
            is_new=is_new,
            resume_id=session_id,
            tenant_id=self._tenant_id,
        )

        # Clean previous session results
        results_path = self._workspace / "session-results.json"
        if results_path.exists():
            results_path.unlink()

        try:
            result = await asyncio.to_thread(
                _run_with_kill,
                cmd,
                env,
                self._timeout,
                cwd=str(self._root_workspace),
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

            # Read MCP results BEFORE retry decision — tools may have executed real trades
            pre_retry_accumulated = self._read_session_results(results_path)
            mcp_executed = [t for t in pre_retry_accumulated.get("executed_trades", []) if t.get("status") == "filled"]

            # Detect "empty message" or lazy responses — but NOT if real trades were executed
            needs_retry = False
            if mcp_executed:
                log.info(
                    "claude_skip_retry_mcp_trades_filled",
                    mcp_trades=len(mcp_executed),
                    tickers=[t["ticker"] for t in mcp_executed],
                    session_type=session_type,
                )
            elif session_id and self._is_empty_message_response(response):
                log.warning("claude_resume_empty_message_detected", session_id=session_id, session_type=session_type)
                needs_retry = True
            elif session_id and self._is_lazy_response(response):
                log.warning(
                    "claude_lazy_response_detected",
                    session_id=session_id,
                    session_type=session_type,
                    reasoning=(response.get("reasoning") or "")[:200],
                )
                needs_retry = True

            if needs_retry:
                retry_cmd = self._build_retry_cmd(session_type, session_id)
                if results_path.exists():
                    results_path.unlink()
                result = await asyncio.to_thread(
                    _run_with_kill,
                    retry_cmd,
                    env,
                    self._timeout,
                    cwd=str(self._root_workspace),
                )
                if result.returncode == 0:
                    response = self._parse_response(result.stdout)
                    log.info("claude_resume_retry_succeeded", session_type=session_type)

            # Extract metadata (session_id, turns, duration) from CLI wrapper
            meta = self._extract_cli_metadata(result.stdout)
            new_session_id = meta.get("session_id")
            if new_session_id:
                self._save_daily_session_id(today, new_session_id)

            # Read accumulated ActionState from MCP server (written after every tool call)
            # If we skipped retry, pre_retry_accumulated already consumed the file
            if needs_retry:
                accumulated = self._read_session_results(results_path)
            else:
                accumulated = pre_retry_accumulated

            log.info(
                "claude_invoke_complete",
                session_type=session_type,
                trades=len(response.get("trades", [])),
                session_id=new_session_id,
                num_turns=meta.get("num_turns", 0),
                tools_used=accumulated.get("tool_call_count", 0),
            )

            return InvokeResult(
                response=response,
                session_id=new_session_id or session_id,
                accumulated=accumulated,
                num_turns=meta.get("num_turns", 0),
                duration_ms=meta.get("duration_ms", 0),
            )

        except subprocess.TimeoutExpired:
            log.error("claude_invoke_timeout", timeout=self._timeout)
            accumulated = self._read_session_results(results_path)
            return InvokeResult(error="Session timed out", accumulated=accumulated)

        except Exception as e:
            log.error("claude_invoke_exception", error=str(e))
            return InvokeResult(error=str(e))

    _SESSION_TASK_MAP: dict[str, str] = {
        "morning": (
            "TASKS: (1) Check regime + VIX for session posture. "
            "(2) Review signal rankings — identify top 3 non-held tickers with momentum. "
            "(3) Check cash level — if cash > 30% of portfolio, deploy into top-ranked commodities or discretionary. "
            "(4) Enter new positions with trailing stops. "
            "(5) Exit any positions in sectors with documented <30% win rate."
        ),
        "midday": (
            "TASKS: (1) Scan held positions — take partial profits on any up >3% intraday. "
            "(2) Tighten trailing stops on winners (ratchet up, never down). "
            "(3) Exit positions that have broken below key support. "
            "(4) Do NOT open large new positions — save powder for closing session. "
            "(5) If cash > 40%, consider one small new entry in a winning sector."
        ),
        "closing": (
            "TASKS: (1) Identify overnight risk — earnings, macro events after close. "
            "(2) Trim or exit positions with earnings tonight or weekend risk (if Friday). "
            "(3) Reduce exposure in sectors with negative momentum. "
            "(4) Ensure trailing stops are set on all open positions. "
            "(5) Assess if current cash level is appropriate for overnight. "
            "This is your LAST session today — make defensive moves now."
        ),
    }

    def _build_cmd(self, session_type: str, session_id: str | None) -> list[str]:
        """Build the claude CLI command."""
        session_tasks = self._SESSION_TASK_MAP.get(
            session_type,
            "TASKS: Assess market state, review positions, execute any needed trades.",
        )
        prompt = (
            f"Session type: {session_type}. "
            "context.md has been UPDATED with fresh market data since your last turn. "
            "Re-read context.md NOW — this is NOT a duplicate of a previous session. "
            f"{session_tasks} "
            "Your 'reasoning' MUST be 3-5 sentences describing: current prices, regime, "
            "what changed since last session, and why you are holding/buying/selling. "
            "NEVER say 'already incorporated', 'stale notification', 'session complete', or 'already processed' "
            "as your reasoning — that is a bug. Each session type has different tasks. "
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

    # ── Empty-message detection & retry ──────────────────────────────────

    _EMPTY_MESSAGE_PATTERNS = (
        "message came through empty",
        "your message is empty",
        "didn't receive a message",
        "empty message",
        "what do you need",
        "how can i help",
    )

    _LAZY_RESPONSE_PATTERNS = (
        "already incorporated",
        "already handled",
        "already analyzed",
        "already reviewed",
        "session complete",
        "no changes needed",
        "no action needed",
        "no updates needed",
        "nothing to update",
        "no new information",
        "stale notification",
        "already processed",
    )

    def _is_empty_message_response(self, response: dict) -> bool:
        """Detect when Claude misread the prompt as empty on resume."""
        reasoning = (response.get("reasoning") or "").lower()
        if not reasoning:
            return False
        # No trades + reasoning matches a confused/empty pattern
        if response.get("trades"):
            return False
        return any(p in reasoning for p in self._EMPTY_MESSAGE_PATTERNS)

    def _is_lazy_response(self, response: dict) -> bool:
        """Detect when Claude shortcuts with 'already incorporated' instead of analyzing.

        Note: length gate intentionally removed — lazy patterns can appear in long responses
        (e.g. "Stale notification — already processed and session complete. All actions executed.")
        """
        reasoning = (response.get("reasoning") or "").lower()
        if not reasoning:
            return False
        if response.get("trades"):
            return False
        return any(p in reasoning for p in self._LAZY_RESPONSE_PATTERNS)

    def _build_retry_cmd(self, session_type: str, session_id: str) -> list[str]:
        """Build a retry command with a more explicit prompt after empty/lazy response."""
        prompt = (
            f"[RETRY — your previous response was rejected as insufficient] "
            f"This is a NEW {session_type} trading session with FRESH market data. "
            f"context.md has been REWRITTEN with updated prices, positions, and regime data. "
            f"You MUST re-read context.md NOW — do not rely on memory from previous turns. "
            f"Then provide a complete trading analysis as JSON matching the Output Format in CLAUDE.md. "
            f"Your 'reasoning' MUST be 3-5 sentences covering: current prices, regime assessment, "
            f"what changed since your last session, and your rationale for holding/buying/selling. "
            f"Saying 'already incorporated', 'session complete', or 'no changes' is a BUG that "
            f"will cause this retry loop to repeat. Actually analyze the fresh data."
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
            "--resume",
            session_id,
        ]
        return cmd

    def _ensure_session_state(self) -> None:
        """Write a minimal session-state.json if none exists.

        Trading sessions write a full session-state.json with live market data.
        Chat sessions that arrive before any trading session (e.g. first message
        of the day) would cause the MCP server to exit(1) — stalling Claude Code
        for 30+ seconds while it waits for MCP startup.

        A minimal state lets the MCP server start so DB-backed tools
        (portfolio, trades, watchlist, signals) work without live market data.
        """
        state_path = self._workspace / "session-state.json"
        if state_path.exists():
            return
        minimal = {
            "tenant_id": self._tenant_id,
            "closes": {},
            "closes_index": [],
            "current_prices": {},
            "held_tickers": [],
            "vix": None,
            "yield_curve": None,
            "regime": None,
            "news_context": "",
            "fear_greed": None,
        }
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(minimal))
        tmp.rename(state_path)
        log.info("chat_minimal_session_state_written", path=str(state_path))

    async def _refresh_session_state_if_stale(self) -> None:
        """Fetch live market data if session-state.json has empty closes.

        Called before chat sessions to ensure MCP tools have market data
        even when no trading session has run yet today.
        """
        state_path = self._workspace / "session-state.json"
        if not state_path.exists():
            return

        try:
            state = json.loads(state_path.read_text())
        except (json.JSONDecodeError, OSError):
            return

        # If closes already populated (by orchestrator), skip
        if state.get("closes"):
            return

        log.info("chat_refreshing_stale_session_state", tenant_id=self._tenant_id)

        try:
            import pandas as pd

            from config.universe import get_dynamic_universe
            from src.data.market_data import MarketDataFetcher
            from src.storage.database import Database

            db_url = self._database_url()
            db = Database(db_url)
            await db.init_db()

            try:
                universe = await get_dynamic_universe(db)
                mdm = MarketDataFetcher(db=db)
                data = await mdm.fetch_universe(tickers=universe, period="6mo")

                if not data:
                    log.warning("chat_refresh_no_market_data")
                    return

                closes = pd.DataFrame({t: df["Close"] for t, df in data.items()})
                closes = closes.sort_index()

                current_prices = {
                    t: float(closes[t].iloc[-1]) for t in closes.columns if not pd.isna(closes[t].iloc[-1])
                }

                # Get held tickers from positions (both portfolios)
                held_tickers: list[str] = []
                try:
                    for pf in ("A", "B"):
                        positions = await db.get_positions(pf, tenant_id=self._tenant_id)
                        held_tickers.extend(p.ticker for p in positions if p.quantity > 0)
                    held_tickers = sorted(set(held_tickers))
                except Exception:
                    pass

                # Get fear & greed
                fear_greed_data: dict | None = None
                try:
                    fg_row = await db.get_latest_sentiment(self._tenant_id, "fear_greed_index")
                    if fg_row:
                        fear_greed_data = {"value": fg_row.value, "classification": fg_row.classification}
                except Exception:
                    pass

                write_session_state(
                    workspace=self._workspace,
                    tenant_id=self._tenant_id,
                    closes_dict={
                        col: {str(k): float(v) for k, v in closes[col].dropna().items()} for col in closes.columns
                    },
                    closes_index=[str(idx) for idx in closes.index],
                    current_prices=current_prices,
                    held_tickers=held_tickers,
                    fear_greed=fear_greed_data,
                )
                log.info(
                    "chat_session_state_refreshed",
                    tickers=len(data),
                    tenant_id=self._tenant_id,
                )
            finally:
                await db.close()
        except Exception as e:
            log.warning("chat_refresh_session_state_failed", error=str(e))

    # ── Chat methods ────────────────────────────────────────────────────────

    def _build_chat_cmd(self, message: str, session_id: str | None) -> list[str]:
        """Build CLI command for interactive chat (non-streaming)."""
        cmd = [
            "claude",
            "-p",
            message,
            "--output-format",
            "json",
            "--mcp-config",
            str(self._workspace / "mcp.json"),
            "--allowedTools",
            "mcp__kukulkan__*",
            "--max-turns",
            str(min(self._max_turns, 10)),
            "--model",
            self._model,
            "--append-system-prompt",
            CHAT_SYSTEM_PROMPT,
        ]
        if session_id:
            cmd.extend(["--resume", session_id])
        return cmd

    def _build_chat_stream_cmd(self, message: str, session_id: str | None) -> list[str]:
        """Build CLI command for streaming chat.

        Note: --output-format stream-json requires --verbose when used with -p.
        """
        cmd = [
            "claude",
            "-p",
            message,
            "--output-format",
            "stream-json",
            "--verbose",
            "--mcp-config",
            str(self._workspace / "mcp.json"),
            "--allowedTools",
            "mcp__kukulkan__*",
            "--max-turns",
            str(min(self._max_turns, 10)),
            "--model",
            self._model,
            "--append-system-prompt",
            CHAT_SYSTEM_PROMPT,
        ]
        if session_id:
            cmd.extend(["--resume", session_id])
        return cmd

    def _parse_stream_event(self, event: dict) -> dict | None:
        """Convert a raw Claude Code stream-json line to an SSE event dict.

        stream-json (--verbose) NDJSON format:
          {"type": "system", "subtype": "init", "session_id": "...", ...}
          {"type": "assistant", "message": {"content": [{"type": "text", "text": "..."}]}, "session_id": "..."}
          {"type": "tool", "tool_use_id": "...", "content": "..."}
          {"type": "result", "subtype": "success", "result": "...", "session_id": "...", ...}
          {"type": "rate_limit_event", ...}  (ignored)
        """
        event_type = event.get("type")

        if event_type == "assistant":
            message = event.get("message", {})
            for block in message.get("content", []):
                btype = block.get("type")
                if btype == "text":
                    text = block.get("text", "")
                    if text:
                        return {"type": "text", "text": text}
                elif btype == "tool_use":
                    return {
                        "type": "tool_use",
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "input": block.get("input", {}),
                    }

        elif event_type == "tool":
            return {
                "type": "tool_result",
                "tool_use_id": event.get("tool_use_id", ""),
                "content": str(event.get("content", ""))[:500],
            }

        elif event_type == "result":
            if event.get("is_error") or event.get("subtype") == "error_during_execution":
                errors = event.get("errors", [])
                msg = errors[0] if errors else "Agent error"
                return {"type": "error", "message": msg}
            return {
                "type": "done",
                "session_id": event.get("session_id"),
                "num_turns": event.get("num_turns", 0) or 0,
                "duration_ms": event.get("duration_ms", 0) or 0,
            }

        # system, rate_limit_event, etc. — silently ignored
        return None

    async def chat(self, message: str, today: date | None = None) -> ChatResult:
        """Run an interactive chat session (non-streaming).

        Resumes the day's trading session if one exists, giving the agent
        full context of today's trading activity and MCP tool access.

        Args:
            message: User's chat message.
            today: Date for session ID lookup (default: today).

        Returns:
            ChatResult with conversational text response and tool call log.
        """
        today = today or date.today()
        session_id = self._get_chat_session_id()
        self._write_mcp_config()
        self._ensure_session_state()
        await self._refresh_session_state_if_stale()

        cmd = self._build_chat_cmd(message, session_id)
        env = {**os.environ}

        results_path = self._workspace / "session-results.json"
        if results_path.exists():
            results_path.unlink()

        log.info("chat_invoke_start", session_id=session_id, tenant_id=self._tenant_id)

        try:
            result = await asyncio.to_thread(
                _run_with_kill,
                cmd,
                env,
                self._timeout,
                cwd=str(self._root_workspace),
            )

            if result.returncode != 0:
                log.error("chat_invoke_failed", returncode=result.returncode, stderr=result.stderr[:300])
                return ChatResult(error=f"Exit code {result.returncode}: {(result.stderr or '')[:200]}")

            # Extract content from the JSON wrapper {"result": "...", "session_id": ..., ...}
            content = result.stdout.strip()
            new_session_id: str | None = None
            num_turns = 0
            duration_ms = 0
            try:
                data = json.loads(result.stdout)
                content = data.get("result", content)
                new_session_id = data.get("session_id")
                num_turns = data.get("num_turns", 0) or 0
                duration_ms = data.get("duration_ms", 0) or 0
            except (json.JSONDecodeError, TypeError):
                pass

            if new_session_id:
                self._save_chat_session_id(new_session_id)

            accumulated = self._read_session_results(results_path)
            tool_calls = accumulated.get("tool_call_logs", [])

            log.info("chat_invoke_complete", num_turns=num_turns, tools_used=len(tool_calls))

            return ChatResult(
                content=content,
                session_id=new_session_id or session_id,
                tool_calls=tool_calls,
                num_turns=num_turns,
                duration_ms=duration_ms,
                accumulated=accumulated,
            )

        except subprocess.TimeoutExpired:
            log.error("chat_invoke_timeout", timeout=self._timeout)
            return ChatResult(error="Chat timed out")
        except Exception as e:
            log.error("chat_invoke_exception", error=str(e))
            return ChatResult(error=str(e))

    async def chat_stream(self, message: str, today: date | None = None) -> AsyncGenerator[dict, None]:
        """Stream a chat response as SSE event dicts (NDJSON from stream-json).

        Yields event dicts with type:
          {"type": "text", "text": "..."}
          {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
          {"type": "tool_result", "tool_use_id": "...", "content": "..."}
          {"type": "done", "session_id": "...", "num_turns": N, "duration_ms": N}
          {"type": "error", "message": "..."}

        Args:
            message: User's chat message.
            today: Date for session ID lookup (default: today).
        """
        today = today or date.today()
        session_id = self._get_chat_session_id()
        self._write_mcp_config()
        self._ensure_session_state()
        await self._refresh_session_state_if_stale()

        cmd = self._build_chat_stream_cmd(message, session_id)
        env = {**os.environ}

        results_path = self._workspace / "session-results.json"
        if results_path.exists():
            results_path.unlink()

        log.info("chat_stream_start", session_id=session_id, tenant_id=self._tenant_id)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(self._root_workspace),
            )
        except Exception as e:
            yield {"type": "error", "message": str(e)}
            return

        try:
            assert proc.stdout is not None  # PIPE guarantees this
            async for raw_line in proc.stdout:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                parsed = self._parse_stream_event(event)
                if parsed is None:
                    continue

                # Persist the new session_id when we see the "done" event
                if parsed.get("type") == "done" and parsed.get("session_id"):
                    self._save_chat_session_id(parsed["session_id"])

                # Clear stale session on resume errors so next message starts fresh
                if parsed.get("type") == "error" and "No conversation found" in parsed.get("message", ""):
                    self._clear_chat_session_id()

                yield parsed

        except Exception as e:
            log.error("chat_stream_error", error=str(e))
            yield {"type": "error", "message": str(e)}
        finally:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()

    def read_chat_accumulated(self) -> dict:
        """Read accumulated ActionState from the last chat session.

        Call after chat_stream() completes to get discovery_proposals,
        watchlist_updates, etc. written by MCP tools during the session.
        """
        results_path = self._workspace / "session-results.json"
        return self._read_session_results(results_path)

    # ── End chat methods ─────────────────────────────────────────────────────

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

        # Use raw_decode to find the first valid JSON object at each '{' position.
        # Unlike greedy/non-greedy regex, this handles nested objects correctly.
        decoder = json.JSONDecoder()
        for i, char in enumerate(text):
            if char == "{":
                try:
                    obj, _ = decoder.raw_decode(text, i)
                    if isinstance(obj, dict):
                        return obj
                except (json.JSONDecodeError, ValueError):
                    continue

        # Fallback: return text as reasoning
        return {"reasoning": text[:1000], "trades": []}

    def _extract_cli_metadata(self, stdout: str) -> dict:
        """Extract metadata (turns, duration, session_id) from Claude Code JSON wrapper."""
        try:
            data = json.loads(stdout)
            return {
                "session_id": data.get("session_id"),
                "num_turns": data.get("num_turns", 0) or 0,
                "duration_ms": data.get("duration_ms", 0) or 0,
            }
        except (json.JSONDecodeError, TypeError):
            return {}

    def _read_session_results(self, results_path: Path, retries: int = 6, delay: float = 0.5) -> dict:
        """Read accumulated ActionState written by MCP server on exit.

        Retries to handle race between MCP server grandchild flush and invoker read.
        Total max wait: retries * delay (default 3s).
        """
        for attempt in range(retries):
            if results_path.exists():
                try:
                    data = json.loads(results_path.read_text())
                    results_path.unlink()
                    return data
                except (json.JSONDecodeError, OSError):
                    pass
            if attempt < retries - 1:
                time.sleep(delay)

        log.warning("session_results_not_found", path=str(results_path), retries=retries)
        return {}


# ── Lightweight Claude CLI utility ───────────────────────────────────────────


async def claude_cli_call(
    prompt: str,
    system: str | None = None,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 1024,
    timeout: int = 120,
) -> str:
    """Simple text-in/text-out Claude CLI call (no MCP, no session).

    Replaces direct Anthropic SDK calls (anthropic.Anthropic().messages.create)
    with Claude Code CLI using the Max subscription.

    Args:
        prompt: User message to send.
        system: Optional system prompt (prepended to user prompt).
        model: Claude model to use.
        max_tokens: Maximum response tokens.
        timeout: Subprocess timeout in seconds.

    Returns:
        Response text, or empty string on failure.
    """
    full_prompt = prompt
    if system:
        full_prompt = f"<system>{system}</system>\n\n{prompt}"

    cmd = [
        "claude",
        "-p",
        full_prompt,
        "--output-format",
        "text",
        "--max-turns",
        "1",
        "--model",
        model,
    ]

    env = {**os.environ}

    try:
        result = await asyncio.to_thread(
            _run_with_kill,
            cmd,
            env,
            timeout,
        )
        if result.returncode != 0:
            log.error("claude_cli_call_failed", returncode=result.returncode, stderr=result.stderr[:200])
            return ""
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        log.error("claude_cli_call_timeout", timeout=timeout)
        return ""
    except Exception as e:
        log.error("claude_cli_call_error", error=str(e))
        return ""


async def claude_cli_json(
    prompt: str,
    system: str | None = None,
    model: str = "claude-sonnet-4-6",
    timeout: int = 120,
) -> dict:
    """Claude CLI call that expects a JSON response.

    Note: We intentionally do NOT use --json-schema enforcement. CLAUDE.md instructions
    + fallback parsing is more resilient than coupling to a schema flag, especially
    when Claude Code wraps output in {"result": "...", "session_id": "..."}.

    Args:
        prompt: User message (should instruct JSON output).
        system: Optional system prompt.
        model: Claude model to use.
        timeout: Subprocess timeout in seconds.

    Returns:
        Parsed dict, or empty dict on failure.
    """
    full_prompt = prompt
    if system:
        full_prompt = f"<system>{system}</system>\n\n{prompt}"

    cmd = [
        "claude",
        "-p",
        full_prompt,
        "--output-format",
        "json",
        "--max-turns",
        "1",
        "--model",
        model,
    ]

    env = {**os.environ}

    try:
        result = await asyncio.to_thread(
            _run_with_kill,
            cmd,
            env,
            timeout,
        )
        if result.returncode != 0:
            log.error("claude_cli_json_failed", returncode=result.returncode)
            return {}

        data = json.loads(result.stdout)
        # Claude Code wraps in {"result": "...", "session_id": "..."}
        if "result" in data and isinstance(data["result"], str):
            text = data["result"]
        else:
            return data

        # Parse JSON from result text
        fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        if fence:
            return json.loads(fence.group(1))
        decoder = json.JSONDecoder()
        for i, char in enumerate(text):
            if char == "{":
                try:
                    obj, _ = decoder.raw_decode(text, i)
                    if isinstance(obj, dict):
                        return obj
                except (json.JSONDecodeError, ValueError):
                    continue
        return {}
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        log.error("claude_cli_json_error", error=str(e))
        return {}
