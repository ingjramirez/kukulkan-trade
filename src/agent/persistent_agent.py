"""Persistent agentic agent for Portfolio B.

Wraps existing AgentRunner/ToolRegistry/AgentMemoryManager with
conversation persistence: sessions are stored in SQLite, old sessions
compressed to summaries, and full history replayed on each trigger.

Three-level fallback:
  use_persistent_agent=True  → this module
  use_agent_loop=True        → Phase 32 agentic loop (agent_runner.py)
  else                       → original mega-prompt (claude_agent.py)
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field

import structlog

from src.agent.context_manager import ContextManager
from src.agent.conversation_store import ConversationStore
from src.agent.session_compressor import CompressionError, SessionCompressor
from src.agent.token_tracker import TokenTracker
from src.storage.database import Database

log = structlog.get_logger()


@dataclass
class _TieredAgentResult:
    """Adapter that normalizes TieredRunResult to the AgentRunResult interface."""

    response: dict
    tool_calls: list
    turns: int
    token_tracker: TokenTracker
    raw_messages: list


@dataclass
class PersistentRunResult:
    """Result from a persistent agent session."""

    response: dict
    session_id: str
    tool_calls: list = field(default_factory=list)
    turns: int = 0
    token_tracker: TokenTracker = field(default_factory=TokenTracker)
    tool_summary: dict | None = None
    compressed_count: int = 0


class PersistentAgent:
    """Persistence layer around AgentRunner for Portfolio B.

    Lifecycle per trigger:
    1. Load conversation history (summaries + recent sessions)
    2. Build context-aware system prompt + messages array
    3. Delegate to AgentRunner (tool-use loop stays unchanged)
    4. Save session to SQLite
    5. Compress old sessions if needed
    """

    def __init__(
        self,
        db: Database,
        api_key: str,
        tenant_id: str = "default",
    ) -> None:
        self._db = db
        self._api_key = api_key
        self._tenant_id = tenant_id
        self._store = ConversationStore(db)
        self._context = ContextManager()
        self._compressor = SessionCompressor(api_key=api_key)

    async def run_session(
        self,
        trigger_type: str,
        market_data: dict,
        portfolio_summary: dict,
        runner_kwargs: dict,
        pinned_context: str = "",
        strategy_directive: str = "",
    ) -> PersistentRunResult:
        """Execute a full persistent agent session.

        Args:
            trigger_type: morning/midday/close/event/weekly_review
            market_data: Dict with regime, vix, spy data for trigger message.
            portfolio_summary: Dict with positions, cash, P&L for trigger.
            runner_kwargs: Dict passed to AgentRunner.run() — must include
                'runner' (AgentRunner instance) and 'system_prompt' (base).
            pinned_context: Active theses, learnings, posture for system prompt.
            strategy_directive: Strategy mode text for system prompt.

        Returns:
            PersistentRunResult with response, session metadata, and compression count.
        """
        session_id = f"{self._tenant_id}-{trigger_type}-{uuid.uuid4().hex[:8]}"

        # Mark session started (crash detection)
        await self._store.mark_session_started(self._tenant_id, session_id, trigger_type)
        log.info("persistent_session_started", session_id=session_id, trigger=trigger_type)

        try:
            async with asyncio.timeout(300):  # 5 minute max per session
                result = await self._execute_session(
                    session_id=session_id,
                    trigger_type=trigger_type,
                    market_data=market_data,
                    portfolio_summary=portfolio_summary,
                    runner_kwargs=runner_kwargs,
                    pinned_context=pinned_context,
                    strategy_directive=strategy_directive,
                )
            return result
        except TimeoutError:
            log.error("persistent_session_timeout", session_id=session_id)
            raise
        except Exception:
            log.exception("persistent_session_failed", session_id=session_id)
            raise

    async def _execute_session(
        self,
        session_id: str,
        trigger_type: str,
        market_data: dict,
        portfolio_summary: dict,
        runner_kwargs: dict,
        pinned_context: str,
        strategy_directive: str,
    ) -> PersistentRunResult:
        """Core session execution (separated for testability)."""
        from src.agent.agent_runner import AgentRunner

        runner: AgentRunner = runner_kwargs["runner"]
        base_system_prompt: str = runner_kwargs.get("system_prompt", "")

        # 1. Build persistent system prompt
        system_prompt = self._context.build_system_prompt(
            pinned_context=pinned_context,
            strategy_directive=strategy_directive,
        )
        # Append the base prompt sections (performance, memory, etc.)
        if base_system_prompt:
            system_prompt = f"{system_prompt}\n\n{base_system_prompt}"

        # 2. Load conversation history
        summaries = await self._store.load_summaries(self._tenant_id)
        recent_sessions = await self._store.load_recent(self._tenant_id)

        # 3. Build trigger message
        trigger_message = self._context.build_trigger_message(
            trigger_type=trigger_type,
            market_data=market_data,
            portfolio_summary=portfolio_summary,
        )

        # 4. Build messages array with history
        messages = self._context.build_messages(summaries, recent_sessions, trigger_message)

        # 5. Run agent loop with conversation context (or tiered runner)
        tiered_runner = runner_kwargs.get("tiered_runner")
        if tiered_runner:
            from src.agent.tiered_runner import TieredRunResult

            tiered_result: TieredRunResult = await tiered_runner.run(
                system_prompt=runner_kwargs.get("cached_system_prompt", system_prompt),
                user_message=trigger_message,
                session_profile=runner_kwargs["session_profile"],
                market_data=runner_kwargs.get("market_data", {}),
                portfolio_summary=runner_kwargs.get("portfolio_summary", {}),
                posture=runner_kwargs.get("posture", "balanced"),
                messages_override=messages,
            )
            # Convert TieredRunResult to AgentRunResult-compatible interface
            result = _TieredAgentResult(
                response=tiered_result.response,
                tool_calls=tiered_result.tool_calls,
                turns=tiered_result.turns,
                token_tracker=tiered_result.token_tracker,
                raw_messages=tiered_result.raw_messages,
            )
        else:
            result = await runner.run(
                system_prompt=system_prompt,
                user_message=trigger_message,
                messages_override=messages,
            )

        # 6. Build full messages for storage (history + agent's new messages)
        # The agent's raw_messages starts from where we injected (our messages + agent turns)
        full_messages = result.raw_messages

        # 7. Save completed session
        total_tokens = result.token_tracker.total_input_tokens + result.token_tracker.total_output_tokens
        cost = result.token_tracker.total_cost_usd
        await self._store.save_session(
            tenant_id=self._tenant_id,
            session_id=session_id,
            trigger_type=trigger_type,
            messages=full_messages,
            token_count=total_tokens,
            cost_usd=round(cost, 4),
        )
        log.info(
            "persistent_session_saved",
            session_id=session_id,
            tokens=total_tokens,
            cost_usd=round(cost, 4),
        )

        # 8. Compress old sessions if needed
        compressed_count = await self._compress_old_sessions()

        # Build tool summary
        tool_summary = None
        if result.tool_calls:
            tool_summary = {
                "tools_used": len(result.tool_calls),
                "turns": result.turns,
                "cost_usd": round(cost, 4),
            }

        return PersistentRunResult(
            response=result.response,
            session_id=session_id,
            tool_calls=result.tool_calls,
            turns=result.turns,
            token_tracker=result.token_tracker,
            tool_summary=tool_summary,
            compressed_count=compressed_count,
        )

    async def _compress_old_sessions(self) -> int:
        """Compress sessions that are older than the recent window.

        Returns the number of sessions compressed.
        """
        candidates = await self._store.get_uncompressed_sessions(self._tenant_id)
        if not candidates:
            return 0

        compressed = 0
        for session in candidates:
            try:
                summary = await self._compressor.compress(session["messages"])
                await self._store.save_summary(session["session_id"], summary)
                compressed += 1
                log.info(
                    "session_compressed",
                    session_id=session["session_id"],
                    summary_len=len(summary),
                )
            except CompressionError as e:
                log.warning(
                    "session_compression_failed",
                    session_id=session["session_id"],
                    error=str(e),
                )
            except Exception as e:
                log.warning(
                    "session_compression_error",
                    session_id=session["session_id"],
                    error=str(e),
                )

        return compressed
