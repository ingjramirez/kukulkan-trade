"""Context management for the persistent agentic agent.

Builds the messages array for each Claude API call from conversation history,
compressed summaries, and trigger context.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger()

# ── Identity and framework prompt ─────────────────────────────────────────────

_SYSTEM_IDENTITY = """You are the AI portfolio manager for Kukulkan Trade — an educational \
trading bot managing Portfolio B on Alpaca paper trading.

You have access to tools for querying portfolio state, market data, news, and executing \
trades. You operate with **persistent memory** — you remember prior sessions and can \
reference past analysis, theses, and decisions.

## Session Context
You are triggered 3x daily (morning open, midday check, pre-close) plus event-driven \
triggers. Between sessions, your conversation history is stored and loaded. You can build \
multi-day theses, track position entries, and learn from past outcomes.

## Decision Framework
1. Assess the current market regime and any changes since last session
2. Review your active positions and theses
3. Use tools to investigate anything that needs attention
4. Make trades only when you have conviction — holding cash is a valid decision
5. Document your reasoning for future reference

## Guardrails (enforced by the system — you cannot override these)
- All trades pass through RiskManager (position limits, sector concentration, circuit breakers)
- Trades >10% of portfolio require human approval via Telegram
- Inverse ETF BUYs (SH, PSQ, RWM) require CORRECTION/CRISIS regime + defensive/crisis posture + Telegram approval
- Budget limits per session/day/month are enforced at the system level
- You can only trade stocks, ETFs, inverse ETFs, and crypto proxies — no options, futures, or margin

## Response Format
After investigation, respond with a JSON object:
```json
{
  "regime_assessment": "Current market regime and changes",
  "reasoning": "Your analysis and rationale",
  "trades": [{"ticker": "NVDA", "side": "BUY", "shares": 50, "conviction": "high", "reason": "..."}],
  "risk_notes": "Any risk concerns or watchlist updates",
  "theses_update": "Updates to multi-day theses (new, confirmed, invalidated)"
}
```
If no trades, return empty trades array with reasoning for holding."""


class ContextManager:
    """Builds the messages array for each Claude API call from conversation history."""

    # Context budget targets (tokens, approximate)
    SYSTEM_PROMPT_BUDGET = 5_000
    PINNED_CONTEXT_BUDGET = 3_000
    SUMMARY_BUDGET = 12_000  # ~25 sessions x 500 tokens
    RECENT_HISTORY_BUDGET = 25_000  # ~5 sessions in full
    COMPRESSION_THRESHOLD = 150_000  # Compress when total exceeds this

    def build_system_prompt(
        self,
        pinned_context: str,
        strategy_directive: str = "",
    ) -> str:
        """Build the system prompt with identity, strategy framework, guardrails,
        and pinned context (active theses, key learnings, risk posture).

        The system prompt is STABLE across turns within a session → cache-friendly.
        Pinned context changes daily at most.
        """
        parts = [_SYSTEM_IDENTITY]

        if strategy_directive:
            parts.append(f"\n## Strategy Directive\n{strategy_directive}")

        if pinned_context:
            parts.append(f"\n{pinned_context}")

        return "\n".join(parts)

    def build_cached_system_prompt(
        self,
        pinned_context: str,
        strategy_directive: str = "",
    ) -> list[dict]:
        """Build system prompt with cache_control markers for Anthropic prompt caching.

        Structure (stable → volatile):
        1. Identity block (never changes) — cache_control: ephemeral
        2. Strategy directive (changes rarely)
        3. Pinned context (changes daily at most) — cache_control: ephemeral

        Returns:
            List of content block dicts with cache_control markers.
        """
        blocks: list[dict] = []

        # Identity block — most stable, always cached
        blocks.append(
            {
                "type": "text",
                "text": _SYSTEM_IDENTITY,
                "cache_control": {"type": "ephemeral"},
            }
        )

        # Strategy directive — changes rarely
        if strategy_directive:
            blocks.append({"type": "text", "text": f"\n## Strategy Directive\n{strategy_directive}"})

        # Pinned context — changes daily at most
        if pinned_context:
            blocks.append(
                {
                    "type": "text",
                    "text": f"\n{pinned_context}",
                    "cache_control": {"type": "ephemeral"},
                }
            )

        return blocks

    def build_messages(
        self,
        summaries: list[dict],
        recent_sessions: list[dict],
        trigger_message: str,
    ) -> list[dict]:
        """Build the Anthropic messages array.

        Structure:
        1. (Optional) Compressed history preamble as a user message
        2. Recent session messages replayed as user/assistant pairs
        3. New trigger message

        Args:
            summaries: Compressed older sessions from ConversationStore.load_summaries()
            recent_sessions: Full message history from ConversationStore.load_recent()
            trigger_message: The new trigger message for this session

        Returns:
            List of message dicts in Anthropic format.
        """
        messages: list[dict] = []

        # 1. Inject compressed history as a preamble
        if summaries:
            summary_text = self._format_summaries(summaries)
            messages.append({"role": "user", "content": summary_text})
            messages.append(
                {
                    "role": "assistant",
                    "content": "Understood. I have context from my prior trading sessions. "
                    "What's the current situation?",
                }
            )

        # 2. Replay recent sessions as the actual conversation
        for session in recent_sessions:
            for msg in session["messages"]:
                messages.append(msg)

        # 3. New trigger message
        messages.append({"role": "user", "content": trigger_message})

        return messages

    def build_trigger_message(
        self,
        trigger_type: str,
        market_data: dict | None = None,
        portfolio_summary: dict | None = None,
    ) -> str:
        """Build the trigger message for this session.

        Args:
            trigger_type: morning/midday/close/event/weekly_review
            market_data: Dict with regime, vix, spy data, etc.
            portfolio_summary: Dict with positions, cash, P&L, etc.

        Returns:
            Formatted trigger message string.
        """
        market = market_data or {}
        portfolio = portfolio_summary or {}

        if trigger_type == "morning":
            return self._build_morning_trigger(market, portfolio)
        elif trigger_type == "midday":
            return self._build_midday_trigger(market, portfolio)
        elif trigger_type == "close":
            return self._build_close_trigger(market, portfolio)
        elif trigger_type == "event":
            return self._build_event_trigger(market, portfolio)
        elif trigger_type == "weekly_review":
            return self._build_weekly_review_trigger(market, portfolio)
        else:
            return f"Session trigger: {trigger_type}. Market data: {market}. Portfolio: {portfolio}."

    def estimate_tokens(self, messages: list[dict]) -> int:
        """Estimate token count for a messages array.

        Uses ~4 chars per token heuristic for estimation.
        Precise counting not needed — this is for budget decisions.
        """
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                # Tool use/result blocks
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text", "") or block.get("content", "")
                        if isinstance(text, str):
                            total_chars += len(text)
                        # Tool input/output
                        inp = block.get("input")
                        if isinstance(inp, dict):
                            total_chars += len(str(inp))
        return total_chars // 4

    def should_compress(self, total_tokens: int) -> bool:
        """Return True if total context exceeds COMPRESSION_THRESHOLD."""
        return total_tokens > self.COMPRESSION_THRESHOLD

    def build_pinned_context(
        self,
        active_theses: list[dict] | None = None,
        key_learnings: list[str] | None = None,
        current_posture: str = "Balanced",
        track_record_summary: str = "",
    ) -> str:
        """Build the pinned context section (~3K tokens).

        This is always present in the system prompt.
        Updated at most daily (weekly for learnings).
        """
        parts = []

        # Active theses
        theses = active_theses or []
        if theses:
            parts.append("## Your Active Theses")
            for thesis in theses:
                ticker = thesis.get("ticker", "?")
                description = thesis.get("description", "")
                entered = thesis.get("entered", "")
                entry_str = f" [Entered {entered}]" if entered else ""
                parts.append(f"- {ticker}: {description}{entry_str}")
        else:
            parts.append("## Your Active Theses\nNo active theses. You may open new positions if conditions warrant.")

        # Key learnings
        learnings = key_learnings or []
        if learnings:
            parts.append("\n## Key Learnings (from weekly reviews)")
            for learning in learnings:
                parts.append(f"- {learning}")

        # Current posture
        parts.append(f"\n## Current Posture: {current_posture}")

        # Track record
        if track_record_summary:
            parts.append(f"\n## Track Record (last 30 days)\n{track_record_summary}")

        return "\n".join(parts)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _format_summaries(self, summaries: list[dict]) -> str:
        """Format compressed summaries into a single context message."""
        lines = ["Here is a summary of your recent trading sessions:\n"]
        for s in summaries:
            created = s.get("created_at", "")
            if hasattr(created, "strftime"):
                date_str = created.strftime("%b %d %H:%M")
            else:
                date_str = str(created)
            trigger = s.get("trigger_type", "session")
            summary = s.get("summary", "")
            lines.append(f"[{date_str} {trigger.capitalize()}]: {summary}")
        return "\n".join(lines)

    def _build_morning_trigger(self, market: dict, portfolio: dict) -> str:
        regime = market.get("regime", "unknown")
        vix = market.get("vix", "N/A")
        spy_change = market.get("spy_change_pct", "N/A")
        cash = portfolio.get("cash", "N/A")
        positions_count = portfolio.get("positions_count", 0)
        total_value = portfolio.get("total_value", "N/A")

        parts = [
            "Good morning. Markets are open.",
            f"Regime: {regime}. VIX: {vix}. SPY overnight: {spy_change}%.",
            f"Portfolio B: ${total_value} total, ${cash} cash, {positions_count} positions.",
            "Review your positions and the market. Make trades if warranted.",
        ]
        if market.get("overnight_summary"):
            parts.insert(2, f"Overnight: {market['overnight_summary']}")
        return "\n".join(parts)

    def _build_midday_trigger(self, market: dict, portfolio: dict) -> str:
        vix = market.get("vix", "N/A")
        cash = portfolio.get("cash", "N/A")
        total_value = portfolio.get("total_value", "N/A")

        parts = [
            "Midday update.",
            f"VIX: {vix}. Portfolio B: ${total_value}, ${cash} cash.",
            "Check for any significant moves since morning. Act if needed.",
        ]
        if market.get("alerts"):
            parts.insert(1, f"Alerts: {market['alerts']}")
        return "\n".join(parts)

    def _build_close_trigger(self, market: dict, portfolio: dict) -> str:
        vix = market.get("vix", "N/A")
        cash = portfolio.get("cash", "N/A")
        total_value = portfolio.get("total_value", "N/A")

        return "\n".join(
            [
                "Markets closing in 15 minutes.",
                f"VIX: {vix}. Portfolio B: ${total_value}, ${cash} cash.",
                "Review overnight risk. Close any positions that shouldn't be held overnight.",
                "Summarize the day and any plans for tomorrow.",
            ]
        )

    def _build_event_trigger(self, market: dict, portfolio: dict) -> str:
        event_type = market.get("event_type", "unknown")
        event_detail = market.get("event_detail", "No details available.")
        return f"ALERT: {event_type}\n{event_detail}"

    def _build_weekly_review_trigger(self, market: dict, portfolio: dict) -> str:
        outcomes = market.get("outcomes_summary", "No outcomes this week.")
        track_record = market.get("track_record", "")
        parts = [
            "Weekly review time.",
            f"This week's outcomes:\n{outcomes}",
        ]
        if track_record:
            parts.append(f"Track record:\n{track_record}")
        parts.append(
            "Review your performance. Update your learnings and theses. What should you do differently next week?"
        )
        return "\n".join(parts)
