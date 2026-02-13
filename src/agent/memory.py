"""Agent memory manager for Portfolio B.

Provides tiered memory (short-term, weekly summaries, agent notes)
that fits within ~1,600 tokens of context per agent call.
"""

from datetime import datetime, timedelta

import structlog

from src.storage.database import Database

log = structlog.get_logger()

MAX_SHORT_TERM = 3
MAX_WEEKLY_SUMMARIES = 4
MAX_AGENT_NOTES = 10


class AgentMemoryManager:
    """Manages the 3-tier memory system for Portfolio B's Claude agent."""

    def build_memory_prompt(self, memories: dict) -> str:
        """Format all 3 memory tiers into a single text block for the system prompt.

        Args:
            memories: Dict from db.get_all_agent_memory_context() with keys
                      short_term, weekly_summary, agent_note.

        Returns:
            Formatted memory text block, or empty string if no memories.
        """
        sections: list[str] = []

        # Short-term: last 3 decisions
        short_term = memories.get("short_term", [])
        if short_term:
            items = [f"- [{m.key}] {m.content}" for m in short_term[-MAX_SHORT_TERM:]]
            sections.append("### Recent Decisions\n" + "\n".join(items))

        # Long-term: weekly summaries
        weekly = memories.get("weekly_summary", [])
        if weekly:
            items = [f"- [{m.key}] {m.content}" for m in weekly[-MAX_WEEKLY_SUMMARIES:]]
            sections.append("### Weekly Lessons\n" + "\n".join(items))

        # Agent notes: persistent observations
        notes = memories.get("agent_note", [])
        if notes:
            items = [f"- [{m.key}] {m.content}" for m in notes[-MAX_AGENT_NOTES:]]
            sections.append("### Your Notes\n" + "\n".join(items))

        if not sections:
            return ""

        return "## Memory\n" + "\n\n".join(sections)

    async def save_short_term(
        self,
        db: Database,
        analysis_date: str,
        response: dict,
        tenant_id: str = "default",
    ) -> None:
        """Extract and save a short-term memory from an agent response.

        Keeps only the last MAX_SHORT_TERM entries by pruning oldest.

        Args:
            db: Database instance.
            analysis_date: ISO date string used as the memory key.
            response: Parsed agent response dict.
        """
        regime = response.get("regime_assessment", "")
        reasoning = response.get("reasoning", "")
        trades = response.get("trades", [])

        # Build compact summary
        trade_summary = (
            ", ".join(f"{t.get('side', '?')} {t.get('ticker', '?')} @{t.get('weight', 0):.0%}" for t in trades[:5])
            if trades
            else "no trades"
        )

        content = f"{regime} | {reasoning[:150]} | Trades: {trade_summary}"

        await db.upsert_agent_memory(
            category="short_term",
            key=analysis_date,
            content=content,
            tenant_id=tenant_id,
        )

        # Prune to keep only the last MAX_SHORT_TERM
        all_short = await db.get_agent_memories("short_term", tenant_id=tenant_id)
        if len(all_short) > MAX_SHORT_TERM:
            to_delete = all_short[: len(all_short) - MAX_SHORT_TERM]
            async with db.session() as s:
                for row in to_delete:
                    existing = await s.get(type(row), row.id)
                    if existing:
                        await s.delete(existing)
                await s.commit()

        log.info("short_term_memory_saved", date=analysis_date)

    async def save_agent_notes(
        self,
        db: Database,
        notes: list[dict],
        tenant_id: str = "default",
    ) -> None:
        """Parse and save agent notes from the response.

        Each note has a "key" and "content". Upserts by key.
        Enforces MAX_AGENT_NOTES limit by removing oldest notes.

        Args:
            db: Database instance.
            notes: List of dicts with "key" and "content".
        """
        if not notes:
            return

        for note in notes:
            key = note.get("key", "")
            content = note.get("content", "")
            if not key or not content:
                continue
            # Truncate key to 100 chars and content to 200 chars
            key = key[:100]
            content = content[:200]
            await db.upsert_agent_memory(
                category="agent_note",
                key=key,
                content=content,
                tenant_id=tenant_id,
            )

        # Enforce max notes limit
        all_notes = await db.get_agent_memories("agent_note", tenant_id=tenant_id)
        if len(all_notes) > MAX_AGENT_NOTES:
            to_delete = all_notes[: len(all_notes) - MAX_AGENT_NOTES]
            async with db.session() as s:
                for row in to_delete:
                    existing = await s.get(type(row), row.id)
                    if existing:
                        await s.delete(existing)
                await s.commit()

        log.info("agent_notes_saved", count=len(notes))

    async def run_weekly_compaction(
        self,
        db: Database,
        agent,
        tenant_id: str = "default",
        outcome_summary: str | None = None,
        track_record_text: str | None = None,
    ) -> None:
        """Evaluate the past week's trading decisions and compress into a summary.

        Fetches recent agent decisions, asks Claude to evaluate them with
        outcome data, and stores as a weekly_summary memory.

        Args:
            db: Database instance.
            agent: ClaudeAgent instance for the evaluation call.
            tenant_id: Tenant UUID.
            outcome_summary: Optional trade outcome feedback to include.
            track_record_text: Optional track record stats text.
        """
        # Fetch last 7 days of short-term memories
        short_term = await db.get_agent_memories("short_term", tenant_id=tenant_id)
        if not short_term:
            log.info("weekly_compaction_skipped_no_data")
            return

        # Build the week key
        now = datetime.utcnow()
        week_key = f"week_{now.strftime('%Y-%W')}"

        # Combine short-term memories into context
        decisions_text = "\n".join(f"- [{m.key}] {m.content}" for m in short_term)

        outcome_section = ""
        if outcome_summary:
            outcome_section = f"\n\nTrade Outcomes (actual P&L):\n{outcome_summary}\n"

        track_record_section = ""
        if track_record_text:
            track_record_section = f"\n\nTrack Record:\n{track_record_text}\n"

        prompt = f"""Evaluate your trading decisions from the past week.

Decisions:
{decisions_text}{outcome_section}{track_record_section}

Answer these questions in a ~200-token summary:
1. Which decisions worked? Why?
2. Which didn't? What went wrong?
3. Patterns in your track record (regime, session, sector)?
4. What should change going forward?

Write a concise evaluation paragraph (no headers, no bullets):"""

        try:
            response = agent.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                system="You are a trading journal summarizer. Be concise and specific.",
                messages=[{"role": "user", "content": prompt}],
            )
            summary = response.content[0].text.strip()
        except Exception as e:
            log.error("weekly_compaction_failed", error=str(e))
            return

        await db.upsert_agent_memory(
            category="weekly_summary",
            key=week_key,
            content=summary,
            expires_at=now + timedelta(weeks=5),
            tenant_id=tenant_id,
        )

        # Prune to keep only last MAX_WEEKLY_SUMMARIES
        all_weekly = await db.get_agent_memories("weekly_summary", tenant_id=tenant_id)
        if len(all_weekly) > MAX_WEEKLY_SUMMARIES:
            to_delete = all_weekly[: len(all_weekly) - MAX_WEEKLY_SUMMARIES]
            async with db.session() as s:
                for row in to_delete:
                    existing = await s.get(type(row), row.id)
                    if existing:
                        await s.delete(existing)
                await s.commit()

        log.info("weekly_compaction_complete", week=week_key)
