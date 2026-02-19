"""SQLite-backed conversation persistence for the persistent agentic agent."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy import select, update

from src.storage.database import Database
from src.storage.models import AgentConversationRow

log = structlog.get_logger()


class ConversationStore:
    """Stores and retrieves agent conversation sessions from SQLite.

    Each session represents one scheduler trigger (morning/midday/close/event).
    Messages are stored as JSON arrays matching the Anthropic messages format.
    Old sessions can be compressed to summaries for context management.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    async def save_session(
        self,
        tenant_id: str,
        session_id: str,
        trigger_type: str,
        messages: list[dict],
        token_count: int,
        cost_usd: float,
    ) -> None:
        """Save a completed session with its full message history.

        If a session with this session_id was previously marked as 'started',
        this updates it to 'completed' and fills in the messages.
        """
        messages_json = json.dumps(messages, default=str)
        try:
            async with self.db.session() as s:
                # Check if session was previously marked as started
                existing = (
                    await s.execute(
                        select(AgentConversationRow).where(
                            AgentConversationRow.session_id == session_id,
                        )
                    )
                ).scalar_one_or_none()

                if existing:
                    existing.messages_json = messages_json
                    existing.token_count = token_count
                    existing.cost_usd = cost_usd
                    existing.session_status = "completed"
                else:
                    s.add(
                        AgentConversationRow(
                            tenant_id=tenant_id,
                            session_id=session_id,
                            trigger_type=trigger_type,
                            messages_json=messages_json,
                            token_count=token_count,
                            cost_usd=cost_usd,
                            session_status="completed",
                        )
                    )
                await s.commit()
        except Exception as e:
            log.error("conversation_save_session_failed", session_id=session_id, error=str(e))
            raise

    async def mark_session_started(
        self,
        tenant_id: str,
        session_id: str,
        trigger_type: str,
    ) -> None:
        """Mark a session as started (for crash detection).

        Creates a row with status='started' and empty messages.
        If the session completes, save_session() will update this row.
        If the process crashes, this row remains with status='started'.
        """
        try:
            async with self.db.session() as s:
                s.add(
                    AgentConversationRow(
                        tenant_id=tenant_id,
                        session_id=session_id,
                        trigger_type=trigger_type,
                        messages_json="[]",
                        token_count=0,
                        cost_usd=0.0,
                        session_status="started",
                    )
                )
                await s.commit()
        except Exception as e:
            log.error("conversation_mark_started_failed", session_id=session_id, error=str(e))
            raise

    async def mark_session_failed(
        self,
        session_id: str,
        token_count: int,
        cost_usd: float,
    ) -> None:
        """Update a 'started' session to 'failed', preserving partial token spend.

        Called from error handlers so the conversation row reflects actual API cost
        even when the session crashes (e.g. 429 rate-limit on turn 3).
        """
        try:
            async with self.db.session() as s:
                await s.execute(
                    update(AgentConversationRow)
                    .where(AgentConversationRow.session_id == session_id)
                    .values(
                        session_status="failed",
                        token_count=token_count,
                        cost_usd=cost_usd,
                    )
                )
                await s.commit()
            log.info(
                "conversation_session_marked_failed",
                session_id=session_id,
                token_count=token_count,
                cost_usd=cost_usd,
            )
        except Exception as e:
            log.error("conversation_mark_failed_error", session_id=session_id, error=str(e))

    async def load_recent(
        self,
        tenant_id: str,
        n: int = 5,
    ) -> list[dict]:
        """Load last N completed sessions' full message history.

        Returns list of dicts:
        [{"session_id", "trigger_type", "messages": [...], "created_at"}, ...]

        Skips crashed sessions (status='started') and sessions where
        messages_json has been cleaned up (empty '[]').
        n is capped at 50 to prevent unbounded result sets.
        """
        n = min(n, 50)
        async with self.db.session() as s:
            result = await s.execute(
                select(AgentConversationRow)
                .where(
                    AgentConversationRow.tenant_id == tenant_id,
                    AgentConversationRow.session_status == "completed",
                )
                .order_by(AgentConversationRow.created_at.desc())
                .limit(n)
            )
            rows = list(result.scalars().all())

        # Return in chronological order (oldest first)
        sessions = []
        for row in reversed(rows):
            messages = json.loads(row.messages_json)
            # Skip sessions where messages have been cleaned up
            if not messages:
                continue
            sessions.append(
                {
                    "session_id": row.session_id,
                    "trigger_type": row.trigger_type,
                    "messages": messages,
                    "created_at": row.created_at,
                }
            )
        return sessions

    async def load_summaries(
        self,
        tenant_id: str,
        n: int = 25,
    ) -> list[dict]:
        """Load compressed summaries for older sessions.

        Returns only sessions that have a non-null summary.
        Ordered chronologically (oldest first).
        """
        async with self.db.session() as s:
            result = await s.execute(
                select(AgentConversationRow)
                .where(
                    AgentConversationRow.tenant_id == tenant_id,
                    AgentConversationRow.summary.isnot(None),
                    AgentConversationRow.session_status == "completed",
                )
                .order_by(AgentConversationRow.created_at.desc())
                .limit(n)
            )
            rows = list(result.scalars().all())

        return [
            {
                "session_id": row.session_id,
                "trigger_type": row.trigger_type,
                "summary": row.summary,
                "created_at": row.created_at,
            }
            for row in reversed(rows)
        ]

    async def get_uncompressed_sessions(
        self,
        tenant_id: str,
        keep_recent: int = 5,
    ) -> list[dict]:
        """Get completed sessions older than the last N that still have no summary.

        These are candidates for compression.
        """
        async with self.db.session() as s:
            # First, get the Nth most recent session's created_at as cutoff
            recent_result = await s.execute(
                select(AgentConversationRow.created_at)
                .where(
                    AgentConversationRow.tenant_id == tenant_id,
                    AgentConversationRow.session_status == "completed",
                )
                .order_by(AgentConversationRow.created_at.desc())
                .limit(keep_recent)
            )
            recent_dates = [row[0] for row in recent_result.all()]
            if len(recent_dates) < keep_recent:
                return []  # Not enough sessions to compress anything

            cutoff = recent_dates[-1]

            # Get uncompressed sessions older than the cutoff
            result = await s.execute(
                select(AgentConversationRow)
                .where(
                    AgentConversationRow.tenant_id == tenant_id,
                    AgentConversationRow.session_status == "completed",
                    AgentConversationRow.summary.is_(None),
                    AgentConversationRow.created_at < cutoff,
                )
                .order_by(AgentConversationRow.created_at)
            )
            rows = list(result.scalars().all())

        return [
            {
                "session_id": row.session_id,
                "trigger_type": row.trigger_type,
                "messages": json.loads(row.messages_json),
                "created_at": row.created_at,
            }
            for row in rows
            if row.messages_json and row.messages_json != "[]"
        ]

    async def save_summary(
        self,
        session_id: str,
        summary: str,
    ) -> None:
        """Save a compressed summary for a session.

        The messages_json stays in DB for debugging (cleaned up later by
        cleanup_old_messages). The summary is used for context building.
        """
        try:
            async with self.db.session() as s:
                await s.execute(
                    update(AgentConversationRow)
                    .where(AgentConversationRow.session_id == session_id)
                    .values(summary=summary)
                )
                await s.commit()
        except Exception as e:
            log.error("conversation_save_summary_failed", session_id=session_id, error=str(e))
            raise

    async def check_crashed_sessions(
        self,
        tenant_id: str,
    ) -> list[str]:
        """Find sessions with status='started' but never 'completed'.

        These represent process crashes. Returns list of session IDs.
        """
        async with self.db.session() as s:
            result = await s.execute(
                select(AgentConversationRow.session_id).where(
                    AgentConversationRow.tenant_id == tenant_id,
                    AgentConversationRow.session_status == "started",
                )
            )
            return [row[0] for row in result.all()]

    async def cleanup_old_messages(
        self,
        tenant_id: str,
        days: int = 30,
    ) -> int:
        """Clear messages_json for sessions older than N days that have summaries.

        Keeps the row + summary, just frees the full message JSON to save space.
        Returns the number of sessions cleaned up.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        try:
            async with self.db.session() as s:
                result = await s.execute(
                    select(AgentConversationRow).where(
                        AgentConversationRow.tenant_id == tenant_id,
                        AgentConversationRow.summary.isnot(None),
                        AgentConversationRow.created_at < cutoff,
                        AgentConversationRow.messages_json != "[]",
                    )
                )
                rows = list(result.scalars().all())
                for row in rows:
                    row.messages_json = "[]"
                await s.commit()
                return len(rows)
        except Exception as e:
            log.error("conversation_cleanup_failed", tenant_id=tenant_id, error=str(e))
            raise

    async def get_session(
        self,
        session_id: str,
    ) -> dict | None:
        """Get a single session by ID. Returns full detail dict or None."""
        async with self.db.session() as s:
            result = await s.execute(
                select(AgentConversationRow).where(
                    AgentConversationRow.session_id == session_id,
                )
            )
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return {
                "session_id": row.session_id,
                "tenant_id": row.tenant_id,
                "trigger_type": row.trigger_type,
                "messages": json.loads(row.messages_json),
                "summary": row.summary,
                "token_count": row.token_count,
                "cost_usd": row.cost_usd,
                "session_status": row.session_status,
                "created_at": row.created_at,
            }

    async def list_sessions(
        self,
        tenant_id: str,
        limit: int = 30,
    ) -> list[dict]:
        """List sessions for a tenant, most recent first. Lightweight (no messages)."""
        async with self.db.session() as s:
            result = await s.execute(
                select(AgentConversationRow)
                .where(AgentConversationRow.tenant_id == tenant_id)
                .order_by(AgentConversationRow.created_at.desc())
                .limit(limit)
            )
            return [
                {
                    "session_id": row.session_id,
                    "trigger_type": row.trigger_type,
                    "summary": row.summary,
                    "token_count": row.token_count,
                    "cost_usd": row.cost_usd,
                    "session_status": row.session_status,
                    "created_at": row.created_at,
                }
                for row in result.scalars().all()
            ]
