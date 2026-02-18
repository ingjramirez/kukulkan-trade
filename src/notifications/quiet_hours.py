"""Quiet hours manager — suppresses Telegram during owner sleep window.

During quiet hours:
- No Telegram messages sent (queued to sentinel_actions instead)
- SSE events still fire (dashboard is opt-in, won't wake owner)

Morning delivery:
- At quiet_hours_end, send single summary of all queued items
- Owner can /execute-all, /cancel-all, or respond per-item
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import structlog

if TYPE_CHECKING:
    from src.storage.database import Database

log = structlog.get_logger()


class QuietHoursManager:
    """Manages notification quiet hours per tenant."""

    def __init__(self, db: Database) -> None:
        self.db = db

    async def is_quiet(self, tenant_id: str) -> bool:
        """Check if current time falls within tenant's quiet hours."""
        tenant = await self.db.get_tenant(tenant_id)
        if tenant is None:
            return False

        tz_name = getattr(tenant, "quiet_hours_timezone", None) or "America/Mexico_City"
        start_str = getattr(tenant, "quiet_hours_start", None) or "21:00"
        end_str = getattr(tenant, "quiet_hours_end", None) or "07:00"

        tz = ZoneInfo(tz_name)
        now = datetime.now(tz).time()
        start = time.fromisoformat(start_str)
        end = time.fromisoformat(end_str)

        # Handle overnight span (e.g., 21:00 -> 07:00)
        if start > end:
            return now >= start or now < end
        else:
            return start <= now < end

    async def queue_notification(
        self,
        tenant_id: str,
        action_type: str,
        ticker: str,
        reason: str,
        source: str,
        alert_level: str,
    ) -> int:
        """Queue a notification as a pending sentinel action."""
        return await self.db.save_sentinel_action(
            tenant_id=tenant_id,
            action_type=action_type,
            ticker=ticker,
            reason=reason,
            source=source,
            alert_level=alert_level,
            status="pending",
        )

    async def get_morning_summary(self, tenant_id: str, max_age_hours: int = 24) -> list[dict]:
        """Get pending sentinel actions for morning delivery, filtered by age."""
        actions = await self.db.get_pending_sentinel_actions(tenant_id)
        if not actions or max_age_hours <= 0:
            return actions
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        result = []
        for a in actions:
            created = a.get("created_at", "")
            if isinstance(created, str) and created:
                try:
                    dt = datetime.fromisoformat(created)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if dt < cutoff:
                        log.debug("morning_summary_skipped_stale", action_id=a.get("id"), age_hours=max_age_hours)
                        continue
                except (ValueError, TypeError):
                    pass  # Include actions with unparseable dates
            result.append(a)
        return result

    async def resolve_action(self, action_id: int, status: str, resolved_by: str) -> None:
        """Resolve a queued sentinel action."""
        await self.db.resolve_sentinel_action(action_id, status, resolved_by)
