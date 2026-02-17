"""Daily and monthly budget tracking for agent sessions.

Queries the agent_budget_log table to enforce spend limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import structlog

from src.agent.token_tracker import TokenTracker
from src.storage.database import Database

log = structlog.get_logger()


@dataclass(frozen=True)
class BudgetStatus:
    """Current budget status for a tenant."""

    daily_spent: float
    daily_limit: float
    monthly_spent: float
    monthly_limit: float

    @property
    def daily_remaining(self) -> float:
        return max(0.0, self.daily_limit - self.daily_spent)

    @property
    def monthly_remaining(self) -> float:
        return max(0.0, self.monthly_limit - self.monthly_spent)

    @property
    def daily_exhausted(self) -> bool:
        return self.daily_spent >= self.daily_limit

    @property
    def monthly_exhausted(self) -> bool:
        return self.monthly_spent >= self.monthly_limit

    @property
    def haiku_only(self) -> bool:
        """True if monthly spend > 80% — use Haiku-only for non-FULL sessions."""
        return self.monthly_spent >= self.monthly_limit * 0.80


class BudgetTracker:
    """Enforces daily and monthly agent budget limits.

    Args:
        db: Database instance for budget log queries.
        daily_limit: Maximum daily spend in USD.
        monthly_limit: Maximum monthly spend in USD.
    """

    def __init__(
        self,
        db: Database,
        daily_limit: float | None = None,
        monthly_limit: float | None = None,
    ) -> None:
        from config.settings import settings

        self._db = db
        self._daily_limit = daily_limit if daily_limit is not None else settings.agent.daily_budget
        self._monthly_limit = monthly_limit if monthly_limit is not None else settings.agent.monthly_budget

    async def check_budget(
        self,
        tenant_id: str,
        today: date | None = None,
    ) -> BudgetStatus:
        """Check current budget status for a tenant.

        Args:
            tenant_id: Tenant UUID.
            today: Date to check (defaults to today).

        Returns:
            BudgetStatus with spend totals and limits.
        """
        today = today or date.today()
        daily_spent = await self._db.get_daily_spend(tenant_id, today)
        monthly_spent = await self._db.get_monthly_spend(tenant_id, today.year, today.month)

        return BudgetStatus(
            daily_spent=daily_spent,
            daily_limit=self._daily_limit,
            monthly_spent=monthly_spent,
            monthly_limit=self._monthly_limit,
        )

    async def record_session(
        self,
        tenant_id: str,
        session_date: date,
        session_label: str,
        session_id: str | None,
        token_tracker: TokenTracker,
        session_profile: str | None = None,
    ) -> None:
        """Save a session's cost to the budget log.

        Args:
            tenant_id: Tenant UUID.
            session_date: Date of the session.
            session_label: Trigger label (morning/midday/close/etc).
            session_id: Unique session ID.
            token_tracker: TokenTracker with all entries for this session.
            session_profile: Session profile used (FULL/LIGHT/etc).
        """
        summary = token_tracker.summary()
        await self._db.save_budget_log(
            tenant_id=tenant_id,
            session_date=session_date,
            session_label=session_label,
            session_id=session_id,
            input_tokens=summary["total_input_tokens"],
            output_tokens=summary["total_output_tokens"],
            cache_read_tokens=summary["total_cache_read_tokens"],
            cache_creation_tokens=summary["total_cache_creation_tokens"],
            cost_usd=summary["total_cost_usd"],
            session_profile=session_profile,
        )
        log.info(
            "budget_session_recorded",
            tenant_id=tenant_id,
            cost_usd=summary["total_cost_usd"],
            session_label=session_label,
        )
        try:
            from src.events.event_bus import Event, EventType, event_bus

            event_bus.publish(
                Event(
                    type=EventType.BUDGET_UPDATED,
                    tenant_id=tenant_id,
                    data={
                        "cost_usd": summary["total_cost_usd"],
                        "session_label": session_label,
                    },
                )
            )
        except Exception:
            pass
