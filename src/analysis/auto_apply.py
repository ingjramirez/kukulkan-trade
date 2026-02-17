"""Auto-apply engine for weekly improvement proposals.

Applies bounded parameter changes (strategy mode, trailing stops,
ticker exclusions, learnings) with flip-flop protection and audit logging.
"""

from __future__ import annotations

import json

import structlog

from src.analysis.weekly_improvement import ImprovementProposal, ProposedChange
from src.storage.database import Database

log = structlog.get_logger()

# Flip-flop protection: if a parameter has changed >= this many times
# in the past 4 weeks, block further automatic changes.
FLIP_FLOP_THRESHOLD = 3
FLIP_FLOP_WEEKS = 4


class AutoApplyEngine:
    """Applies validated improvement proposals to tenant configuration."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def apply(
        self,
        tenant_id: str,
        proposal: ImprovementProposal,
        snapshot_id: int | None = None,
    ) -> list[dict]:
        """Apply all changes in the proposal, returning a list of applied changes.

        Returns:
            List of dicts with keys: parameter, old_value, new_value, reason, status.
            status is "applied" or "blocked_flipflop".
        """
        results: list[dict] = []

        for change in proposal.changes:
            try:
                result = await self._apply_single(tenant_id, change, snapshot_id)
                results.append(result)
            except Exception as e:
                log.error(
                    "auto_apply_failed",
                    tenant_id=tenant_id,
                    parameter=change.parameter,
                    error=str(e),
                )
                results.append(
                    {
                        "parameter": change.parameter,
                        "old_value": change.old_value,
                        "new_value": change.new_value,
                        "reason": change.reason,
                        "status": "error",
                        "error": str(e),
                    }
                )

        return results

    async def _apply_single(
        self,
        tenant_id: str,
        change: ProposedChange,
        snapshot_id: int | None,
    ) -> dict:
        """Apply a single change with flip-flop check."""
        result = {
            "parameter": change.parameter,
            "old_value": change.old_value,
            "new_value": change.new_value,
            "reason": change.reason,
        }

        # Check flip-flop for mutable parameters
        if change.category in ("strategy_mode", "trailing_stop"):
            if await self._is_flip_flopping(tenant_id, change.parameter):
                result["status"] = "blocked_flipflop"
                log.warning(
                    "flip_flop_blocked",
                    tenant_id=tenant_id,
                    parameter=change.parameter,
                )
                return result

        # Dispatch to category-specific handler
        if change.category == "strategy_mode":
            await self._apply_strategy_mode(tenant_id, change)
        elif change.category == "trailing_stop":
            await self._apply_trailing_stop(tenant_id, change)
        elif change.category == "universe_exclude":
            await self._apply_universe_exclude(tenant_id, change)
        elif change.category == "learning":
            await self._apply_learning(tenant_id, change)
        else:
            result["status"] = "unknown_category"
            return result

        # Log to changelog
        await self._db.insert_parameter_changelog(
            tenant_id=tenant_id,
            parameter=change.parameter,
            old_value=change.old_value,
            new_value=change.new_value,
            reason=change.reason,
            snapshot_id=snapshot_id,
        )

        result["status"] = "applied"
        log.info(
            "auto_apply_change",
            tenant_id=tenant_id,
            parameter=change.parameter,
            old_value=change.old_value,
            new_value=change.new_value,
        )
        return result

    async def _is_flip_flopping(self, tenant_id: str, parameter: str) -> bool:
        """Check if a parameter has been changed too frequently."""
        recent = await self._db.get_parameter_changes_for(tenant_id, parameter, weeks=FLIP_FLOP_WEEKS)
        return len(recent) >= FLIP_FLOP_THRESHOLD

    async def _apply_strategy_mode(self, tenant_id: str, change: ProposedChange) -> None:
        """Update tenant's strategy_mode."""
        await self._db.update_tenant(tenant_id, {"strategy_mode": change.new_value})

    async def _apply_trailing_stop(self, tenant_id: str, change: ProposedChange) -> None:
        """Update tenant's trailing_stop_multiplier."""
        multiplier = float(change.new_value)
        await self._db.update_tenant(tenant_id, {"trailing_stop_multiplier": multiplier})

    async def _apply_universe_exclude(self, tenant_id: str, change: ProposedChange) -> None:
        """Append a ticker to the tenant's ticker_exclusions list."""
        tenant = await self._db.get_tenant(tenant_id)
        if not tenant:
            return

        current: list[str] = []
        if tenant.ticker_exclusions:
            try:
                current = json.loads(tenant.ticker_exclusions)
            except (json.JSONDecodeError, TypeError):
                pass

        ticker = change.new_value.upper().strip()
        if ticker not in current:
            current.append(ticker)
            await self._db.update_tenant(
                tenant_id,
                {"ticker_exclusions": json.dumps(current)},
            )

    async def _apply_learning(self, tenant_id: str, change: ProposedChange) -> None:
        """Store a learning as an agent memory note."""
        key = f"learning:{change.parameter}"
        await self._db.upsert_agent_memory(
            category="agent_note",
            key=key,
            content=change.new_value,
            tenant_id=tenant_id,
        )
