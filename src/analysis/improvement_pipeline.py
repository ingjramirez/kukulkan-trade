"""Weekly improvement pipeline: collect → analyze → apply → save → notify.

Orchestrates the full weekly self-improvement loop for a single tenant.
"""

from __future__ import annotations

import json

import structlog

from src.analysis.auto_apply import AutoApplyEngine
from src.analysis.weekly_improvement import (
    ImprovementAnalyzer,
    ImprovementProposal,
    WeeklyDataCollector,
    WeeklyPerformanceData,
)
from src.storage.database import Database

log = structlog.get_logger()

# Minimum trades to justify running the analyzer
MIN_TRADES_FOR_ANALYSIS = 1


class WeeklyImprovementPipeline:
    """End-to-end weekly self-improvement pipeline."""

    def __init__(self, db: Database, analyzer: ImprovementAnalyzer | None = None) -> None:
        self._db = db
        self._analyzer = analyzer or ImprovementAnalyzer()

    async def run(
        self,
        tenant_id: str = "default",
        notifier: object | None = None,
    ) -> dict:
        """Run the full weekly improvement pipeline.

        Returns:
            Summary dict with keys: tenant_id, status, total_trades, changes_applied, snapshot_id.
        """
        result = {"tenant_id": tenant_id, "status": "skipped"}

        try:
            # Step 1: Collect performance data
            collector = WeeklyDataCollector(self._db)
            data = await collector.collect(tenant_id)

            # Skip if too few trades
            if len(data.outcomes) < MIN_TRADES_FOR_ANALYSIS:
                log.info(
                    "improvement_skip_no_trades",
                    tenant_id=tenant_id,
                    trades=len(data.outcomes),
                )
                result["total_trades"] = len(data.outcomes)
                return result

            # Step 2: Load previous changes for context
            previous_changes = await self._load_previous_changes(tenant_id)

            # Step 3: Analyze with Sonnet
            proposal = await self._analyzer.analyze(data, previous_changes)

            # Step 4: Save snapshot (get ID before applying, so changelog entries link back)
            snapshot_id = await self._save_snapshot(data, proposal, applied_changes=None)

            # Step 5: Apply changes (with snapshot_id for audit trail)
            engine = AutoApplyEngine(self._db)
            applied = await engine.apply(tenant_id, proposal, snapshot_id=snapshot_id)

            # Step 5b: Update snapshot with applied results and report
            await self._db.update_improvement_snapshot_applied(
                snapshot_id,
                applied_changes=json.dumps(applied) if applied else None,
                report_text=_format_report(data, proposal, applied),
            )

            # Step 6: Notify via Telegram
            if notifier and hasattr(notifier, "send_message"):
                report = _format_telegram_report(data, proposal, applied)
                try:
                    await notifier.send_message(report, parse_mode="HTML")
                except Exception as e:
                    log.warning("improvement_telegram_failed", error=str(e))

            result.update(
                {
                    "status": "completed",
                    "total_trades": len(data.outcomes),
                    "changes_applied": len([a for a in applied if a.get("status") == "applied"]),
                    "changes_blocked": len([a for a in applied if a.get("status") == "blocked_flipflop"]),
                    "snapshot_id": snapshot_id,
                }
            )

            log.info(
                "improvement_pipeline_complete",
                tenant_id=tenant_id,
                trades=len(data.outcomes),
                changes=len(applied),
            )

            try:
                from src.events.event_bus import Event, EventType, event_bus

                event_bus.publish(
                    Event(
                        type=EventType.IMPROVEMENT_REPORT,
                        tenant_id=tenant_id,
                        data={
                            "changes_applied": len([a for a in applied if a.get("status") == "applied"]),
                            "proposals_total": len(applied),
                            "summary": (proposal.summary or "")[:200],
                        },
                    )
                )
            except Exception as exc:
                log.debug("event_publish_failed", error=str(exc))

        except Exception as e:
            log.exception("improvement_pipeline_failed", tenant_id=tenant_id, error=str(e))
            result["status"] = "error"
            result["error"] = str(e)

        return result

    async def _load_previous_changes(self, tenant_id: str) -> list[dict]:
        """Load recent changelog entries for the analyzer's context."""
        entries = await self._db.get_parameter_changelog(tenant_id, limit=20)
        return [
            {
                "parameter": e.parameter,
                "old_value": e.old_value,
                "new_value": e.new_value,
                "reason": e.reason,
            }
            for e in entries
        ]

    async def _save_snapshot(
        self,
        data: WeeklyPerformanceData,
        proposal: ImprovementProposal,
        applied_changes: list[dict] | None = None,
    ) -> int:
        """Save the weekly snapshot. applied_changes can be None initially."""
        return await self._db.save_improvement_snapshot(
            tenant_id=data.tenant_id,
            week_start=data.week_start,
            week_end=data.week_end,
            total_trades=len(data.outcomes),
            win_rate_pct=data.track_record.win_rate_pct if data.track_record else None,
            avg_pnl_pct=data.track_record.avg_pnl_pct if data.track_record else None,
            avg_alpha_vs_spy=data.track_record.avg_alpha_vs_spy if data.track_record else None,
            total_cost_usd=data.total_cost_usd,
            strategy_mode=data.current_strategy_mode,
            trailing_stop_multiplier=data.current_trailing_stop_multiplier,
            proposal_json=json.dumps(proposal.raw_json) if proposal.raw_json else None,
            applied_changes=json.dumps(applied_changes) if applied_changes else None,
            report_text=None,
        )


def _format_report(
    data: WeeklyPerformanceData,
    proposal: ImprovementProposal,
    applied: list[dict],
) -> str:
    """Format a plain-text report for DB storage."""
    lines = [
        f"Weekly Improvement Report: {data.week_start} to {data.week_end}",
        f"Trades: {len(data.outcomes)}",
    ]

    if data.track_record:
        lines.append(f"Win rate: {data.track_record.win_rate_pct:.1f}%")
        lines.append(f"Avg P&L: {data.track_record.avg_pnl_pct:+.2f}%")

    lines.append(f"AI cost: ${data.total_cost_usd:.2f}")
    lines.append(f"Summary: {proposal.summary}")

    if applied:
        lines.append("\nChanges:")
        for a in applied:
            status = a.get("status", "unknown")
            lines.append(f"  [{status}] {a['parameter']}: {a.get('old_value', '-')} → {a['new_value']}")
    else:
        lines.append("\nNo changes proposed.")

    return "\n".join(lines)


def _escape_html(text: str) -> str:
    """Escape HTML special chars for Telegram."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _format_telegram_report(
    data: WeeklyPerformanceData,
    proposal: ImprovementProposal,
    applied: list[dict],
) -> str:
    """Format an HTML Telegram message matching existing bot style."""
    lines = [
        "<b>Weekly Self-Improvement Report</b>",
        f"{data.week_start.isoformat()} to {data.week_end.isoformat()}",
        "",
    ]

    # Performance summary
    lines.append(f"Trades: {len(data.outcomes)}")
    if data.track_record:
        lines.append(f"Win rate: {data.track_record.win_rate_pct:.1f}%")
        lines.append(f"Avg P&L: {data.track_record.avg_pnl_pct:+.2f}%")
        if data.track_record.avg_alpha_vs_spy is not None:
            lines.append(f"Alpha vs SPY: {data.track_record.avg_alpha_vs_spy:+.2f}%")
    lines.append(f"AI cost: ${data.total_cost_usd:.2f}")
    lines.append("")

    # AI Assessment
    if proposal.summary:
        lines.append(f"<b>Assessment:</b> {_escape_html(proposal.summary)}")
        lines.append("")

    # Applied changes
    applied_items = [a for a in applied if a.get("status") == "applied"]
    blocked_items = [a for a in applied if a.get("status") == "blocked_flipflop"]

    if applied_items:
        lines.append(f"<b>Changes Applied ({len(applied_items)})</b>")
        for a in applied_items:
            param = _escape_html(a["parameter"])
            old = _escape_html(str(a.get("old_value", "-")))
            new = _escape_html(str(a["new_value"]))
            lines.append(f"  {param}: {old} → {new}")
        lines.append("")

    if blocked_items:
        lines.append(f"<b>Blocked — Flip-Flop ({len(blocked_items)})</b>")
        for a in blocked_items:
            lines.append(f"  {_escape_html(a['parameter'])}")
        lines.append("")

    if not applied_items and not blocked_items:
        lines.append("No parameter changes this week.")

    return "\n".join(lines)
