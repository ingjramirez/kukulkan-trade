"""Weekly self-improvement: data collection + AI analysis for Portfolio B.

Collects 7-day performance data, sends it to Sonnet for structured
analysis, and returns an ImprovementProposal with bounded changes.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

import structlog

from src.analysis.outcome_tracker import OutcomeTracker, TradeOutcome
from src.analysis.track_record import TrackRecord, TrackRecordStats
from src.storage.database import Database

log = structlog.get_logger()

# Bounds for auto-tunable parameters
VALID_STRATEGY_MODES = {"conservative", "standard", "aggressive"}
TRAILING_STOP_MULTIPLIER_MIN = 0.5
TRAILING_STOP_MULTIPLIER_MAX = 2.0
MAX_TICKER_EXCLUSIONS_PER_WEEK = 3
MAX_LEARNINGS_PER_WEEK = 3


@dataclass
class WeeklyPerformanceData:
    """Aggregated 7-day performance data for the improvement analyzer."""

    tenant_id: str
    week_start: date
    week_end: date
    outcomes: list[TradeOutcome] = field(default_factory=list)
    track_record: TrackRecordStats | None = None
    current_strategy_mode: str = "conservative"
    current_trailing_stop_multiplier: float = 1.0
    current_ticker_exclusions: list[str] = field(default_factory=list)
    total_cost_usd: float = 0.0
    posture_history: list[str] = field(default_factory=list)
    universe_size: int = 0


@dataclass
class ProposedChange:
    """A single proposed parameter change."""

    category: str  # strategy_mode, trailing_stop, universe_exclude, learning
    parameter: str
    old_value: str | None
    new_value: str
    reason: str


@dataclass
class ImprovementProposal:
    """Validated proposal from the AI analyzer."""

    changes: list[ProposedChange] = field(default_factory=list)
    summary: str = ""
    raw_json: dict = field(default_factory=dict)


class WeeklyDataCollector:
    """Collects 7-day performance data from existing analysis modules."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def collect(self, tenant_id: str = "default") -> WeeklyPerformanceData:
        """Gather all performance data for the past 7 days."""
        today = date.today()
        week_start = today - timedelta(days=7)

        # Get trade outcomes for past 7 days
        tracker = OutcomeTracker(self._db)
        outcomes = await tracker.get_recent_outcomes(days=7, tenant_id=tenant_id)

        # Compute track record stats
        record = TrackRecord()
        track_stats = record.compute(outcomes) if outcomes else None

        # Get current tenant config
        tenant = await self._db.get_tenant(tenant_id)
        strategy_mode = tenant.strategy_mode if tenant else "conservative"
        trail_mult = tenant.trailing_stop_multiplier if tenant and tenant.trailing_stop_multiplier else 1.0
        exclusions: list[str] = []
        if tenant and tenant.ticker_exclusions:
            try:
                exclusions = json.loads(tenant.ticker_exclusions)
            except (json.JSONDecodeError, TypeError):
                pass

        # Get cost data for the week
        total_cost = 0.0
        for day_offset in range(7):
            day = week_start + timedelta(days=day_offset)
            total_cost += await self._db.get_daily_spend(tenant_id, day)

        # Get posture history
        posture_rows = await self._db.get_posture_history(tenant_id)
        posture_labels = [r.effective_posture for r in posture_rows[:7]]

        # Universe size
        approved = await self._db.get_approved_tickers(tenant_id=tenant_id)
        universe_size = len(approved) if approved else 0

        return WeeklyPerformanceData(
            tenant_id=tenant_id,
            week_start=week_start,
            week_end=today,
            outcomes=outcomes,
            track_record=track_stats,
            current_strategy_mode=strategy_mode,
            current_trailing_stop_multiplier=trail_mult,
            current_ticker_exclusions=exclusions,
            total_cost_usd=round(total_cost, 4),
            posture_history=posture_labels,
            universe_size=universe_size,
        )


ANALYZER_SYSTEM_PROMPT = """You are an AI trading performance analyst for Kukulkan Trade.
You analyze weekly Portfolio B performance data and propose bounded parameter adjustments.

RULES:
1. strategy_mode: only "conservative", "standard", or "aggressive"
2. trailing_stop_multiplier: float between 0.5 and 2.0 (1.0 = default, <1.0 = tighter stops, >1.0 = wider stops)
3. universe_exclude: up to 3 tickers to exclude (only tickers with consistent losses)
4. learning: up to 3 short observations to remember (stored as agent memory)
5. You CANNOT change budget/cost caps — those are owner-controlled
6. Be conservative with changes — only propose when data clearly supports it
7. If performance is acceptable (win rate >=50%, positive P&L), propose minimal changes

Respond with ONLY valid JSON matching this schema:
{
  "changes": [
    {
      "category": "strategy_mode|trailing_stop|universe_exclude|learning",
      "parameter": "<param name>",
      "new_value": "<value>",
      "reason": "<1-2 sentence justification>"
    }
  ],
  "summary": "<2-3 sentence overall assessment>"
}

If no changes are needed, return {"changes": [], "summary": "..."}."""


class ImprovementAnalyzer:
    """Sends weekly performance data to Sonnet for improvement proposals."""

    def __init__(self, model: str = "claude-sonnet-4-5-20250929") -> None:
        self._model = model

    def _build_prompt(self, data: WeeklyPerformanceData, previous_changes: list[dict] | None = None) -> str:
        """Build the user prompt from collected data."""
        parts = [
            f"## Weekly Performance Review: {data.week_start} to {data.week_end}",
            f"Tenant: {data.tenant_id}",
            f"Total trades: {len(data.outcomes)}",
        ]

        if data.track_record:
            stats = data.track_record
            parts.append(f"Win rate: {stats.win_rate_pct:.1f}%")
            parts.append(f"Avg P&L: {stats.avg_pnl_pct:+.2f}%")
            if stats.avg_alpha_vs_spy is not None:
                parts.append(f"Avg alpha vs SPY: {stats.avg_alpha_vs_spy:+.2f}%")
            if stats.best_sector:
                parts.append(f"Best sector: {stats.best_sector}")
            if stats.worst_sector:
                parts.append(f"Worst sector: {stats.worst_sector}")

            if stats.by_sector:
                parts.append("\n### By Sector:")
                for s in stats.by_sector:
                    parts.append(f"  {s.value}: {s.win_rate_pct:.0f}%W, {s.avg_pnl_pct:+.1f}%, n={s.total}")

            if stats.by_conviction:
                parts.append("\n### By Conviction:")
                for c in stats.by_conviction:
                    parts.append(f"  {c.value}: {c.win_rate_pct:.0f}%W, {c.avg_pnl_pct:+.1f}%, n={c.total}")

        # Losing trades detail
        losers = [o for o in data.outcomes if o.pnl_pct < -0.5]
        if losers:
            parts.append(f"\n### Losing Trades ({len(losers)}):")
            for o in losers[:10]:
                parts.append(f"  {o.ticker}: {o.pnl_pct:+.1f}%, {o.sector}, {o.conviction}")

        parts.append("\n### Current Config:")
        parts.append(f"  Strategy mode: {data.current_strategy_mode}")
        parts.append(f"  Trailing stop multiplier: {data.current_trailing_stop_multiplier}")
        parts.append(f"  Ticker exclusions: {data.current_ticker_exclusions}")
        parts.append(f"  AI cost this week: ${data.total_cost_usd:.2f}")
        parts.append(f"  Universe size: {data.universe_size}")
        parts.append(f"  Recent postures: {data.posture_history}")

        if previous_changes:
            parts.append("\n### Recent Parameter Changes (last 4 weeks):")
            for ch in previous_changes[:10]:
                parts.append(f"  {ch['parameter']}: {ch['old_value']} → {ch['new_value']} ({ch.get('reason', '')})")

        parts.append("\nPropose parameter adjustments (JSON only):")
        return "\n".join(parts)

    async def analyze(
        self,
        data: WeeklyPerformanceData,
        previous_changes: list[dict] | None = None,
    ) -> ImprovementProposal:
        """Call Sonnet to analyze performance and propose changes."""
        import anthropic

        prompt = self._build_prompt(data, previous_changes)

        try:
            client = anthropic.Anthropic(max_retries=5)
            response = await asyncio.to_thread(
                client.messages.create,
                model=self._model,
                max_tokens=1024,
                system=ANALYZER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            raw_text = response.content[0].text.strip()
            return self._parse_response(raw_text, data)

        except Exception as e:
            log.error("improvement_analyzer_failed", error=str(e))
            return ImprovementProposal(summary=f"Analysis failed: {e}")

    def _parse_response(self, raw_text: str, data: WeeklyPerformanceData) -> ImprovementProposal:
        """Parse and validate the Sonnet JSON response."""
        # Strip markdown code fences if present
        fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", raw_text, re.DOTALL)
        text = fence_match.group(1).strip() if fence_match else raw_text.strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            log.warning("improvement_parse_failed", raw=raw_text[:200])
            return ImprovementProposal(summary="Failed to parse AI response", raw_json={})

        changes: list[ProposedChange] = []
        raw_changes = parsed.get("changes", [])

        exclude_count = 0
        learning_count = 0

        for ch in raw_changes:
            category = ch.get("category", "")
            new_value = str(ch.get("new_value", ""))
            reason = ch.get("reason", "")
            parameter = ch.get("parameter", category)

            if category == "strategy_mode":
                if new_value not in VALID_STRATEGY_MODES:
                    log.warning("invalid_strategy_mode_proposal", value=new_value)
                    continue
                if new_value == data.current_strategy_mode:
                    continue
                changes.append(
                    ProposedChange(
                        category=category,
                        parameter="strategy_mode",
                        old_value=data.current_strategy_mode,
                        new_value=new_value,
                        reason=reason,
                    )
                )

            elif category == "trailing_stop":
                try:
                    mult = float(new_value)
                except ValueError:
                    continue
                mult = max(TRAILING_STOP_MULTIPLIER_MIN, min(TRAILING_STOP_MULTIPLIER_MAX, mult))
                if abs(mult - data.current_trailing_stop_multiplier) < 0.01:
                    continue
                changes.append(
                    ProposedChange(
                        category=category,
                        parameter="trailing_stop_multiplier",
                        old_value=str(data.current_trailing_stop_multiplier),
                        new_value=str(round(mult, 2)),
                        reason=reason,
                    )
                )

            elif category == "universe_exclude":
                if exclude_count >= MAX_TICKER_EXCLUSIONS_PER_WEEK:
                    continue
                ticker = new_value.upper().strip()
                if not ticker or ticker in data.current_ticker_exclusions:
                    continue
                exclude_count += 1
                changes.append(
                    ProposedChange(
                        category=category,
                        parameter=f"ticker_exclusion:{ticker}",
                        old_value=None,
                        new_value=ticker,
                        reason=reason,
                    )
                )

            elif category == "learning":
                if learning_count >= MAX_LEARNINGS_PER_WEEK:
                    continue
                if not new_value.strip():
                    continue
                learning_count += 1
                changes.append(
                    ProposedChange(
                        category=category,
                        parameter=parameter or f"learning_{learning_count}",
                        old_value=None,
                        new_value=new_value.strip()[:500],  # Cap length
                        reason=reason,
                    )
                )

        return ImprovementProposal(
            changes=changes,
            summary=parsed.get("summary", ""),
            raw_json=parsed,
        )
