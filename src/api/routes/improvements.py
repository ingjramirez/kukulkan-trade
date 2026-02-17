"""API routes for weekly self-improvement: snapshots, changelog, trend."""

import json

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_authorized_tenant_id, get_db
from src.api.schemas import (
    ImprovementSnapshotDetailResponse,
    ImprovementSnapshotListResponse,
    ImprovementTrendDataPoint,
    ImprovementTrendResponse,
    ParameterChangelogResponse,
)
from src.storage.database import Database

router = APIRouter(prefix="/api/agent/improvements", tags=["improvements"])


@router.get("/changelog", response_model=list[ParameterChangelogResponse])
async def get_changelog(
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> list[ParameterChangelogResponse]:
    """Get parameter change audit log."""
    entries = await db.get_parameter_changelog(tenant_id)
    return [
        ParameterChangelogResponse(
            id=e.id,
            parameter=e.parameter,
            old_value=e.old_value,
            new_value=e.new_value,
            reason=e.reason,
            snapshot_id=e.snapshot_id,
            applied_at=e.applied_at,
        )
        for e in entries
    ]


@router.get("/trend", response_model=ImprovementTrendResponse)
async def get_trend(
    weeks: int = 8,
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> ImprovementTrendResponse:
    """Get performance trend analysis across weekly snapshots."""
    from src.analysis.trend_analyzer import TrendAnalyzer

    analyzer = TrendAnalyzer()
    result = await analyzer.compute_trend(db, tenant_id=tenant_id, weeks=weeks)
    return ImprovementTrendResponse(
        classification=result.classification,
        win_rate_slope=result.win_rate_slope,
        pnl_slope=result.pnl_slope,
        data_points=[
            ImprovementTrendDataPoint(
                week_label=dp.week_label,
                win_rate_pct=dp.win_rate_pct,
                avg_pnl_pct=dp.avg_pnl_pct,
                total_trades=dp.total_trades,
            )
            for dp in result.data_points
        ],
        weeks_analyzed=result.weeks_analyzed,
    )


@router.get("", response_model=list[ImprovementSnapshotListResponse])
async def list_snapshots(
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> list[ImprovementSnapshotListResponse]:
    """List recent weekly improvement snapshots."""
    rows = await db.get_improvement_snapshots(tenant_id)
    return [
        ImprovementSnapshotListResponse(
            id=r.id,
            week_start=r.week_start,
            week_end=r.week_end,
            total_trades=r.total_trades,
            win_rate_pct=r.win_rate_pct,
            avg_pnl_pct=r.avg_pnl_pct,
            strategy_mode=r.strategy_mode,
            trailing_stop_multiplier=r.trailing_stop_multiplier,
            created_at=r.created_at,
        )
        for r in rows
    ]


@router.get("/{snapshot_id}", response_model=ImprovementSnapshotDetailResponse)
async def get_snapshot(
    snapshot_id: int,
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> ImprovementSnapshotDetailResponse:
    """Get a single improvement snapshot with full details."""
    row = await db.get_improvement_snapshot(snapshot_id, tenant_id=tenant_id)
    if not row:
        raise HTTPException(status_code=404, detail="Snapshot not found")

    # Parse JSON fields
    proposal = None
    if row.proposal_json:
        try:
            proposal = json.loads(row.proposal_json)
        except (json.JSONDecodeError, TypeError):
            pass

    applied = None
    if row.applied_changes:
        try:
            applied = json.loads(row.applied_changes)
        except (json.JSONDecodeError, TypeError):
            pass

    return ImprovementSnapshotDetailResponse(
        id=row.id,
        week_start=row.week_start,
        week_end=row.week_end,
        total_trades=row.total_trades,
        win_rate_pct=row.win_rate_pct,
        avg_pnl_pct=row.avg_pnl_pct,
        avg_alpha_vs_spy=row.avg_alpha_vs_spy,
        total_cost_usd=row.total_cost_usd,
        strategy_mode=row.strategy_mode,
        trailing_stop_multiplier=row.trailing_stop_multiplier,
        proposal_json=proposal,
        applied_changes=applied,
        report_text=row.report_text,
        created_at=row.created_at,
    )
