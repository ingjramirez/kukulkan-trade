"""GET /api/portfolios/B/outcomes — Trade outcome analysis endpoints."""

from fastapi import APIRouter, Depends, Query

from src.analysis.decision_quality import DecisionQualityTracker
from src.analysis.outcome_tracker import OutcomeTracker
from src.analysis.track_record import TrackRecord
from src.api.deps import get_authorized_tenant_id, get_db
from src.api.schemas import (
    CategoryWinRateResponse,
    DecisionQualityResponse,
    TrackRecordResponse,
    TradeOutcomeResponse,
)
from src.storage.database import Database

router = APIRouter(prefix="/api/portfolios/B", tags=["outcomes"])


@router.get("/outcomes", response_model=list[TradeOutcomeResponse])
async def list_outcomes(
    days: int = Query(30, ge=1, le=365),
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> list[TradeOutcomeResponse]:
    """Get trade outcomes for Portfolio B with benchmark comparisons."""
    tracker = OutcomeTracker(db)
    outcomes = await tracker.get_recent_outcomes(days=days, tenant_id=tenant_id)
    return [
        TradeOutcomeResponse(
            ticker=o.ticker,
            side=o.side,
            entry_price=o.entry_price,
            current_price=o.current_price,
            exit_price=o.exit_price,
            pnl_pct=o.pnl_pct,
            hold_days=o.hold_days,
            sector=o.sector,
            sector_etf_pct=o.sector_etf_pct,
            spy_pct=o.spy_pct,
            alpha_vs_sector=o.alpha_vs_sector,
            alpha_vs_spy=o.alpha_vs_spy,
            conviction=o.conviction,
            reasoning=o.reasoning,
        )
        for o in outcomes
    ]


@router.get("/track-record", response_model=TrackRecordResponse)
async def get_track_record(
    days: int = Query(90, ge=1, le=365),
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> TrackRecordResponse:
    """Get win rate analysis for Portfolio B trades."""
    tracker = OutcomeTracker(db)
    outcomes = await tracker.get_recent_outcomes(days=days, tenant_id=tenant_id)
    stats = TrackRecord().compute(outcomes)
    return TrackRecordResponse(
        total_trades=stats.total_trades,
        wins=stats.wins,
        losses=stats.losses,
        scratches=stats.scratches,
        win_rate_pct=stats.win_rate_pct,
        avg_pnl_pct=stats.avg_pnl_pct,
        avg_alpha_vs_spy=stats.avg_alpha_vs_spy,
        by_sector=[
            CategoryWinRateResponse(
                category=s.category,
                value=s.value,
                total=s.total,
                wins=s.wins,
                losses=s.losses,
                win_rate_pct=s.win_rate_pct,
                avg_pnl_pct=s.avg_pnl_pct,
                avg_alpha_vs_spy=s.avg_alpha_vs_spy,
            )
            for s in stats.by_sector
        ],
        by_conviction=[
            CategoryWinRateResponse(
                category=c.category,
                value=c.value,
                total=c.total,
                wins=c.wins,
                losses=c.losses,
                win_rate_pct=c.win_rate_pct,
                avg_pnl_pct=c.avg_pnl_pct,
                avg_alpha_vs_spy=c.avg_alpha_vs_spy,
            )
            for c in stats.by_conviction
        ],
        best_sector=stats.best_sector,
        worst_sector=stats.worst_sector,
    )


@router.get("/decision-quality", response_model=DecisionQualityResponse)
async def get_decision_quality(
    days: int = Query(30, ge=1, le=365),
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> DecisionQualityResponse:
    """Get forward-return analysis of recent decisions."""
    tracker = DecisionQualityTracker(db)
    qualities = await tracker.analyze_recent(days=days, tenant_id=tenant_id)
    summary = DecisionQualityTracker.summarize(qualities)
    return DecisionQualityResponse(
        total_decisions=summary.total_decisions,
        favorable_1d_pct=summary.favorable_1d_pct,
        favorable_3d_pct=summary.favorable_3d_pct,
        favorable_5d_pct=summary.favorable_5d_pct,
    )
