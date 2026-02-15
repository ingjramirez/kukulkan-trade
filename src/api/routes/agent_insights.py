"""API routes for agent insights: posture, playbook, calibration, budget."""

from datetime import date

from fastapi import APIRouter, Depends

from src.api.deps import get_authorized_tenant_id, get_db
from src.api.schemas import (
    BudgetStatusResponse,
    ConvictionCalibrationResponse,
    PlaybookCellResponse,
    PostureHistoryResponse,
)
from src.storage.database import Database

router = APIRouter(prefix="/api/agent", tags=["agent-insights"])


@router.get("/posture", response_model=list[PostureHistoryResponse])
async def get_posture_history(
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> list[PostureHistoryResponse]:
    """Get posture history for a tenant."""
    rows = await db.get_posture_history(tenant_id)
    return [
        PostureHistoryResponse(
            session_date=r.session_date,
            session_label=r.session_label,
            posture=r.posture,
            effective_posture=r.effective_posture,
            reason=r.reason,
            created_at=r.created_at,
        )
        for r in rows
    ]


@router.get("/playbook", response_model=list[PlaybookCellResponse])
async def get_latest_playbook(
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> list[PlaybookCellResponse]:
    """Get the latest empirical playbook snapshot."""
    rows = await db.get_latest_playbook(tenant_id)
    return [
        PlaybookCellResponse(
            regime=r.regime,
            sector=r.sector,
            total_trades=r.total_trades,
            wins=r.wins,
            losses=r.losses,
            win_rate_pct=r.win_rate_pct,
            avg_pnl_pct=r.avg_pnl_pct,
            recommendation=r.recommendation,
        )
        for r in rows
    ]


@router.get("/calibration", response_model=list[ConvictionCalibrationResponse])
async def get_latest_calibration(
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> list[ConvictionCalibrationResponse]:
    """Get the latest conviction calibration snapshot."""
    rows = await db.get_latest_calibration(tenant_id)
    return [
        ConvictionCalibrationResponse(
            conviction_level=r.conviction_level,
            total_trades=r.total_trades,
            wins=r.wins,
            losses=r.losses,
            win_rate_pct=r.win_rate_pct,
            avg_pnl_pct=r.avg_pnl_pct,
            assessment=r.assessment,
            suggested_multiplier=r.suggested_multiplier,
        )
        for r in rows
    ]


@router.get("/budget", response_model=BudgetStatusResponse)
async def get_budget_status(
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> BudgetStatusResponse:
    """Get current daily/monthly agent budget status for a tenant."""
    from config.settings import settings

    today = date.today()
    daily_spent = await db.get_daily_spend(tenant_id, today)
    monthly_spent = await db.get_monthly_spend(tenant_id, today.year, today.month)

    daily_limit = settings.agent.daily_budget
    monthly_limit = settings.agent.monthly_budget

    return BudgetStatusResponse(
        daily_spent=round(daily_spent, 4),
        daily_limit=daily_limit,
        daily_remaining=round(max(0.0, daily_limit - daily_spent), 4),
        monthly_spent=round(monthly_spent, 4),
        monthly_limit=monthly_limit,
        monthly_remaining=round(max(0.0, monthly_limit - monthly_spent), 4),
        daily_exhausted=daily_spent >= daily_limit,
        monthly_exhausted=monthly_spent >= monthly_limit,
        haiku_only=monthly_spent >= monthly_limit * 0.80,
    )
