"""API routes for agent insights: posture, playbook, calibration."""

from fastapi import APIRouter, Depends

from src.api.deps import get_authorized_tenant_id, get_db
from src.api.schemas import (
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
