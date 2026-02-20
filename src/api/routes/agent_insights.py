"""API routes for agent insights: posture, playbook, calibration, inverse exposure."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from src.api.deps import get_authorized_tenant_id, get_db
from src.api.schemas import (
    ConvictionCalibrationResponse,
    InverseExposureResponse,
    InversePositionResponse,
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


@router.get("/inverse-exposure", response_model=InverseExposureResponse)
async def get_inverse_exposure(
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> InverseExposureResponse:
    """Get current inverse ETF exposure for Portfolio B."""
    from config.universe import INVERSE_ETF_META, is_equity_hedge
    from src.analysis.risk_manager import (
        MAX_INVERSE_POSITIONS,
        MAX_SINGLE_INVERSE_PCT,
        MAX_TOTAL_INVERSE_PCT,
    )

    positions = await db.get_positions("B", tenant_id=tenant_id)
    portfolio = await db.get_portfolio("B", tenant_id=tenant_id)
    total_value = portfolio.total_value if portfolio else 0.0

    inverse_positions: list[InversePositionResponse] = []
    inverse_total_value = 0.0

    # Fetch trades once outside the loop to avoid N+1 queries
    all_trades = await db.get_trades("B", tenant_id=tenant_id)

    for p in positions:
        if p.ticker not in INVERSE_ETF_META:
            continue
        value = p.shares * p.avg_price
        inverse_total_value += value
        pct = value / total_value * 100 if total_value > 0 else 0.0

        # Compute days held from most recent BUY
        buy_trades = [t for t in all_trades if t.ticker == p.ticker and t.side == "BUY"]
        days_held = None
        hold_alert = None
        if buy_trades:
            latest_buy = buy_trades[0]
            if latest_buy.executed_at:
                now = datetime.now(timezone.utc)
                executed = latest_buy.executed_at
                if executed.tzinfo is None:
                    now = now.replace(tzinfo=None)
                days_held = (now - executed).days
                if days_held >= 5:
                    hold_alert = "review"
                elif days_held >= 3:
                    hold_alert = "warning"

        inverse_positions.append(
            InversePositionResponse(
                ticker=p.ticker,
                value=round(value, 2),
                pct=round(pct, 1),
                equity_hedge=is_equity_hedge(p.ticker),
                days_held=days_held,
                hold_alert=hold_alert,
            )
        )

    # Compute net equity pct
    equity_invested = sum(p.shares * p.avg_price for p in positions) - inverse_total_value
    equity_hedge_value = sum(ip.value for ip in inverse_positions if ip.equity_hedge)
    net_equity_pct = (equity_invested - equity_hedge_value) / total_value * 100 if total_value > 0 else 0.0

    return InverseExposureResponse(
        total_value=round(inverse_total_value, 2),
        total_pct=round(inverse_total_value / total_value * 100, 1) if total_value > 0 else 0.0,
        net_equity_pct=round(net_equity_pct, 1),
        positions=inverse_positions,
        rules={
            "max_single_pct": MAX_SINGLE_INVERSE_PCT * 100,
            "max_total_pct": MAX_TOTAL_INVERSE_PCT * 100,
            "max_positions": MAX_INVERSE_POSITIONS,
        },
    )
