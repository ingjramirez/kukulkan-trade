"""GET /api/signals — Ticker signal engine endpoints."""

import json

from fastapi import APIRouter, Depends

from src.api.deps import get_authorized_tenant_id, get_db
from src.api.schemas import SignalRankingsResponse, TickerSignalResponse
from src.storage.database import Database

router = APIRouter(prefix="/api/signals", tags=["signals"])


@router.get("/rankings", response_model=SignalRankingsResponse)
async def get_signal_rankings(
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> SignalRankingsResponse:
    """Get the latest signal engine rankings for all tickers."""
    rows = await db.get_latest_signals(tenant_id)
    if not rows:
        return SignalRankingsResponse(total=0, scored_at=None, signals=[])

    signals = []
    for r in rows:
        alerts = json.loads(r.alerts) if r.alerts else []
        signals.append(
            TickerSignalResponse(
                ticker=r.ticker,
                composite_score=r.composite_score,
                rank=r.rank,
                prev_rank=r.prev_rank,
                rank_velocity=r.rank_velocity,
                momentum_20d=r.momentum_20d,
                momentum_63d=r.momentum_63d,
                rsi=r.rsi,
                macd_histogram=r.macd_histogram,
                sma_trend_score=r.sma_trend_score,
                bollinger_pct_b=r.bollinger_pct_b,
                volume_ratio=r.volume_ratio,
                alerts=alerts,
                scored_at=r.scored_at,
            )
        )

    return SignalRankingsResponse(
        total=len(signals),
        scored_at=rows[0].scored_at,
        signals=signals,
    )
