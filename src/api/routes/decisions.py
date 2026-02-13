"""GET /api/agent/decisions + tool-logs — Claude AI trade decisions and tool logs."""

import json
from datetime import date

from fastapi import APIRouter, Depends, Query

from src.api.deps import get_authorized_tenant_id, get_db
from src.api.schemas import AgentDecisionResponse, ToolCallLogResponse
from src.storage.database import Database

router = APIRouter(prefix="/api/agent", tags=["agent"])


@router.get("/decisions", response_model=list[AgentDecisionResponse])
async def list_decisions(
    limit: int = Query(10, ge=1, le=100),
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> list[AgentDecisionResponse]:
    rows = await db.get_agent_decisions(limit=limit, tenant_id=tenant_id)
    results = []
    for r in rows:
        proposed = None
        if r.proposed_trades:
            try:
                proposed = json.loads(r.proposed_trades)
            except (json.JSONDecodeError, TypeError):
                proposed = [r.proposed_trades]
        results.append(
            AgentDecisionResponse(
                id=r.id,
                date=r.date,
                prompt_summary=r.prompt_summary,
                response_summary=r.response_summary,
                proposed_trades=proposed,
                reasoning=r.reasoning,
                model_used=r.model_used,
                tokens_used=r.tokens_used,
                regime=r.regime,
                session_label=r.session_label,
                created_at=r.created_at,
            )
        )
    return results


@router.get("/tool-logs", response_model=list[ToolCallLogResponse])
async def list_tool_logs(
    session_date: date | None = Query(None, description="Filter by session date"),
    limit: int = Query(50, ge=1, le=200),
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> list[ToolCallLogResponse]:
    rows = await db.get_tool_call_logs(tenant_id=tenant_id, session_date=session_date, limit=limit)
    return [
        ToolCallLogResponse(
            id=r.id,
            session_date=r.session_date,
            session_label=r.session_label,
            turn=r.turn,
            tool_name=r.tool_name,
            tool_input=r.tool_input,
            tool_output_preview=r.tool_output_preview,
            success=r.success,
            error=r.error,
            influenced_decision=r.influenced_decision,
            created_at=r.created_at,
        )
        for r in rows
    ]
