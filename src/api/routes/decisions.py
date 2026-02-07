"""GET /api/agent/decisions — Claude AI trade decisions."""

import json

from fastapi import APIRouter, Depends, Query

from src.api.deps import get_current_user, get_db
from src.api.schemas import AgentDecisionResponse
from src.storage.database import Database

router = APIRouter(prefix="/api/agent", tags=["agent"])


@router.get("/decisions", response_model=list[AgentDecisionResponse])
async def list_decisions(
    limit: int = Query(10, ge=1, le=100),
    db: Database = Depends(get_db),
    _user: str = Depends(get_current_user),
) -> list[AgentDecisionResponse]:
    rows = await db.get_agent_decisions(limit=limit)
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
                created_at=r.created_at,
            )
        )
    return results
