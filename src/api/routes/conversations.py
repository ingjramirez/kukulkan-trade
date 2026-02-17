"""GET /api/agent/conversations — persistent agent conversation history."""

from fastapi import APIRouter, Depends, Query

from src.agent.conversation_store import ConversationStore
from src.api.deps import get_authorized_tenant_id, get_db
from src.api.schemas import ConversationDetailResponse, ConversationSessionResponse
from src.storage.database import Database

router = APIRouter(prefix="/api/agent", tags=["agent"])


@router.get("/conversations", response_model=list[ConversationSessionResponse])
async def list_conversations(
    limit: int = Query(30, ge=1, le=100),
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> list[ConversationSessionResponse]:
    """List agent conversation sessions (most recent first, no messages)."""
    store = ConversationStore(db)
    sessions = await store.list_sessions(tenant_id, limit=limit)
    return [
        ConversationSessionResponse(
            session_id=s["session_id"],
            trigger_type=s["trigger_type"],
            summary=s["summary"],
            token_count=s["token_count"],
            cost_usd=s["cost_usd"],
            session_status=s["session_status"],
            created_at=s["created_at"],
        )
        for s in sessions
    ]


@router.get("/conversations/{session_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    session_id: str,
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> ConversationDetailResponse:
    """Get a single conversation session with full messages."""
    from fastapi import HTTPException

    store = ConversationStore(db)
    session = await store.get_session(session_id)
    if session is None or session["tenant_id"] != tenant_id:
        raise HTTPException(status_code=404, detail="Session not found")
    return ConversationDetailResponse(
        session_id=session["session_id"],
        tenant_id=session["tenant_id"],
        trigger_type=session["trigger_type"],
        summary=session.get("summary"),
        token_count=session.get("token_count", 0),
        cost_usd=session.get("cost_usd", 0.0),
        session_status=session.get("session_status", "completed"),
        messages=session.get("messages", []),
        created_at=session["created_at"],
    )
