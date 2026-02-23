"""Chat API — interactive conversation with the AI trading agent.

Endpoints:
    POST /api/chat          — non-streaming (returns when response is complete)
    POST /api/chat/stream   — SSE streaming (events as the agent responds)
    GET  /api/chat/history  — recent message history

SSE event format (stream endpoint):
    data: {"type": "text",        "text": "..."}
    data: {"type": "tool_use",    "id": "...", "name": "...", "input": {...}}
    data: {"type": "tool_result", "tool_use_id": "...", "content": "..."}
    data: {"type": "done",        "session_id": "...", "num_turns": N, "duration_ms": N}
    data: {"type": "error",       "message": "..."}
"""

from __future__ import annotations

import json
from datetime import datetime

import structlog
from fastapi import Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field

from src.agent.claude_invoker import ClaudeInvoker
from src.api.deps import get_authorized_tenant_id, get_db, get_invoker
from src.storage.database import Database

log = structlog.get_logger()
router = APIRouter()


# ── Schemas ───────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    content: str
    session_id: str | None
    tool_calls: list[dict]
    num_turns: int
    duration_ms: int


class ChatMessageResponse(BaseModel):
    id: int
    role: str
    content: str
    tool_calls: list[dict]
    session_id: str | None
    created_at: datetime


class ChatHistoryResponse(BaseModel):
    messages: list[ChatMessageResponse]


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/api/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    invoker: ClaudeInvoker = Depends(get_invoker),
    db: Database = Depends(get_db),
    tenant_id: str = Depends(get_authorized_tenant_id),
) -> ChatResponse:
    """Send a message to the agent and receive a complete response.

    Resumes today's trading session (if one ran) so the agent has full context
    of the day's portfolio activity. MCP tools are available for live data lookups.
    """
    # Persist the user's message
    session_id_before = invoker._get_daily_session_id(__import__("datetime").date.today())
    await db.save_chat_message(
        tenant_id=tenant_id,
        role="user",
        content=req.message,
        session_id=session_id_before,
    )

    result = await invoker.chat(req.message)

    if result.error:
        log.error("chat_error", error=result.error, tenant_id=tenant_id)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=result.error)

    # Persist the assistant's response
    await db.save_chat_message(
        tenant_id=tenant_id,
        role="assistant",
        content=result.content,
        session_id=result.session_id,
        tool_calls_json=json.dumps(result.tool_calls) if result.tool_calls else None,
    )

    return ChatResponse(
        content=result.content,
        session_id=result.session_id,
        tool_calls=result.tool_calls,
        num_turns=result.num_turns,
        duration_ms=result.duration_ms,
    )


@router.post("/api/chat/stream")
async def chat_stream(
    req: ChatRequest,
    invoker: ClaudeInvoker = Depends(get_invoker),
    db: Database = Depends(get_db),
    tenant_id: str = Depends(get_authorized_tenant_id),
) -> StreamingResponse:
    """Stream the agent's response as Server-Sent Events.

    The client should consume the SSE stream and display text events as they arrive.
    Tool use events allow the UI to show live "agent is checking portfolio..." indicators.
    The final "done" event signals stream completion.
    """
    from datetime import date as date_cls

    today = date_cls.today()
    session_id_before = invoker._get_daily_session_id(today)

    # Persist user message before streaming starts
    await db.save_chat_message(
        tenant_id=tenant_id,
        role="user",
        content=req.message,
        session_id=session_id_before,
    )

    async def generate():
        accumulated_text: list[str] = []
        tool_calls: list[dict] = []
        final_session_id: str | None = session_id_before

        try:
            async for event in invoker.chat_stream(req.message):
                yield f"data: {json.dumps(event, default=str)}\n\n"

                etype = event.get("type")
                if etype == "text":
                    accumulated_text.append(event.get("text", ""))
                elif etype == "tool_use":
                    tool_calls.append({"name": event.get("name", ""), "input": event.get("input", {})})
                elif etype == "done":
                    final_session_id = event.get("session_id") or final_session_id

        except Exception as e:
            log.error("chat_stream_error", error=str(e), tenant_id=tenant_id)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            # Persist assistant message at stream end
            content = "".join(accumulated_text)
            if content:
                try:
                    await db.save_chat_message(
                        tenant_id=tenant_id,
                        role="assistant",
                        content=content,
                        session_id=final_session_id,
                        tool_calls_json=json.dumps(tool_calls) if tool_calls else None,
                    )
                except Exception as e:
                    log.warning("chat_message_save_failed", error=str(e))

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/api/chat/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    days: int = Query(7, ge=1, le=30),
    db: Database = Depends(get_db),
    tenant_id: str = Depends(get_authorized_tenant_id),
) -> ChatHistoryResponse:
    """Return recent chat history for the current tenant.

    Messages are ordered oldest-first so the UI can render a conversation thread.
    """
    rows = await db.get_chat_messages(tenant_id=tenant_id, days=days)
    messages = [
        ChatMessageResponse(
            id=row.id,
            role=row.role,
            content=row.content,
            tool_calls=json.loads(row.tool_calls_json) if row.tool_calls_json else [],
            session_id=row.session_id,
            created_at=row.created_at,
        )
        for row in rows
    ]
    return ChatHistoryResponse(messages=messages)
