"""SSE endpoints for real-time event streaming.

GET /api/events/stream     — SSE stream (tenant-scoped via JWT)
GET /api/events/recent     — Recent events for catch-up on reconnect
GET /api/events/connections — Active SSE connections (admin only)
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from config.settings import settings
from src.api.deps import get_authorized_tenant_id, get_current_user, require_admin
from src.events.event_bus import Event, EventType, event_bus

log = structlog.get_logger()

router = APIRouter(prefix="/api/events", tags=["events"])


async def _sse_generator(
    sub_id: str,
    queue: asyncio.Queue[Event],
    heartbeat_s: float,
) -> Any:
    """Yield SSE-formatted events from the subscriber queue."""
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=heartbeat_s)
                yield event.to_sse()
            except asyncio.TimeoutError:
                heartbeat = Event(type=EventType.HEARTBEAT, tenant_id="", data={"status": "ok"})
                yield heartbeat.to_sse()
    finally:
        event_bus.unsubscribe(sub_id)
        log.debug("sse_client_disconnected", sub_id=sub_id)


@router.get("/stream")
async def stream_events(
    tenant_id: str = Depends(get_authorized_tenant_id),
    user: dict[str, str | None] = Depends(get_current_user),
) -> StreamingResponse:
    """SSE event stream. Tenant users receive only their events; admins receive all."""
    # Admins (no tenant_id in JWT) get all events
    scope = None if user.get("tenant_id") is None else tenant_id
    sub_id, queue = event_bus.subscribe(tenant_id=scope)

    log.info("sse_client_connected", sub_id=sub_id, tenant_id=scope)

    return StreamingResponse(
        _sse_generator(sub_id, queue, settings.sse_heartbeat_s),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx hint
        },
    )


@router.get("/recent")
async def recent_events(
    tenant_id: str = Depends(get_authorized_tenant_id),
    limit: int = Query(default=50, ge=1, le=100),
) -> list[dict[str, Any]]:
    """Return recent events for catch-up after reconnect."""
    events = event_bus.get_recent(tenant_id=tenant_id, limit=limit)
    return [
        {
            "id": e.id,
            "type": e.type.value,
            "data": e.data,
            "timestamp": e.timestamp,
        }
        for e in events
    ]


@router.get("/connections")
async def active_connections(
    _admin: dict[str, str | None] = Depends(require_admin),
) -> dict[str, Any]:
    """Admin-only: list active SSE connections."""
    connections = event_bus.get_connections()
    return {
        "total": len(connections),
        "connections": connections,
    }
