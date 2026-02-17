"""In-memory async event bus for SSE real-time events.

Single-process pub/sub: producers call publish(), SSE endpoint subscribers
receive events via async queues. No external dependencies (no Redis).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """All event types pushed to the SSE stream."""

    # Trading events
    TRADE_EXECUTED = "trade_executed"
    TRAILING_STOP_TRIGGERED = "trailing_stop_triggered"
    TRADE_REJECTED = "trade_rejected"
    TRADE_APPROVAL_REQUESTED = "trade_approval_requested"
    TRADE_APPROVAL_RESOLVED = "trade_approval_resolved"

    # Alert events
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"
    SESSION_STARTED = "session_started"
    SESSION_COMPLETED = "session_completed"
    SESSION_SKIPPED = "session_skipped"
    POSTURE_CHANGED = "posture_changed"

    # Data refresh events
    POSITIONS_UPDATED = "positions_updated"
    PORTFOLIO_SNAPSHOT = "portfolio_snapshot"
    INTRADAY_UPDATE = "intraday_update"
    BUDGET_UPDATED = "budget_updated"
    WATCHLIST_UPDATED = "watchlist_updated"

    # Sentinel events
    SENTINEL_ALERT = "sentinel_alert"
    SENTINEL_ESCALATION = "sentinel_escalation"

    # System
    SYSTEM_ERROR = "system_error"
    IMPROVEMENT_REPORT = "improvement_report"
    HEARTBEAT = "heartbeat"


@dataclass(frozen=True)
class Event:
    """Immutable event published to the bus."""

    type: EventType
    tenant_id: str
    data: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)

    def to_sse(self) -> str:
        """Format as an SSE message string."""
        import json

        lines = [
            f"id: {self.id}",
            f"event: {self.type.value}",
            f"data: {json.dumps(self.data)}",
        ]
        return "\n".join(lines) + "\n\n"


@dataclass
class _Subscriber:
    """Internal: tracks a single SSE connection."""

    queue: asyncio.Queue[Event]
    tenant_id: str | None  # None = admin (receives all tenants)
    connected_at: float = field(default_factory=time.time)


class EventBus:
    """Async event bus: publish events, subscribe via async queues.

    Thread-safe for single-process async usage (all ops on the event loop).
    """

    def __init__(self, max_queue: int = 64, history_size: int = 100) -> None:
        self._subscribers: dict[str, _Subscriber] = {}
        self._max_queue = max_queue
        self._history: deque[Event] = deque(maxlen=history_size)

    def publish(self, event: Event) -> int:
        """Publish an event to all matching subscribers.

        Returns the number of subscribers that received the event.
        Drops the event for subscribers whose queues are full (non-blocking).
        """
        self._history.append(event)
        delivered = 0
        for sub in self._subscribers.values():
            if sub.tenant_id is not None and sub.tenant_id != event.tenant_id:
                continue
            try:
                sub.queue.put_nowait(event)
                delivered += 1
            except asyncio.QueueFull:
                pass  # drop for slow consumers
        return delivered

    def subscribe(self, tenant_id: str | None = None) -> tuple[str, asyncio.Queue[Event]]:
        """Create a new subscription. Returns (subscriber_id, queue).

        Args:
            tenant_id: Scope events to this tenant. None = admin (all events).
        """
        sub_id = uuid.uuid4().hex[:12]
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._max_queue)
        self._subscribers[sub_id] = _Subscriber(queue=queue, tenant_id=tenant_id)
        return sub_id, queue

    def unsubscribe(self, sub_id: str) -> None:
        """Remove a subscription."""
        self._subscribers.pop(sub_id, None)

    def get_connections(self) -> list[dict[str, Any]]:
        """Return info about active connections (for admin endpoint)."""
        now = time.time()
        return [
            {
                "id": sub_id,
                "tenant_id": sub.tenant_id,
                "connected_seconds": round(now - sub.connected_at),
                "queue_size": sub.queue.qsize(),
            }
            for sub_id, sub in self._subscribers.items()
        ]

    def get_recent(self, tenant_id: str | None = None, limit: int = 50) -> list[Event]:
        """Return recent events from history buffer, optionally filtered by tenant."""
        events = list(self._history)
        if tenant_id is not None:
            events = [e for e in events if e.tenant_id == tenant_id]
        return events[-limit:]

    def clear(self) -> None:
        """Clear all subscribers and history. Useful for testing."""
        self._subscribers.clear()
        self._history.clear()

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


# Module-level singleton
event_bus = EventBus()
