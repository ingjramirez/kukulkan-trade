"""Tests for the in-memory event bus."""

from __future__ import annotations

import json

import pytest

from src.events.event_bus import Event, EventBus, EventType


@pytest.fixture
def bus() -> EventBus:
    return EventBus(max_queue=8, history_size=10)


# --- publish / subscribe ---


async def test_publish_delivers_to_subscriber(bus: EventBus):
    sub_id, queue = bus.subscribe(tenant_id="t1")
    event = Event(type=EventType.TRADE_EXECUTED, tenant_id="t1", data={"ticker": "AAPL"})
    delivered = bus.publish(event)

    assert delivered == 1
    received = queue.get_nowait()
    assert received.type == EventType.TRADE_EXECUTED
    assert received.data["ticker"] == "AAPL"
    bus.unsubscribe(sub_id)


async def test_tenant_scoping_filters_events(bus: EventBus):
    """Subscriber for t1 should not receive events for t2."""
    sub_id, queue = bus.subscribe(tenant_id="t1")
    event = Event(type=EventType.TRADE_EXECUTED, tenant_id="t2", data={})
    delivered = bus.publish(event)

    assert delivered == 0
    assert queue.empty()
    bus.unsubscribe(sub_id)


async def test_admin_subscriber_receives_all(bus: EventBus):
    """Admin (tenant_id=None) receives events from any tenant."""
    sub_id, queue = bus.subscribe(tenant_id=None)
    bus.publish(Event(type=EventType.TRADE_EXECUTED, tenant_id="t1", data={}))
    bus.publish(Event(type=EventType.TRADE_EXECUTED, tenant_id="t2", data={}))

    assert queue.qsize() == 2
    bus.unsubscribe(sub_id)


async def test_unsubscribe_stops_delivery(bus: EventBus):
    sub_id, queue = bus.subscribe(tenant_id="t1")
    bus.unsubscribe(sub_id)
    bus.publish(Event(type=EventType.TRADE_EXECUTED, tenant_id="t1", data={}))
    assert queue.empty()


async def test_queue_overflow_drops_events(bus: EventBus):
    """When queue is full, new events are silently dropped."""
    sub_id, queue = bus.subscribe(tenant_id="t1")
    # Fill the queue (max_queue=8)
    for i in range(10):
        bus.publish(Event(type=EventType.TRADE_EXECUTED, tenant_id="t1", data={"i": i}))

    assert queue.qsize() == 8
    bus.unsubscribe(sub_id)


# --- SSE formatting ---


def test_event_to_sse_format():
    event = Event(
        type=EventType.TRADE_EXECUTED,
        tenant_id="t1",
        data={"ticker": "AAPL"},
        id="abc123",
    )
    sse = event.to_sse()
    assert sse.startswith("id: abc123\n")
    assert "event: trade_executed\n" in sse
    data_line = [line for line in sse.split("\n") if line.startswith("data:")][0]
    payload = json.loads(data_line[len("data: ") :])
    assert payload["ticker"] == "AAPL"
    assert sse.endswith("\n\n")


# --- history ---


async def test_history_stores_events(bus: EventBus):
    for i in range(5):
        bus.publish(Event(type=EventType.TRADE_EXECUTED, tenant_id="t1", data={"i": i}))

    history = bus.get_recent(tenant_id="t1")
    assert len(history) == 5
    assert history[0].data["i"] == 0
    assert history[-1].data["i"] == 4


async def test_history_respects_max_size():
    bus = EventBus(history_size=3)
    for i in range(5):
        bus.publish(Event(type=EventType.TRADE_EXECUTED, tenant_id="t1", data={"i": i}))

    history = bus.get_recent()
    assert len(history) == 3
    assert history[0].data["i"] == 2  # oldest kept


async def test_history_filters_by_tenant(bus: EventBus):
    bus.publish(Event(type=EventType.TRADE_EXECUTED, tenant_id="t1", data={}))
    bus.publish(Event(type=EventType.TRADE_EXECUTED, tenant_id="t2", data={}))

    assert len(bus.get_recent(tenant_id="t1")) == 1
    assert len(bus.get_recent(tenant_id="t2")) == 1
    assert len(bus.get_recent()) == 2


# --- connections ---


async def test_get_connections(bus: EventBus):
    sub_id, _ = bus.subscribe(tenant_id="t1")
    connections = bus.get_connections()
    assert len(connections) == 1
    assert connections[0]["id"] == sub_id
    assert connections[0]["tenant_id"] == "t1"
    bus.unsubscribe(sub_id)


async def test_clear_resets_everything(bus: EventBus):
    bus.subscribe(tenant_id="t1")
    bus.publish(Event(type=EventType.TRADE_EXECUTED, tenant_id="t1", data={}))
    bus.clear()
    assert bus.subscriber_count == 0
    assert len(bus.get_recent()) == 0


async def test_subscriber_count(bus: EventBus):
    assert bus.subscriber_count == 0
    s1, _ = bus.subscribe(tenant_id="t1")
    s2, _ = bus.subscribe(tenant_id="t2")
    assert bus.subscriber_count == 2
    bus.unsubscribe(s1)
    assert bus.subscriber_count == 1
    bus.unsubscribe(s2)
