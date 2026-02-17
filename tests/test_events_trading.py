"""Tests for trading event producers (TRADE_EXECUTED, TRAILING_STOP_TRIGGERED)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.events.event_bus import EventType, event_bus


@pytest.fixture(autouse=True)
def _clean_bus():
    event_bus.clear()
    yield
    event_bus.clear()


def _make_trade(ticker: str, side: str, shares: int, price: float, portfolio: str):
    """Create a mock trade object."""
    trade = MagicMock()
    trade.ticker = ticker
    trade.side.value = side
    trade.shares = shares
    trade.price = price
    trade.portfolio.value = portfolio
    return trade


async def test_trade_executed_event_published():
    """Verify TRADE_EXECUTED events are published for each executed trade."""
    sub_id, queue = event_bus.subscribe(tenant_id="default")

    from src.events.event_bus import Event

    event_bus.publish(
        Event(
            type=EventType.TRADE_EXECUTED,
            tenant_id="default",
            data={"ticker": "AAPL", "side": "BUY", "shares": 10, "price": 150.0, "portfolio": "B"},
        )
    )

    assert not queue.empty()
    evt = queue.get_nowait()
    assert evt.type == EventType.TRADE_EXECUTED
    assert evt.data["ticker"] == "AAPL"
    assert evt.data["side"] == "BUY"
    event_bus.unsubscribe(sub_id)


async def test_trade_executed_multiple_trades():
    """Multiple trades produce multiple events."""
    sub_id, queue = event_bus.subscribe(tenant_id="default")

    from src.events.event_bus import Event

    for ticker in ("AAPL", "MSFT", "GOOG"):
        event_bus.publish(
            Event(
                type=EventType.TRADE_EXECUTED,
                tenant_id="default",
                data={"ticker": ticker, "side": "BUY", "shares": 5, "price": 100.0, "portfolio": "B"},
            )
        )

    assert queue.qsize() == 3
    event_bus.unsubscribe(sub_id)


async def test_trailing_stop_triggered_event():
    """TRAILING_STOP_TRIGGERED event carries stop details."""
    sub_id, queue = event_bus.subscribe(tenant_id="default")

    from src.events.event_bus import Event

    event_bus.publish(
        Event(
            type=EventType.TRAILING_STOP_TRIGGERED,
            tenant_id="default",
            data={"ticker": "AAPL", "price": 145.0, "stop_price": 146.0, "portfolio": "B"},
        )
    )

    evt = queue.get_nowait()
    assert evt.type == EventType.TRAILING_STOP_TRIGGERED
    assert evt.data["ticker"] == "AAPL"
    assert evt.data["stop_price"] == 146.0
    event_bus.unsubscribe(sub_id)


async def test_trade_events_in_history():
    """Trade events appear in the history buffer."""
    from src.events.event_bus import Event

    event_bus.publish(
        Event(
            type=EventType.TRADE_EXECUTED,
            tenant_id="default",
            data={"ticker": "TSLA"},
        )
    )
    history = event_bus.get_recent(tenant_id="default")
    assert len(history) == 1
    assert history[0].data["ticker"] == "TSLA"


async def test_trade_event_tenant_isolation():
    """Trade events for t1 are not visible to t2 subscriber."""
    sub_id, queue = event_bus.subscribe(tenant_id="t2")

    from src.events.event_bus import Event

    event_bus.publish(Event(type=EventType.TRADE_EXECUTED, tenant_id="t1", data={"ticker": "AAPL"}))

    assert queue.empty()
    event_bus.unsubscribe(sub_id)
