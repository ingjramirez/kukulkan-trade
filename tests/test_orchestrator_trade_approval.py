"""Tests for large trade approval wiring in the orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.events.event_bus import EventType, event_bus
from src.storage.models import OrderSide, PortfolioName, TradeSchema


def _make_trade(ticker: str = "AAPL", shares: float = 100, price: float = 150.0) -> TradeSchema:
    return TradeSchema(
        ticker=ticker,
        side=OrderSide.BUY,
        shares=shares,
        price=price,
        total=shares * price,
        portfolio=PortfolioName.B,
        reason="test trade",
    )


@pytest.fixture(autouse=True)
def _clean_bus():
    event_bus.clear()
    yield
    event_bus.clear()


class TestRequestLargeTradeApproval:
    """Test the orchestrator's _request_large_trade_approval method."""

    async def test_no_notifier_auto_rejects(self) -> None:
        """Without Telegram, large trade is auto-rejected."""
        from src.orchestrator import Orchestrator

        mock_db = MagicMock()
        mock_notifier = MagicMock()
        mock_notifier._token = ""
        mock_notifier._chat_id = ""
        mock_executor = MagicMock()

        orch = Orchestrator.__new__(Orchestrator)
        orch._db = mock_db
        orch._notifier = mock_notifier
        orch._executor = mock_executor

        trade = _make_trade()
        result = await orch._request_large_trade_approval(trade, 15.0, "test reason", "default")
        assert result == "reject"

    async def test_approved_returns_approve(self) -> None:
        """Telegram approval returns 'approve'."""
        from src.orchestrator import Orchestrator

        mock_db = MagicMock()
        mock_notifier = MagicMock()
        mock_notifier._token = "test-token"
        mock_notifier._chat_id = "123"
        mock_notifier.send_large_trade_approval = AsyncMock(return_value=42)
        mock_notifier.wait_for_large_trade_approval = AsyncMock(return_value="approve")
        mock_executor = MagicMock()

        orch = Orchestrator.__new__(Orchestrator)
        orch._db = mock_db
        orch._notifier = mock_notifier
        orch._executor = mock_executor

        trade = _make_trade()
        result = await orch._request_large_trade_approval(trade, 15.0, "test reason", "default")
        assert result == "approve"

    async def test_rejected_returns_reject(self) -> None:
        """Telegram rejection returns 'reject'."""
        from src.orchestrator import Orchestrator

        mock_db = MagicMock()
        mock_notifier = MagicMock()
        mock_notifier._token = "test-token"
        mock_notifier._chat_id = "123"
        mock_notifier.send_large_trade_approval = AsyncMock(return_value=42)
        mock_notifier.wait_for_large_trade_approval = AsyncMock(return_value="reject")
        mock_executor = MagicMock()

        orch = Orchestrator.__new__(Orchestrator)
        orch._db = mock_db
        orch._notifier = mock_notifier
        orch._executor = mock_executor

        trade = _make_trade()
        result = await orch._request_large_trade_approval(trade, 15.0, "test reason", "default")
        assert result == "reject"

    async def test_send_failure_auto_rejects(self) -> None:
        """If Telegram send fails (returns None), auto-reject."""
        from src.orchestrator import Orchestrator

        mock_db = MagicMock()
        mock_notifier = MagicMock()
        mock_notifier._token = "test-token"
        mock_notifier._chat_id = "123"
        mock_notifier.send_large_trade_approval = AsyncMock(return_value=None)
        mock_executor = MagicMock()

        orch = Orchestrator.__new__(Orchestrator)
        orch._db = mock_db
        orch._notifier = mock_notifier
        orch._executor = mock_executor

        trade = _make_trade()
        result = await orch._request_large_trade_approval(trade, 15.0, "test reason", "default")
        assert result == "reject"

    async def test_sse_events_published(self) -> None:
        """Both TRADE_APPROVAL_REQUESTED and TRADE_APPROVAL_RESOLVED SSE events are published."""
        from src.orchestrator import Orchestrator

        mock_db = MagicMock()
        mock_notifier = MagicMock()
        mock_notifier._token = "test-token"
        mock_notifier._chat_id = "123"
        mock_notifier.send_large_trade_approval = AsyncMock(return_value=42)
        mock_notifier.wait_for_large_trade_approval = AsyncMock(return_value="approve")
        mock_executor = MagicMock()

        orch = Orchestrator.__new__(Orchestrator)
        orch._db = mock_db
        orch._notifier = mock_notifier
        orch._executor = mock_executor

        sub_id, queue = event_bus.subscribe(tenant_id="default")
        trade = _make_trade()
        await orch._request_large_trade_approval(trade, 15.0, "test reason", "default")

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        types = [e.type for e in events]
        assert EventType.TRADE_APPROVAL_REQUESTED in types
        assert EventType.TRADE_APPROVAL_RESOLVED in types

        # Check REQUESTED payload
        requested = next(e for e in events if e.type == EventType.TRADE_APPROVAL_REQUESTED)
        assert requested.data["ticker"] == "AAPL"
        assert requested.data["portfolio_pct"] == 15.0

        # Check RESOLVED payload
        resolved = next(e for e in events if e.type == EventType.TRADE_APPROVAL_RESOLVED)
        assert resolved.data["ticker"] == "AAPL"
        assert resolved.data["approved"] is True

        event_bus.unsubscribe(sub_id)

    async def test_sse_resolved_on_auto_reject(self) -> None:
        """TRADE_APPROVAL_RESOLVED with approved=False when no notifier."""
        from src.orchestrator import Orchestrator

        mock_db = MagicMock()
        mock_notifier = MagicMock()
        mock_notifier._token = ""
        mock_notifier._chat_id = ""
        mock_executor = MagicMock()

        orch = Orchestrator.__new__(Orchestrator)
        orch._db = mock_db
        orch._notifier = mock_notifier
        orch._executor = mock_executor

        sub_id, queue = event_bus.subscribe(tenant_id="default")
        trade = _make_trade()
        await orch._request_large_trade_approval(trade, 15.0, "test", "default")

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        resolved = next(e for e in events if e.type == EventType.TRADE_APPROVAL_RESOLVED)
        assert resolved.data["approved"] is False
        event_bus.unsubscribe(sub_id)
