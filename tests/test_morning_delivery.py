"""Tests for morning queue delivery via Telegram."""

from unittest.mock import AsyncMock

from src.notifications.telegram_bot import TelegramNotifier


def _make_notifier() -> TelegramNotifier:
    notifier = TelegramNotifier(bot_token="test", chat_id="123")
    notifier.send_message = AsyncMock(return_value=True)
    return notifier


def _make_db(pending: list[dict] | None = None) -> AsyncMock:
    db = AsyncMock()
    db.get_pending_sentinel_actions = AsyncMock(return_value=pending or [])
    return db


class TestMorningDelivery:
    async def test_morning_delivery_formats_critical_first(self) -> None:
        pending = [
            {
                "id": 1, "action_type": "sell", "ticker": "AAPL",
                "reason": "Stop proximity 1.5%", "source": "afterhours_sentinel",
                "alert_level": "critical", "created_at": "2026-02-16T22:00:00",
            },
            {
                "id": 2, "action_type": "review", "ticker": "MSFT",
                "reason": "VIX elevated", "source": "afterhours_sentinel",
                "alert_level": "warning", "created_at": "2026-02-16T22:30:00",
            },
        ]
        notifier = _make_notifier()
        db = _make_db(pending)
        result = await notifier.deliver_morning_queue(db, "default")
        assert result is True
        notifier.send_message.assert_called_once()

        msg = notifier.send_message.call_args[0][0]
        # Critical should appear before warnings
        crit_pos = msg.find("CRITICAL")
        warn_pos = msg.find("WARNINGS")
        assert crit_pos < warn_pos
        assert "AAPL" in msg
        assert "MSFT" in msg

    async def test_morning_delivery_empty_queue_no_message(self) -> None:
        notifier = _make_notifier()
        db = _make_db([])
        result = await notifier.deliver_morning_queue(db, "default")
        assert result is False
        notifier.send_message.assert_not_called()

    async def test_morning_delivery_includes_commands(self) -> None:
        pending = [
            {
                "id": 1, "action_type": "review", "ticker": "TSLA",
                "reason": "Gap risk", "source": "gap_risk",
                "alert_level": "warning", "created_at": "2026-02-16T23:00:00",
            },
        ]
        notifier = _make_notifier()
        db = _make_db(pending)
        await notifier.deliver_morning_queue(db, "default")
        msg = notifier.send_message.call_args[0][0]
        assert "/execute-all" in msg
        assert "/cancel-all" in msg
        assert "/cancel N" in msg

    async def test_morning_delivery_warnings_only(self) -> None:
        pending = [
            {
                "id": 1, "action_type": "review", "ticker": "NVDA",
                "reason": "Earnings tonight", "source": "gap_risk",
                "alert_level": "warning", "created_at": "2026-02-16T23:00:00",
            },
        ]
        notifier = _make_notifier()
        db = _make_db(pending)
        await notifier.deliver_morning_queue(db, "default")
        msg = notifier.send_message.call_args[0][0]
        assert "0 critical" in msg
        assert "1 warnings" in msg
