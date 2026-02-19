"""Tests for QuietHoursManager and Telegram quiet hours integration."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

from src.notifications.quiet_hours import QuietHoursManager


def _make_tenant(
    quiet_start: str = "21:00",
    quiet_end: str = "07:00",
    tz: str = "America/Mexico_City",
) -> MagicMock:
    tenant = MagicMock()
    tenant.quiet_hours_start = quiet_start
    tenant.quiet_hours_end = quiet_end
    tenant.quiet_hours_timezone = tz
    return tenant


def _make_db(tenant: MagicMock | None = None) -> AsyncMock:
    db = AsyncMock()
    db.get_tenant = AsyncMock(return_value=tenant or _make_tenant())
    db.save_sentinel_action = AsyncMock(return_value=1)
    db.get_pending_sentinel_actions = AsyncMock(return_value=[])
    db.resolve_sentinel_action = AsyncMock(return_value=True)
    return db


class TestIsQuiet:
    async def test_is_quiet_during_quiet_hours(self) -> None:
        db = _make_db(_make_tenant("21:00", "07:00", "America/Mexico_City"))
        mgr = QuietHoursManager(db)
        # 22:00 MX = quiet
        with patch("src.notifications.quiet_hours.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 2, 16, 22, 0, tzinfo=ZoneInfo("America/Mexico_City"))
            result = await mgr.is_quiet("default")
        assert result is True

    async def test_is_not_quiet_during_active_hours(self) -> None:
        db = _make_db(_make_tenant("21:00", "07:00", "America/Mexico_City"))
        mgr = QuietHoursManager(db)
        # 14:00 MX = active (outside 21:00-07:00 window)
        with patch("src.notifications.quiet_hours.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 2, 16, 14, 0, tzinfo=ZoneInfo("America/Mexico_City"))
            result = await mgr.is_quiet("default")
        assert result is False

    async def test_quiet_hours_overnight_span(self) -> None:
        """21:00-07:00 spans midnight — 02:00 should be quiet."""
        db = _make_db(_make_tenant("21:00", "07:00", "America/Mexico_City"))
        mgr = QuietHoursManager(db)
        with patch("src.notifications.quiet_hours.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 2, 17, 2, 0, tzinfo=ZoneInfo("America/Mexico_City"))
            result = await mgr.is_quiet("default")
        assert result is True

    async def test_same_day_span(self) -> None:
        """09:00-17:00 is a same-day span."""
        db = _make_db(_make_tenant("09:00", "17:00", "America/Mexico_City"))
        mgr = QuietHoursManager(db)
        with patch("src.notifications.quiet_hours.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 2, 16, 12, 0, tzinfo=ZoneInfo("America/Mexico_City"))
            result = await mgr.is_quiet("default")
        assert result is True

    async def test_tenant_not_found_returns_false(self) -> None:
        db = AsyncMock()
        db.get_tenant = AsyncMock(return_value=None)
        mgr = QuietHoursManager(db)
        assert await mgr.is_quiet("nonexistent") is False

    async def test_dst_spring_forward_quiet(self) -> None:
        """2026-03-08 is US spring-forward (2 AM → 3 AM). 1:30 AM CST should still be quiet."""
        db = _make_db(_make_tenant("21:00", "07:00", "America/Chicago"))
        mgr = QuietHoursManager(db)
        # 1:30 AM CST on spring-forward day = within quiet window
        with patch("src.notifications.quiet_hours.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 8, 1, 30, tzinfo=ZoneInfo("America/Chicago"))
            result = await mgr.is_quiet("default")
        assert result is True

    async def test_dst_spring_forward_active_after(self) -> None:
        """After spring-forward, 8 AM CDT should be active."""
        db = _make_db(_make_tenant("21:00", "07:00", "America/Chicago"))
        mgr = QuietHoursManager(db)
        with patch("src.notifications.quiet_hours.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 8, 8, 0, tzinfo=ZoneInfo("America/Chicago"))
            result = await mgr.is_quiet("default")
        assert result is False

    async def test_dst_fall_back_quiet(self) -> None:
        """2026-11-01 is US fall-back (2 AM → 1 AM). 1:30 AM CST should be quiet."""
        db = _make_db(_make_tenant("21:00", "07:00", "America/Chicago"))
        mgr = QuietHoursManager(db)
        with patch("src.notifications.quiet_hours.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 11, 1, 1, 30, tzinfo=ZoneInfo("America/Chicago"))
            result = await mgr.is_quiet("default")
        assert result is True

    async def test_dst_fall_back_active_after(self) -> None:
        """After fall-back, 8 AM CST should be active."""
        db = _make_db(_make_tenant("21:00", "07:00", "America/Chicago"))
        mgr = QuietHoursManager(db)
        with patch("src.notifications.quiet_hours.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 11, 1, 8, 0, tzinfo=ZoneInfo("America/Chicago"))
            result = await mgr.is_quiet("default")
        assert result is False


class TestQueueNotification:
    async def test_queue_notification_creates_pending(self) -> None:
        db = _make_db()
        mgr = QuietHoursManager(db)
        action_id = await mgr.queue_notification(
            tenant_id="default",
            action_type="sell",
            ticker="AAPL",
            reason="Stop proximity",
            source="afterhours_sentinel",
            alert_level="critical",
        )
        assert action_id == 1
        db.save_sentinel_action.assert_called_once_with(
            tenant_id="default",
            action_type="sell",
            ticker="AAPL",
            reason="Stop proximity",
            source="afterhours_sentinel",
            alert_level="critical",
            status="pending",
        )


class TestMorningSummary:
    async def test_morning_summary_returns_pending_only(self) -> None:
        # Use a recent timestamp so the 24h filter doesn't exclude it
        from datetime import timedelta, timezone

        recent = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        pending = [
            {
                "id": 1,
                "action_type": "sell",
                "ticker": "AAPL",
                "reason": "Stop",
                "source": "ah",
                "alert_level": "critical",
                "created_at": recent,
            },
        ]
        db = _make_db()
        db.get_pending_sentinel_actions = AsyncMock(return_value=pending)
        mgr = QuietHoursManager(db)
        result = await mgr.get_morning_summary("default")
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"

    async def test_morning_summary_filters_stale_actions(self) -> None:
        """Actions older than 24h should be excluded from morning delivery."""
        from datetime import timedelta, timezone

        stale = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        recent = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        pending = [
            {
                "id": 1,
                "action_type": "sell",
                "ticker": "OLD",
                "reason": "Stale",
                "source": "ah",
                "alert_level": "critical",
                "created_at": stale,
            },
            {
                "id": 2,
                "action_type": "review",
                "ticker": "NEW",
                "reason": "Fresh",
                "source": "ah",
                "alert_level": "warning",
                "created_at": recent,
            },
        ]
        db = _make_db()
        db.get_pending_sentinel_actions = AsyncMock(return_value=pending)
        mgr = QuietHoursManager(db)
        result = await mgr.get_morning_summary("default")
        assert len(result) == 1
        assert result[0]["ticker"] == "NEW"


class TestResolveAction:
    async def test_resolve_action_executed(self) -> None:
        db = _make_db()
        mgr = QuietHoursManager(db)
        await mgr.resolve_action(1, "executed", "agent")
        db.resolve_sentinel_action.assert_called_once_with(1, "executed", "agent")

    async def test_resolve_action_cancelled(self) -> None:
        db = _make_db()
        mgr = QuietHoursManager(db)
        await mgr.resolve_action(2, "cancelled", "owner_telegram")
        db.resolve_sentinel_action.assert_called_once_with(2, "cancelled", "owner_telegram")


class TestSendMessageOrQueue:
    async def test_sends_during_active_hours(self) -> None:
        from src.notifications.telegram_bot import TelegramNotifier

        notifier = TelegramNotifier(bot_token="test", chat_id="123")
        notifier.send_message = AsyncMock(return_value=True)

        db = _make_db()
        with patch("src.notifications.quiet_hours.QuietHoursManager.is_quiet", return_value=False):
            result = await notifier.send_message_or_queue(
                db=db,
                tenant_id="default",
                message="Test alert",
            )
        assert result is True
        notifier.send_message.assert_called_once()

    async def test_queues_during_quiet_hours(self) -> None:
        from src.notifications.telegram_bot import TelegramNotifier

        notifier = TelegramNotifier(bot_token="test", chat_id="123")
        notifier.send_message = AsyncMock(return_value=True)

        db = _make_db()
        with patch("src.notifications.quiet_hours.QuietHoursManager.is_quiet", return_value=True):
            result = await notifier.send_message_or_queue(
                db=db,
                tenant_id="default",
                message="Test alert",
                ticker="AAPL",
                alert_level="warning",
                source="sentinel",
            )
        assert result is False
        notifier.send_message.assert_not_called()
        db.save_sentinel_action.assert_called_once()
