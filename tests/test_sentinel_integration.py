"""Integration tests for sentinel wiring: settings, EventType, Telegram formatting."""

from unittest.mock import AsyncMock, patch

from config.settings import Settings
from src.agent.sentinel import AlertLevel, SentinelAlert, SentinelResult
from src.events.event_bus import EventType


class TestSentinelSettings:
    def test_default_sentinel_enabled(self) -> None:
        s = Settings()
        assert s.sentinel_enabled is True

    def test_default_sentinel_interval(self) -> None:
        s = Settings()
        assert s.sentinel_interval_min == 30

    def test_default_max_escalations(self) -> None:
        s = Settings()
        assert s.sentinel_max_escalations_per_day == 2

    def test_default_escalation_cooldown(self) -> None:
        s = Settings()
        assert s.sentinel_escalation_cooldown_s == 1800


class TestSentinelEventTypes:
    def test_sentinel_alert_event_exists(self) -> None:
        assert EventType.SENTINEL_ALERT == "sentinel_alert"

    def test_sentinel_escalation_event_exists(self) -> None:
        assert EventType.SENTINEL_ESCALATION == "sentinel_escalation"

    def test_event_types_are_strings(self) -> None:
        assert isinstance(EventType.SENTINEL_ALERT.value, str)
        assert isinstance(EventType.SENTINEL_ESCALATION.value, str)


class TestTelegramSentinelAlert:
    @patch("src.notifications.telegram_bot.TelegramNotifier.send_message", new_callable=AsyncMock)
    async def test_send_sentinel_alert_warning(self, mock_send) -> None:
        from src.notifications.telegram_bot import TelegramNotifier

        mock_send.return_value = True
        notifier = TelegramNotifier(bot_token="test", chat_id="123")

        alerts = [
            {"level": "warning", "check_type": "stop_proximity", "ticker": "XLK", "message": "XLK near stop"},
        ]
        result = await notifier.send_sentinel_alert(alerts, "warning")

        assert result is True
        mock_send.assert_called_once()
        text = mock_send.call_args[0][0]
        assert "Sentinel WARNING" in text
        assert "stop_proximity" in text
        assert "XLK near stop" in text

    @patch("src.notifications.telegram_bot.TelegramNotifier.send_message", new_callable=AsyncMock)
    async def test_send_sentinel_alert_critical(self, mock_send) -> None:
        from src.notifications.telegram_bot import TelegramNotifier

        mock_send.return_value = True
        notifier = TelegramNotifier(bot_token="test", chat_id="123")

        alerts = [
            {"level": "critical", "check_type": "regime_shift", "ticker": "^VIX", "message": "VIX at 40"},
            {"level": "warning", "check_type": "stop_proximity", "ticker": "AAPL", "message": "AAPL near stop"},
        ]
        result = await notifier.send_sentinel_alert(alerts, "critical")

        assert result is True
        text = mock_send.call_args[0][0]
        assert "Sentinel CRITICAL" in text
        assert "VIX at 40" in text
        assert "AAPL near stop" in text

    @patch("src.notifications.telegram_bot.TelegramNotifier.send_message", new_callable=AsyncMock)
    async def test_send_sentinel_alert_empty_alerts(self, mock_send) -> None:
        from src.notifications.telegram_bot import TelegramNotifier

        notifier = TelegramNotifier(bot_token="test", chat_id="123")

        result = await notifier.send_sentinel_alert([], "clear")

        assert result is True
        mock_send.assert_not_called()

    @patch("src.notifications.telegram_bot.TelegramNotifier.send_message", new_callable=AsyncMock)
    async def test_alert_icons_match_level(self, mock_send) -> None:
        from src.notifications.telegram_bot import TelegramNotifier

        mock_send.return_value = True
        notifier = TelegramNotifier(bot_token="test", chat_id="123")

        alerts = [
            {"level": "critical", "check_type": "test", "ticker": "X", "message": "crit msg"},
            {"level": "warning", "check_type": "test", "ticker": "Y", "message": "warn msg"},
            {"level": "clear", "check_type": "test", "ticker": "Z", "message": "clear msg"},
        ]
        await notifier.send_sentinel_alert(alerts, "critical")

        text = mock_send.call_args[0][0]
        # Red circle for critical header
        assert "\U0001f534" in text  # 🔴
        # Yellow circle for warning
        assert "\U0001f7e1" in text  # 🟡
        # Green circle for clear
        assert "\U0001f7e2" in text  # 🟢


class TestSentinelResultSerialization:
    """Test that SentinelResult can be converted to dicts for SSE/Telegram."""

    def test_alerts_to_dicts(self) -> None:
        result = SentinelResult(
            alerts=[
                SentinelAlert(
                    level=AlertLevel.WARNING,
                    check_type="stop_proximity",
                    ticker="XLK",
                    message="XLK near stop",
                    details={"price": 100.0},
                ),
            ]
        )
        alert_dicts = [
            {
                "level": a.level.value,
                "check_type": a.check_type,
                "ticker": a.ticker,
                "message": a.message,
            }
            for a in result.alerts
        ]
        assert len(alert_dicts) == 1
        assert alert_dicts[0]["level"] == "warning"
        assert alert_dicts[0]["check_type"] == "stop_proximity"
