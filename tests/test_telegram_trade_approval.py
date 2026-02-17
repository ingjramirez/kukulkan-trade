"""Tests for Telegram large trade approval methods."""

from unittest.mock import AsyncMock, MagicMock

from src.notifications.telegram_bot import TelegramNotifier
from src.storage.models import OrderSide, PortfolioName, TradeSchema


def _make_trade(ticker: str = "AAPL") -> TradeSchema:
    return TradeSchema(
        ticker=ticker,
        side=OrderSide.BUY,
        shares=100,
        price=150.0,
        total=15000.0,
        portfolio=PortfolioName.B,
        reason="strong momentum breakout",
    )


class TestSendLargeTradeApproval:
    async def test_sends_html_with_trade_details(self) -> None:
        notifier = TelegramNotifier(bot_token="test", chat_id="123")
        mock_bot = MagicMock()
        mock_msg = MagicMock(message_id=42)
        mock_bot.send_message = AsyncMock(return_value=mock_msg)
        notifier._bot = mock_bot

        msg_id = await notifier.send_large_trade_approval(
            trade=_make_trade("AAPL"),
            trade_pct=15.0,
            approval_reason="Trade is 15% of portfolio",
            request_id="abc123",
        )
        assert msg_id == 42

        call_kwargs = mock_bot.send_message.call_args.kwargs
        text = call_kwargs["text"]
        assert "AAPL" in text
        assert "BUY" in text
        assert "100" in text
        assert "15.0%" in text

    async def test_message_includes_reason(self) -> None:
        notifier = TelegramNotifier(bot_token="test", chat_id="123")
        mock_bot = MagicMock()
        mock_msg = MagicMock(message_id=42)
        mock_bot.send_message = AsyncMock(return_value=mock_msg)
        notifier._bot = mock_bot

        await notifier.send_large_trade_approval(
            trade=_make_trade(),
            trade_pct=12.5,
            approval_reason="test reason",
            request_id="abc123",
        )
        text = mock_bot.send_message.call_args.kwargs["text"]
        assert "strong momentum breakout" in text

    async def test_keyboard_has_approve_reject(self) -> None:
        notifier = TelegramNotifier(bot_token="test", chat_id="123")
        mock_bot = MagicMock()
        mock_msg = MagicMock(message_id=42)
        mock_bot.send_message = AsyncMock(return_value=mock_msg)
        notifier._bot = mock_bot

        await notifier.send_large_trade_approval(
            trade=_make_trade(),
            trade_pct=15.0,
            approval_reason="test",
            request_id="abc123",
        )
        keyboard = mock_bot.send_message.call_args.kwargs["reply_markup"]
        buttons = keyboard.inline_keyboard[0]
        button_texts = [b.text for b in buttons]
        assert "Approve" in button_texts
        assert "Reject" in button_texts
        # Verify callback data format
        callback_data = [b.callback_data for b in buttons]
        assert "abc123:approve" in callback_data
        assert "abc123:reject" in callback_data

    async def test_no_chat_id_returns_none(self) -> None:
        notifier = TelegramNotifier(bot_token="test", chat_id="")
        msg_id = await notifier.send_large_trade_approval(
            trade=_make_trade(),
            trade_pct=15.0,
            approval_reason="test",
            request_id="abc123",
        )
        assert msg_id is None

    async def test_send_failure_returns_none(self) -> None:
        notifier = TelegramNotifier(bot_token="test", chat_id="123")
        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock(side_effect=Exception("network error"))
        notifier._bot = mock_bot

        msg_id = await notifier.send_large_trade_approval(
            trade=_make_trade(),
            trade_pct=15.0,
            approval_reason="test",
            request_id="abc123",
        )
        assert msg_id is None


class TestWaitForLargeTradeApproval:
    async def test_timeout_returns_reject(self) -> None:
        notifier = TelegramNotifier(bot_token="test", chat_id="123")
        mock_bot = MagicMock()
        mock_bot.get_updates = AsyncMock(return_value=[])
        notifier._bot = mock_bot

        result = await notifier.wait_for_large_trade_approval("abc123", timeout_seconds=1)
        assert result == "reject"

    async def test_approve_response(self) -> None:
        notifier = TelegramNotifier(bot_token="test", chat_id="123")
        mock_bot = MagicMock()

        mock_cb = MagicMock()
        mock_cb.data = "abc123:approve"
        mock_cb.message.chat_id = None
        mock_cb.message.chat.id = 123

        mock_update = MagicMock()
        mock_update.update_id = 1
        mock_update.callback_query = mock_cb

        mock_bot.get_updates = AsyncMock(return_value=[mock_update])
        notifier._bot = mock_bot

        result = await notifier.wait_for_large_trade_approval("abc123", timeout_seconds=5)
        assert result == "approve"

    async def test_reject_response(self) -> None:
        notifier = TelegramNotifier(bot_token="test", chat_id="123")
        mock_bot = MagicMock()

        mock_cb = MagicMock()
        mock_cb.data = "abc123:reject"
        mock_cb.message.chat_id = None
        mock_cb.message.chat.id = 123

        mock_update = MagicMock()
        mock_update.update_id = 1
        mock_update.callback_query = mock_cb

        mock_bot.get_updates = AsyncMock(return_value=[mock_update])
        notifier._bot = mock_bot

        result = await notifier.wait_for_large_trade_approval("abc123", timeout_seconds=5)
        assert result == "reject"

    async def test_wrong_chat_id_ignored(self) -> None:
        notifier = TelegramNotifier(bot_token="test", chat_id="123")
        mock_bot = MagicMock()

        mock_cb = MagicMock()
        mock_cb.data = "abc123:approve"
        mock_cb.message.chat_id = None
        mock_cb.message.chat.id = 999  # Wrong chat

        mock_update = MagicMock()
        mock_update.update_id = 1
        mock_update.callback_query = mock_cb

        mock_bot.get_updates = AsyncMock(return_value=[mock_update])
        notifier._bot = mock_bot

        result = await notifier.wait_for_large_trade_approval("abc123", timeout_seconds=1)
        assert result == "reject"  # Timeout because wrong chat ignored
