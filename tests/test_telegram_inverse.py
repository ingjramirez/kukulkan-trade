"""Tests for Telegram inverse ETF approval methods."""

from unittest.mock import AsyncMock, MagicMock

from src.notifications.telegram_bot import TelegramNotifier
from src.storage.models import OrderSide, PortfolioName, TradeSchema


def _make_trade(ticker: str = "SH") -> TradeSchema:
    return TradeSchema(
        ticker=ticker,
        side=OrderSide.BUY,
        shares=100,
        price=15.0,
        total=1500.0,
        portfolio=PortfolioName.B,
        reason="hedge against correction",
    )


class TestSendInverseTradeApproval:
    async def test_sends_html_with_metadata(self) -> None:
        notifier = TelegramNotifier(bot_token="test", chat_id="123")
        mock_bot = MagicMock()
        mock_msg = MagicMock(message_id=42)
        mock_bot.send_message = AsyncMock(return_value=mock_msg)
        notifier._bot = mock_bot

        msg_id = await notifier.send_inverse_trade_approval(
            trade=_make_trade("SH"),
            regime="CORRECTION",
            request_id="abc123",
        )
        assert msg_id == 42

        call_kwargs = mock_bot.send_message.call_args.kwargs
        text = call_kwargs["text"]
        assert "SH" in text
        assert "Short S&amp;P 500" in text
        assert "CORRECTION" in text
        assert "decay" in text.lower()

    async def test_keyboard_has_approve_reject(self) -> None:
        notifier = TelegramNotifier(bot_token="test", chat_id="123")
        mock_bot = MagicMock()
        mock_msg = MagicMock(message_id=42)
        mock_bot.send_message = AsyncMock(return_value=mock_msg)
        notifier._bot = mock_bot

        await notifier.send_inverse_trade_approval(
            trade=_make_trade(),
            regime="CRISIS",
            request_id="abc123",
        )
        call_kwargs = mock_bot.send_message.call_args.kwargs
        keyboard = call_kwargs["reply_markup"]
        buttons = keyboard.inline_keyboard[0]
        button_texts = [b.text for b in buttons]
        assert "Approve" in button_texts
        assert "Reject" in button_texts

    async def test_no_chat_id_returns_none(self) -> None:
        notifier = TelegramNotifier(bot_token="test", chat_id="")
        msg_id = await notifier.send_inverse_trade_approval(
            trade=_make_trade(),
            regime="CORRECTION",
            request_id="abc123",
        )
        assert msg_id is None


class TestWaitForInverseApproval:
    async def test_timeout_returns_reject(self) -> None:
        notifier = TelegramNotifier(bot_token="test", chat_id="123")
        mock_bot = MagicMock()
        mock_bot.get_updates = AsyncMock(return_value=[])
        notifier._bot = mock_bot

        # Use very short timeout to make test fast
        result = await notifier.wait_for_inverse_approval("abc123", timeout_seconds=1)
        assert result == "reject"

    async def test_approve_response(self) -> None:
        notifier = TelegramNotifier(bot_token="test", chat_id="123")
        mock_bot = MagicMock()

        # Simulate callback query with approve
        mock_cb = MagicMock()
        mock_cb.data = "abc123:approve"
        mock_cb.message.chat_id = None
        mock_cb.message.chat.id = 123

        mock_update = MagicMock()
        mock_update.update_id = 1
        mock_update.callback_query = mock_cb

        mock_bot.get_updates = AsyncMock(return_value=[mock_update])
        notifier._bot = mock_bot

        result = await notifier.wait_for_inverse_approval("abc123", timeout_seconds=5)
        assert result == "approve"

    async def test_wrong_chat_id_ignored(self) -> None:
        notifier = TelegramNotifier(bot_token="test", chat_id="123")
        mock_bot = MagicMock()

        # Simulate callback from wrong chat
        mock_cb = MagicMock()
        mock_cb.data = "abc123:approve"
        mock_cb.message.chat_id = None
        mock_cb.message.chat.id = 999  # Wrong chat

        mock_update = MagicMock()
        mock_update.update_id = 1
        mock_update.callback_query = mock_cb

        mock_bot.get_updates = AsyncMock(return_value=[mock_update])
        notifier._bot = mock_bot

        result = await notifier.wait_for_inverse_approval("abc123", timeout_seconds=1)
        assert result == "reject"  # Timeout
