"""Tests for Telegram notification system.

Tests message formatting, splitting, and send logic with mocked Bot API.
"""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.complexity_detector import ComplexityResult
from src.notifications.telegram_bot import (
    TelegramNotifier,
    _escape_html,
    _split_message,
    format_approval_request,
    format_daily_brief,
    format_trade_confirmation,
)
from src.storage.models import OrderSide, PortfolioName, TradeSchema

# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_trade(
    ticker: str = "XLK",
    side: OrderSide = OrderSide.BUY,
    shares: float = 10,
    price: float = 200.0,
    portfolio: PortfolioName = PortfolioName.A,
    reason: str = "test trade",
) -> TradeSchema:
    return TradeSchema(
        ticker=ticker,
        side=side,
        shares=shares,
        price=price,
        total=shares * price,
        portfolio=portfolio,
        reason=reason,
    )


# ── HTML Escaping ────────────────────────────────────────────────────────────


class TestEscapeHtml:
    def test_escapes_ampersand(self) -> None:
        assert _escape_html("A & B") == "A &amp; B"

    def test_escapes_angle_brackets(self) -> None:
        assert _escape_html("<script>") == "&lt;script&gt;"

    def test_no_escape_needed(self) -> None:
        assert _escape_html("plain text") == "plain text"

    def test_combined(self) -> None:
        assert _escape_html("a < b & c > d") == "a &lt; b &amp; c &gt; d"


# ── Message Splitting ────────────────────────────────────────────────────────


class TestSplitMessage:
    def test_short_message_no_split(self) -> None:
        result = _split_message("hello", 4096)
        assert result == ["hello"]

    def test_splits_at_newline(self) -> None:
        msg = "line1\nline2\nline3"
        result = _split_message(msg, 12)
        assert len(result) == 2
        assert result[0] == "line1\nline2"
        assert result[1] == "line3"

    def test_splits_at_max_when_no_newline(self) -> None:
        msg = "a" * 100
        result = _split_message(msg, 30)
        assert len(result) == 4
        assert all(len(chunk) <= 30 for chunk in result)

    def test_exact_length_no_split(self) -> None:
        msg = "x" * 4096
        result = _split_message(msg, 4096)
        assert result == [msg]


# ── Daily Brief Formatting ───────────────────────────────────────────────────


class TestFormatDailyBrief:
    def test_basic_format(self) -> None:
        msg = format_daily_brief(
            brief_date=date(2026, 2, 5),
            regime="BULL",
            portfolio_a={
                "total_value": 34000, "cash": 0,
                "top_ticker": "QQQ", "daily_return_pct": 1.5,
            },
            portfolio_b={
                "total_value": 67000, "cash": 1000,
                "reasoning": "Bullish on tech",
                "daily_return_pct": -0.2,
            },
            proposed_trades=[],
        )
        assert "2026-02-05" in msg
        assert "Portfolio A" in msg
        assert "QQQ" in msg
        assert "$34,000" in msg

    def test_with_trades(self) -> None:
        trades = [_make_trade(ticker="QQQ", side=OrderSide.BUY, shares=5, price=500)]
        msg = format_daily_brief(
            brief_date=date(2026, 2, 5),
            regime="NEUTRAL",
            portfolio_a={"total_value": 33000, "daily_return_pct": None},
            portfolio_b={"total_value": 66000, "daily_return_pct": None, "reasoning": "N/A"},
            proposed_trades=trades,
        )
        assert "Proposed Trades (1)" in msg
        assert "QQQ" in msg
        assert "BUY" in msg

    def test_with_commentary(self) -> None:
        msg = format_daily_brief(
            brief_date=date(2026, 2, 5),
            regime="BEAR",
            portfolio_a={"total_value": 30000, "daily_return_pct": -2.0},
            portfolio_b={"total_value": 65000, "daily_return_pct": -0.5, "reasoning": "Defensive"},
            proposed_trades=[],
            commentary="Markets are volatile today.",
        )
        assert "Market Commentary" in msg
        assert "Markets are volatile today." in msg

    def test_session_label(self) -> None:
        msg = format_daily_brief(
            brief_date=date(2026, 2, 5),
            regime=None,
            portfolio_a={"total_value": 33000, "daily_return_pct": None},
            portfolio_b={"total_value": 66000, "daily_return_pct": None, "reasoning": "N/A"},
            proposed_trades=[],
            session="Morning",
        )
        assert "(Morning)" in msg

    def test_combined_total(self) -> None:
        msg = format_daily_brief(
            brief_date=date(2026, 2, 5),
            regime="ROTATION",
            portfolio_a={"total_value": 35000, "daily_return_pct": 1.0},
            portfolio_b={"total_value": 68000, "daily_return_pct": -0.3, "reasoning": "test"},
            proposed_trades=[],
        )
        assert "$103,000" in msg

    def test_no_trades_shows_reasons(self) -> None:
        msg = format_daily_brief(
            brief_date=date(2026, 2, 5),
            regime="BULL",
            portfolio_a={
                "total_value": 33424, "cash": 0,
                "top_ticker": "GDX", "daily_return_pct": None,
                "reason": "Holding momentum target GDX",
            },
            portfolio_b={
                "total_value": 66978, "cash": 1000,
                "reasoning": "Conservative hold — maintaining current positions",
                "daily_return_pct": None,
            },
            proposed_trades=[],
        )
        assert "No Trades Today" in msg
        assert "A: Holding momentum target GDX" in msg
        assert "B: Conservative hold" in msg

    def test_no_trades_no_reasons(self) -> None:
        """No-trade section still appears even without reason strings."""
        msg = format_daily_brief(
            brief_date=date(2026, 2, 5),
            regime=None,
            portfolio_a={"total_value": 33000, "daily_return_pct": None},
            portfolio_b={"total_value": 66000, "daily_return_pct": None, "reasoning": ""},
            proposed_trades=[],
        )
        assert "No Trades Today" in msg
        # No reason lines when empty
        assert "  A:" not in msg
        assert "  B:" not in msg

    def test_trades_present_hides_no_trade_section(self) -> None:
        """When trades exist, no-trade section must not appear."""
        trades = [_make_trade(ticker="QQQ", side=OrderSide.BUY, shares=5, price=500)]
        msg = format_daily_brief(
            brief_date=date(2026, 2, 5),
            regime="BULL",
            portfolio_a={
                "total_value": 33000, "daily_return_pct": None,
                "reason": "Rebalancing to QQQ",
            },
            portfolio_b={
                "total_value": 66000, "daily_return_pct": None,
                "reasoning": "Buying QQQ",
            },
            proposed_trades=trades,
        )
        assert "Proposed Trades (1)" in msg
        assert "No Trades Today" not in msg


# ── Trade Confirmation Formatting ────────────────────────────────────────────


class TestFormatTradeConfirmation:
    def test_single_trade(self) -> None:
        trades = [_make_trade(ticker="XLK", side=OrderSide.BUY, shares=10, price=200)]
        msg = format_trade_confirmation(trades)
        assert "Trades Executed (1)" in msg
        assert "BUY" in msg
        assert "XLK" in msg
        assert "$2,000" in msg

    def test_multiple_portfolios(self) -> None:
        trades = [
            _make_trade(ticker="XLK", portfolio=PortfolioName.A),
            _make_trade(ticker="XLF", portfolio=PortfolioName.B, side=OrderSide.SELL),
        ]
        msg = format_trade_confirmation(trades)
        assert "Portfolio A" in msg
        assert "Portfolio B" in msg

    def test_buy_sell_totals(self) -> None:
        trades = [
            _make_trade(side=OrderSide.BUY, shares=10, price=100),
            _make_trade(side=OrderSide.SELL, shares=5, price=200),
        ]
        msg = format_trade_confirmation(trades)
        assert "Total bought: $1,000" in msg
        assert "Total sold: $1,000" in msg

    def test_with_reason(self) -> None:
        trades = [_make_trade(reason="Momentum signal")]
        msg = format_trade_confirmation(trades)
        assert "Momentum signal" in msg


# ── TelegramNotifier ─────────────────────────────────────────────────────────


class TestTelegramNotifier:
    @pytest.fixture
    def notifier(self) -> TelegramNotifier:
        return TelegramNotifier(bot_token="test-token", chat_id="12345")

    async def test_send_message_success(self, notifier: TelegramNotifier) -> None:
        mock_bot = AsyncMock()
        notifier._bot = mock_bot

        result = await notifier.send_message("Hello")
        assert result is True
        mock_bot.send_message.assert_called_once()

    async def test_send_message_no_chat_id(self) -> None:
        notifier = TelegramNotifier(bot_token="test-token", chat_id="")
        result = await notifier.send_message("Hello")
        assert result is False

    async def test_send_message_api_failure(self, notifier: TelegramNotifier) -> None:
        mock_bot = AsyncMock()
        mock_bot.send_message.side_effect = Exception("API error")
        notifier._bot = mock_bot

        result = await notifier.send_message("Hello")
        assert result is False

    async def test_send_message_splits_long(self, notifier: TelegramNotifier) -> None:
        mock_bot = AsyncMock()
        notifier._bot = mock_bot

        long_msg = "line\n" * 2000  # ~10,000 chars
        result = await notifier.send_message(long_msg)
        assert result is True
        assert mock_bot.send_message.call_count >= 2

    async def test_send_daily_brief(self, notifier: TelegramNotifier) -> None:
        mock_bot = AsyncMock()
        notifier._bot = mock_bot

        result = await notifier.send_daily_brief(
            brief_date=date(2026, 2, 5),
            regime="BULL",
            portfolio_a={"total_value": 34000, "daily_return_pct": 1.0},
            portfolio_b={"total_value": 67000, "daily_return_pct": -0.1, "reasoning": "test"},
            proposed_trades=[],
        )
        assert result is True
        sent_text = mock_bot.send_message.call_args[1]["text"]
        assert "Kukulkan Daily Brief" in sent_text

    async def test_send_trade_confirmation_empty(self, notifier: TelegramNotifier) -> None:
        mock_bot = AsyncMock()
        notifier._bot = mock_bot

        result = await notifier.send_trade_confirmation([])
        assert result is True
        sent_text = mock_bot.send_message.call_args[1]["text"]
        assert "No trades executed" in sent_text

    async def test_send_trade_confirmation_with_trades(self, notifier: TelegramNotifier) -> None:
        mock_bot = AsyncMock()
        notifier._bot = mock_bot

        trades = [_make_trade()]
        result = await notifier.send_trade_confirmation(trades)
        assert result is True
        sent_text = mock_bot.send_message.call_args[1]["text"]
        assert "Trades Executed" in sent_text

    async def test_send_error(self, notifier: TelegramNotifier) -> None:
        mock_bot = AsyncMock()
        notifier._bot = mock_bot

        result = await notifier.send_error("Something went wrong")
        assert result is True
        sent_text = mock_bot.send_message.call_args[1]["text"]
        assert "Kukulkan Error" in sent_text
        assert "Something went wrong" in sent_text

    def test_bot_lazy_init(self) -> None:
        notifier = TelegramNotifier(bot_token="test-token", chat_id="12345")
        assert notifier._bot is None
        bot = notifier.bot
        assert bot is not None

    @patch("src.notifications.telegram_bot.settings")
    def test_bot_no_token_raises(self, mock_settings) -> None:
        mock_settings.telegram.bot_token = ""
        notifier = TelegramNotifier(bot_token="", chat_id="12345")
        with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
            _ = notifier.bot


# ── Approval Request Formatting ─────────────────────────────────────────────


class TestFormatApprovalRequest:
    def test_contains_score(self) -> None:
        complexity = ComplexityResult(
            score=65, should_escalate=True, signals=["VIX elevated at 32.0"]
        )
        msg = format_approval_request(complexity)
        assert "65/100" in msg

    def test_contains_signals(self) -> None:
        complexity = ComplexityResult(
            score=40,
            should_escalate=False,
            signals=["Drawdown 6.2% from peak", "Regime changed: BULL → BEAR"],
        )
        msg = format_approval_request(complexity)
        assert "Drawdown" in msg
        assert "Regime changed" in msg

    def test_html_escapes_signals(self) -> None:
        complexity = ComplexityResult(
            score=50, should_escalate=True, signals=["A<B & C>D"]
        )
        msg = format_approval_request(complexity)
        assert "&lt;" in msg
        assert "&amp;" in msg

    def test_header_present(self) -> None:
        complexity = ComplexityResult(score=50, should_escalate=True, signals=["test"])
        msg = format_approval_request(complexity)
        assert "Model Escalation Request" in msg


# ── Approval Request Sending ────────────────────────────────────────────────


class TestSendApprovalRequest:
    async def test_sends_with_keyboard(self) -> None:
        notifier = TelegramNotifier(bot_token="test-token", chat_id="12345")
        mock_bot = AsyncMock()
        mock_msg = MagicMock()
        mock_msg.message_id = 42
        mock_bot.send_message.return_value = mock_msg
        notifier._bot = mock_bot

        complexity = ComplexityResult(
            score=60, should_escalate=True, signals=["VIX elevated at 28.0"]
        )
        result = await notifier.send_approval_request(complexity, "req123")

        assert result == 42
        call_kwargs = mock_bot.send_message.call_args[1]
        assert call_kwargs["reply_markup"] is not None
        # Check keyboard has 3 buttons
        keyboard = call_kwargs["reply_markup"]
        assert len(keyboard.inline_keyboard) == 1
        assert len(keyboard.inline_keyboard[0]) == 3

    async def test_keyboard_callback_data(self) -> None:
        notifier = TelegramNotifier(bot_token="test-token", chat_id="12345")
        mock_bot = AsyncMock()
        mock_msg = MagicMock()
        mock_msg.message_id = 1
        mock_bot.send_message.return_value = mock_msg
        notifier._bot = mock_bot

        complexity = ComplexityResult(score=50, should_escalate=True, signals=["test"])
        await notifier.send_approval_request(complexity, "abc123")

        keyboard = mock_bot.send_message.call_args[1]["reply_markup"]
        buttons = keyboard.inline_keyboard[0]
        assert buttons[0].callback_data == "abc123:opus"
        assert buttons[1].callback_data == "abc123:sonnet"
        assert buttons[2].callback_data == "abc123:skip"

    async def test_returns_none_without_chat_id(self) -> None:
        notifier = TelegramNotifier(bot_token="test-token", chat_id="")
        complexity = ComplexityResult(score=50, should_escalate=True, signals=["test"])
        result = await notifier.send_approval_request(complexity, "req123")
        assert result is None

    async def test_returns_none_on_api_error(self) -> None:
        notifier = TelegramNotifier(bot_token="test-token", chat_id="12345")
        mock_bot = AsyncMock()
        mock_bot.send_message.side_effect = Exception("API error")
        notifier._bot = mock_bot

        complexity = ComplexityResult(score=50, should_escalate=True, signals=["test"])
        result = await notifier.send_approval_request(complexity, "req123")
        assert result is None


# ── Approval Polling ────────────────────────────────────────────────────────


class TestWaitForApproval:
    async def test_receives_opus_choice(self) -> None:
        notifier = TelegramNotifier(bot_token="test-token", chat_id="12345")
        mock_bot = AsyncMock()

        # Simulate callback query with opus choice from authorised chat
        mock_update = MagicMock()
        mock_update.update_id = 1
        mock_update.callback_query.data = "req123:opus"
        mock_update.callback_query.message.chat.id = 12345
        mock_update.callback_query.message.chat_id = None
        mock_bot.get_updates.return_value = [mock_update]
        notifier._bot = mock_bot

        result = await notifier.wait_for_approval("req123", timeout_seconds=5)
        assert result == "opus"

    async def test_receives_skip_choice(self) -> None:
        notifier = TelegramNotifier(bot_token="test-token", chat_id="12345")
        mock_bot = AsyncMock()

        mock_update = MagicMock()
        mock_update.update_id = 1
        mock_update.callback_query.data = "req123:skip"
        mock_update.callback_query.message.chat.id = 12345
        mock_update.callback_query.message.chat_id = None
        mock_bot.get_updates.return_value = [mock_update]
        notifier._bot = mock_bot

        result = await notifier.wait_for_approval("req123", timeout_seconds=5)
        assert result == "skip"

    async def test_ignores_unrelated_updates(self) -> None:
        notifier = TelegramNotifier(bot_token="test-token", chat_id="12345")
        mock_bot = AsyncMock()

        # First poll: unrelated update. Second poll: matching update.
        unrelated = MagicMock()
        unrelated.update_id = 1
        unrelated.callback_query.data = "other_req:opus"
        unrelated.callback_query.message.chat.id = 12345
        unrelated.callback_query.message.chat_id = None

        matching = MagicMock()
        matching.update_id = 2
        matching.callback_query.data = "req123:sonnet"
        matching.callback_query.message.chat.id = 12345
        matching.callback_query.message.chat_id = None

        mock_bot.get_updates.side_effect = [[unrelated], [matching]]
        notifier._bot = mock_bot

        result = await notifier.wait_for_approval("req123", timeout_seconds=10)
        assert result == "sonnet"

    async def test_timeout_defaults_to_sonnet(self) -> None:
        notifier = TelegramNotifier(bot_token="test-token", chat_id="12345")
        mock_bot = AsyncMock()
        mock_bot.get_updates.return_value = []  # No updates ever
        notifier._bot = mock_bot

        result = await notifier.wait_for_approval("req123", timeout_seconds=1)
        assert result == "sonnet"

    async def test_rejects_callback_from_wrong_chat(self) -> None:
        """Callbacks from unauthorized chat IDs must be ignored."""
        notifier = TelegramNotifier(bot_token="test-token", chat_id="12345")
        mock_bot = AsyncMock()

        # Attacker sends callback from a different chat
        attacker_update = MagicMock()
        attacker_update.update_id = 1
        attacker_update.callback_query.data = "req123:opus"
        attacker_update.callback_query.message.chat.id = 99999  # wrong chat
        attacker_update.callback_query.message.chat_id = None

        mock_bot.get_updates.return_value = [attacker_update]
        notifier._bot = mock_bot

        # Should timeout (ignore the attacker's callback) and default to sonnet
        result = await notifier.wait_for_approval("req123", timeout_seconds=1)
        assert result == "sonnet"

    async def test_rejects_ticker_callback_from_wrong_chat(self) -> None:
        """Ticker approval callbacks from unauthorized chat IDs must be ignored."""
        notifier = TelegramNotifier(bot_token="test-token", chat_id="12345")
        mock_bot = AsyncMock()

        attacker_update = MagicMock()
        attacker_update.update_id = 1
        attacker_update.callback_query.data = "req456:approve"
        attacker_update.callback_query.message.chat.id = 99999
        attacker_update.callback_query.message.chat_id = None

        mock_bot.get_updates.return_value = [attacker_update]
        notifier._bot = mock_bot

        result = await notifier.wait_for_ticker_approval("req456", timeout_seconds=1)
        assert result == "reject"
