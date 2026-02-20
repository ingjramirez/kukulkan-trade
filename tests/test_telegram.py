"""Tests for Telegram notification system.

Tests message formatting, splitting, and send logic with mocked Bot API.
"""

from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from src.notifications.telegram_bot import (
    TelegramNotifier,
    _escape_html,
    _split_message,
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
                "total_value": 34000,
                "cash": 0,
                "top_ticker": "QQQ",
                "daily_return_pct": 1.5,
            },
            portfolio_b={
                "total_value": 67000,
                "cash": 1000,
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
                "total_value": 33424,
                "cash": 0,
                "top_ticker": "GDX",
                "daily_return_pct": None,
                "reason": "Holding momentum target GDX",
            },
            portfolio_b={
                "total_value": 66978,
                "cash": 1000,
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
                "total_value": 33000,
                "daily_return_pct": None,
                "reason": "Rebalancing to QQQ",
            },
            portfolio_b={
                "total_value": 66000,
                "daily_return_pct": None,
                "reasoning": "Buying QQQ",
            },
            proposed_trades=trades,
        )
        assert "Proposed Trades (1)" in msg
        assert "No Trades Today" not in msg

    def test_portfolio_a_disabled_skips_section(self) -> None:
        """When run_portfolio_a=False, Portfolio A section is skipped."""
        msg = format_daily_brief(
            brief_date=date(2026, 2, 5),
            regime="BULL",
            portfolio_a={"total_value": 33000, "daily_return_pct": 1.0},
            portfolio_b={
                "total_value": 66000,
                "daily_return_pct": -0.2,
                "reasoning": "Holding",
            },
            proposed_trades=[],
            run_portfolio_a=False,
            run_portfolio_b=True,
        )
        assert "Portfolio A" not in msg
        assert "Portfolio B" in msg
        # Combined should only include B
        assert "$66,000" in msg

    def test_portfolio_b_disabled_skips_section(self) -> None:
        """When run_portfolio_b=False, Portfolio B section is skipped."""
        msg = format_daily_brief(
            brief_date=date(2026, 2, 5),
            regime="BULL",
            portfolio_a={
                "total_value": 33000,
                "daily_return_pct": 1.0,
                "top_ticker": "QQQ",
            },
            portfolio_b={
                "total_value": 66000,
                "daily_return_pct": -0.2,
                "reasoning": "Holding",
            },
            proposed_trades=[],
            run_portfolio_a=True,
            run_portfolio_b=False,
        )
        assert "Portfolio A" in msg
        assert "Portfolio B" not in msg
        # Combined should only include A
        assert "$33,000" in msg


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

    async def test_send_message_api_failure_exhausts_retries(
        self,
        notifier: TelegramNotifier,
    ) -> None:
        mock_bot = AsyncMock()
        mock_bot.send_message.side_effect = Exception("API error")
        notifier._bot = mock_bot

        result = await notifier.send_message("Hello", max_retries=0)
        assert result is False
        assert mock_bot.send_message.call_count == 1

    async def test_send_message_retries_then_succeeds(
        self,
        notifier: TelegramNotifier,
    ) -> None:
        mock_bot = AsyncMock()
        mock_bot.send_message.side_effect = [
            Exception("Timed out"),
            None,  # success on retry
        ]
        notifier._bot = mock_bot

        result = await notifier.send_message("Hello", max_retries=1)
        assert result is True
        assert mock_bot.send_message.call_count == 2

    async def test_send_message_retries_all_fail(
        self,
        notifier: TelegramNotifier,
    ) -> None:
        mock_bot = AsyncMock()
        mock_bot.send_message.side_effect = Exception("ConnectError")
        notifier._bot = mock_bot

        result = await notifier.send_message("Hello", max_retries=2)
        assert result is False
        assert mock_bot.send_message.call_count == 3  # 1 initial + 2 retries

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
