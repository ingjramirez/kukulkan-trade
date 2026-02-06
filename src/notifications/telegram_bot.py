"""Telegram bot for daily briefs and trade approval.

Sends formatted market summaries and trade proposals.
Receives approval/rejection commands from the user.
"""

import asyncio
from datetime import date

import structlog
from telegram import Bot, Update
from telegram.constants import ParseMode

from config.settings import settings
from src.storage.models import TradeSchema

log = structlog.get_logger()

# Max Telegram message length
MAX_MSG_LEN = 4096


class TelegramNotifier:
    """Sends notifications and receives trade approvals via Telegram."""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        self._token = bot_token or settings.telegram.bot_token
        self._chat_id = chat_id or settings.telegram.chat_id
        self._bot: Bot | None = None

    @property
    def bot(self) -> Bot:
        """Lazy-init Telegram bot."""
        if self._bot is None:
            if not self._token:
                raise ValueError("TELEGRAM_BOT_TOKEN not set")
            self._bot = Bot(token=self._token)
        return self._bot

    async def send_message(self, text: str, parse_mode: str | None = ParseMode.HTML) -> bool:
        """Send a text message to the configured chat.

        Args:
            text: Message text (supports HTML formatting).
            parse_mode: Telegram parse mode (HTML or Markdown).

        Returns:
            True if sent successfully.
        """
        if not self._chat_id:
            log.warning("telegram_no_chat_id")
            return False

        try:
            # Split long messages
            chunks = _split_message(text, MAX_MSG_LEN)
            for chunk in chunks:
                await self.bot.send_message(
                    chat_id=self._chat_id,
                    text=chunk,
                    parse_mode=parse_mode,
                )
            log.debug("telegram_message_sent", chunks=len(chunks))
            return True
        except Exception as e:
            log.error("telegram_send_failed", error=str(e))
            return False

    async def send_daily_brief(
        self,
        brief_date: date,
        regime: str | None,
        portfolio_a: dict,
        portfolio_b: dict,
        portfolio_c: dict,
        proposed_trades: list[TradeSchema],
        commentary: str = "",
    ) -> bool:
        """Send the formatted daily brief.

        Args:
            brief_date: Date of the brief.
            regime: Current market regime.
            portfolio_a: Dict with keys: total_value, cash, top_ticker, daily_return_pct.
            portfolio_b: Dict with keys: total_value, cash, selected, daily_return_pct.
            portfolio_c: Dict with keys: total_value, cash, reasoning, daily_return_pct.
            proposed_trades: Trades proposed for today.
            commentary: AI-generated market commentary.

        Returns:
            True if sent successfully.
        """
        msg = format_daily_brief(
            brief_date=brief_date,
            regime=regime,
            portfolio_a=portfolio_a,
            portfolio_b=portfolio_b,
            portfolio_c=portfolio_c,
            proposed_trades=proposed_trades,
            commentary=commentary,
        )
        return await self.send_message(msg)

    async def send_trade_confirmation(self, trades: list[TradeSchema]) -> bool:
        """Send confirmation after trades are executed.

        Args:
            trades: List of executed trades.

        Returns:
            True if sent successfully.
        """
        if not trades:
            return await self.send_message("No trades executed today.")

        msg = format_trade_confirmation(trades)
        return await self.send_message(msg)

    async def send_error(self, error_msg: str) -> bool:
        """Send an error notification.

        Args:
            error_msg: Error description.

        Returns:
            True if sent successfully.
        """
        text = f"⚠️ <b>Atlas Error</b>\n\n{_escape_html(error_msg)}"
        return await self.send_message(text)


# ── Message formatting ───────────────────────────────────────────────────────


def format_daily_brief(
    brief_date: date,
    regime: str | None,
    portfolio_a: dict,
    portfolio_b: dict,
    portfolio_c: dict,
    proposed_trades: list[TradeSchema],
    commentary: str = "",
) -> str:
    """Format the full daily brief as HTML.

    Args:
        brief_date: Date of the brief.
        regime: Current regime string.
        portfolio_a: Portfolio A summary dict.
        portfolio_b: Portfolio B summary dict.
        portfolio_c: Portfolio C summary dict.
        proposed_trades: Today's proposed trades.
        commentary: AI commentary text.

    Returns:
        HTML-formatted message string.
    """
    regime_emoji = {
        "BULL": "🟢", "ROTATION": "🔄", "NEUTRAL": "⚪", "BEAR": "🔴",
    }.get(regime or "", "❓")

    # Header
    lines = [
        f"<b>Atlas Daily Brief — {brief_date.isoformat()}</b>",
        f"Regime: {regime_emoji} <b>{regime or 'Unknown'}</b>",
        "",
    ]

    # Portfolio A
    a_ret = portfolio_a.get("daily_return_pct")
    a_ret_str = f"{a_ret:+.2f}%" if a_ret is not None else "N/A"
    lines.extend([
        "<b>Portfolio A</b> (Momentum)",
        f"  Value: ${portfolio_a.get('total_value', 0):,.0f} ({a_ret_str})",
        f"  Holding: {portfolio_a.get('top_ticker', 'cash')}",
        "",
    ])

    # Portfolio B
    b_ret = portfolio_b.get("daily_return_pct")
    b_ret_str = f"{b_ret:+.2f}%" if b_ret is not None else "N/A"
    selected = portfolio_b.get("selected", [])
    lines.extend([
        "<b>Portfolio B</b> (Sector Rotation)",
        f"  Value: ${portfolio_b.get('total_value', 0):,.0f} ({b_ret_str})",
        f"  Holdings: {', '.join(selected) if selected else 'cash'}",
        "",
    ])

    # Portfolio C
    c_ret = portfolio_c.get("daily_return_pct")
    c_ret_str = f"{c_ret:+.2f}%" if c_ret is not None else "N/A"
    lines.extend([
        "<b>Portfolio C</b> (AI Autonomy)",
        f"  Value: ${portfolio_c.get('total_value', 0):,.0f} ({c_ret_str})",
        f"  AI: {_escape_html(portfolio_c.get('reasoning', 'N/A')[:150])}",
        "",
    ])

    # Total
    total = (
        portfolio_a.get("total_value", 0)
        + portfolio_b.get("total_value", 0)
        + portfolio_c.get("total_value", 0)
    )
    initial = 99_999.0
    total_ret = ((total - initial) / initial) * 100 if initial > 0 else 0
    lines.extend([
        f"<b>Combined:</b> ${total:,.0f} ({total_ret:+.2f}%)",
        "",
    ])

    # Proposed trades
    if proposed_trades:
        lines.append(f"<b>Proposed Trades ({len(proposed_trades)})</b>")
        for t in proposed_trades:
            emoji = "🟢" if t.side.value == "BUY" else "🔴"
            lines.append(
                f"  {emoji} {t.side.value} {t.shares:.0f}x {t.ticker} "
                f"@ ${t.price:.2f} [{t.portfolio.value}]"
            )
        lines.append("")

    # Commentary
    if commentary:
        lines.extend([
            "<b>Market Commentary</b>",
            _escape_html(commentary),
        ])

    return "\n".join(lines)


def format_trade_confirmation(trades: list[TradeSchema]) -> str:
    """Format executed trade confirmation as HTML.

    Args:
        trades: List of executed trades.

    Returns:
        HTML-formatted confirmation message.
    """
    lines = [f"<b>Trades Executed ({len(trades)})</b>", ""]

    by_portfolio: dict[str, list[TradeSchema]] = {}
    for t in trades:
        by_portfolio.setdefault(t.portfolio.value, []).append(t)

    for portfolio in sorted(by_portfolio.keys()):
        lines.append(f"<b>Portfolio {portfolio}</b>")
        for t in by_portfolio[portfolio]:
            emoji = "🟢" if t.side.value == "BUY" else "🔴"
            lines.append(
                f"  {emoji} {t.side.value} {t.shares:.0f}x {t.ticker} "
                f"@ ${t.price:.2f} = ${t.total:,.0f}"
            )
            if t.reason:
                lines.append(f"     ↳ {_escape_html(t.reason)}")
        lines.append("")

    total_buy = sum(t.total for t in trades if t.side.value == "BUY")
    total_sell = sum(t.total for t in trades if t.side.value == "SELL")
    lines.append(f"Total bought: ${total_buy:,.0f} | Total sold: ${total_sell:,.0f}")

    return "\n".join(lines)


def _escape_html(text: str) -> str:
    """Escape HTML special characters for Telegram."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _split_message(text: str, max_len: int) -> list[str]:
    """Split a long message into chunks that fit Telegram's limit.

    Args:
        text: Full message text.
        max_len: Maximum characters per message.

    Returns:
        List of message chunks.
    """
    if len(text) <= max_len:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        # Try to split at a newline
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks
