"""Telegram bot for daily briefs and trade approval.

Sends formatted market summaries and trade proposals.
Receives approval/rejection commands from the user.
"""

from __future__ import annotations

import asyncio
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.storage.models import DiscoveredTickerRow

import structlog
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode

from config.settings import settings
from src.agent.strategy_directives import STRATEGY_LABELS
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

    async def send_message(
        self,
        text: str,
        parse_mode: str | None = ParseMode.HTML,
        max_retries: int = 2,
    ) -> bool:
        """Send a text message to the configured chat.

        Args:
            text: Message text (supports HTML formatting).
            parse_mode: Telegram parse mode (HTML or Markdown).
            max_retries: Number of retry attempts on failure (default 2).

        Returns:
            True if sent successfully.
        """
        if not self._chat_id:
            log.warning("telegram_no_chat_id")
            return False

        chunks = _split_message(text, MAX_MSG_LEN)
        for attempt in range(1, max_retries + 2):  # 1 initial + max_retries
            try:
                for chunk in chunks:
                    await self.bot.send_message(
                        chat_id=self._chat_id,
                        text=chunk,
                        parse_mode=parse_mode,
                    )
                log.debug("telegram_message_sent", chunks=len(chunks))
                return True
            except Exception as e:
                if attempt <= max_retries:
                    delay = attempt * 3  # 3s, 6s
                    log.warning(
                        "telegram_send_retry",
                        attempt=attempt,
                        max_retries=max_retries,
                        delay=delay,
                        error=str(e),
                    )
                    await asyncio.sleep(delay)
                else:
                    log.error("telegram_send_failed", error=str(e), attempts=attempt)
                    return False
        return False  # unreachable, but satisfies type checker

    async def send_message_or_queue(
        self,
        db: object,
        tenant_id: str,
        message: str,
        action_type: str = "review",
        ticker: str = "",
        alert_level: str = "warning",
        source: str = "system",
    ) -> bool:
        """Send Telegram message, or queue if quiet hours.

        Returns True if sent immediately, False if queued.
        """
        from src.notifications.quiet_hours import QuietHoursManager

        quiet_mgr = QuietHoursManager(db)
        if await quiet_mgr.is_quiet(tenant_id):
            await quiet_mgr.queue_notification(
                tenant_id=tenant_id,
                action_type=action_type,
                ticker=ticker,
                reason=message,
                source=source,
                alert_level=alert_level,
            )
            log.info("quiet_hours_queued", tenant_id=tenant_id, ticker=ticker)
            return False
        await self.send_message(message)
        return True

    async def deliver_morning_queue(self, db: object, tenant_id: str) -> bool:
        """Deliver all queued notifications as a single morning summary.

        Returns True if a summary was sent, False if queue was empty.
        """
        from src.notifications.quiet_hours import QuietHoursManager

        quiet_mgr = QuietHoursManager(db)
        pending = await quiet_mgr.get_morning_summary(tenant_id)
        if not pending:
            return False

        critical = [a for a in pending if a["alert_level"] == "critical"]
        warnings = [a for a in pending if a["alert_level"] == "warning"]

        message = "Morning Queue — Items from overnight\n\n"

        if critical:
            message += "CRITICAL:\n"
            for i, action in enumerate(critical, 1):
                message += (
                    f"  {i}. [{action['action_type'].upper()}] {action['ticker']}"
                    f" — {action['reason'][:100]}\n"
                    f"     Source: {action['source']} | {action['created_at']}\n"
                )
        if warnings:
            message += "\nWARNINGS:\n"
            for i, action in enumerate(warnings, len(critical) + 1):
                message += f"  {i}. [{action['action_type'].upper()}] {action['ticker']} — {action['reason'][:100]}\n"

        message += (
            f"\nTotal: {len(critical)} critical, {len(warnings)} warnings\n\n"
            "Reply:\n"
            "  /execute-all — execute all queued actions at market open\n"
            "  /cancel-all — cancel all queued actions\n"
            "  /cancel N — cancel specific item by number\n"
            "  (Or wait — morning AI session will review and decide)"
        )
        await self.send_message(message, parse_mode=None)
        return True

    async def send_daily_brief(
        self,
        brief_date: date,
        regime: str | None,
        portfolio_a: dict,
        portfolio_b: dict,
        proposed_trades: list[TradeSchema],
        commentary: str = "",
        session: str = "",
        strategy_mode: str = "conservative",
        run_portfolio_a: bool = True,
        run_portfolio_b: bool = True,
        trailing_stop_alerts: list[dict] | None = None,
        agent_tool_summary: dict | None = None,
    ) -> bool:
        """Send the formatted daily brief.

        Args:
            brief_date: Date of the brief.
            regime: Current market regime.
            portfolio_a: Dict with keys: total_value, cash, top_ticker, daily_return_pct.
            portfolio_b: Dict with keys: total_value, cash, reasoning, daily_return_pct.
            proposed_trades: Trades proposed for today.
            commentary: AI-generated market commentary.
            session: Run label (e.g. "Morning", "Midday", "Closing").
            strategy_mode: Active strategy persona.
            run_portfolio_a: Whether Portfolio A is enabled.
            run_portfolio_b: Whether Portfolio B is enabled.

        Returns:
            True if sent successfully.
        """
        msg = format_daily_brief(
            brief_date=brief_date,
            regime=regime,
            portfolio_a=portfolio_a,
            portfolio_b=portfolio_b,
            proposed_trades=proposed_trades,
            commentary=commentary,
            session=session,
            strategy_mode=strategy_mode,
            run_portfolio_a=run_portfolio_a,
            run_portfolio_b=run_portfolio_b,
            trailing_stop_alerts=trailing_stop_alerts or [],
            agent_tool_summary=agent_tool_summary,
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
        text = f"⚠️ <b>Kukulkan Error</b>\n\n{_escape_html(error_msg)}"
        return await self.send_message(text)

    async def send_sentinel_alert(self, alerts: list[dict], max_level: str) -> bool:
        """Send a sentinel alert summary to Telegram.

        Args:
            alerts: List of alert dicts with level, check_type, ticker, message.
            max_level: Highest alert level ("clear", "warning", "critical").

        Returns:
            True if sent successfully.
        """
        if not alerts:
            return True

        icon = {"critical": "🔴", "warning": "🟡", "clear": "🟢"}.get(max_level, "⚪")
        header = f"{icon} <b>Sentinel {max_level.upper()}</b>\n"

        lines = [header]
        for a in alerts:
            level_icon = {"critical": "🔴", "warning": "🟡", "clear": "🟢"}.get(a.get("level", ""), "⚪")
            lines.append(f"{level_icon} <b>[{a.get('check_type', '')}]</b> {_escape_html(a.get('message', ''))}")

        text = "\n".join(lines)
        return await self.send_message(text)

    async def send_inverse_trade_approval(
        self,
        trade: "TradeSchema",
        regime: str | None,
        request_id: str,
    ) -> int | None:
        """Send an inverse ETF trade approval request with inline keyboard.

        Args:
            trade: The proposed inverse ETF trade.
            regime: Current market regime.
            request_id: Unique ID to match callback responses.

        Returns:
            Message ID if sent successfully, None otherwise.
        """
        if not self._chat_id:
            log.warning("telegram_no_chat_id")
            return None

        from config.universe import INVERSE_ETF_META

        meta = INVERSE_ETF_META.get(trade.ticker, {})
        benchmark = meta.get("benchmark", "?")
        description = meta.get("description", trade.ticker)

        text = (
            f"🛡️ <b>Inverse ETF Trade Approval</b>\n\n"
            f"Ticker: <b>{_escape_html(trade.ticker)}</b> ({_escape_html(description)})\n"
            f"Side: {trade.side.value}\n"
            f"Shares: {trade.shares:.0f} @ ${trade.price:.2f}\n"
            f"Total: ${trade.total:,.0f}\n"
            f"Benchmark: {benchmark}\n"
            f"Regime: {_escape_html(regime or 'Unknown')}\n\n"
            f"⚠️ Inverse ETFs decay over time. Plan exit within 3-5 days.\n\n"
            f"Approve this hedge trade?"
        )

        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("Approve", callback_data=f"{request_id}:approve"),
                    InlineKeyboardButton("Reject", callback_data=f"{request_id}:reject"),
                ]
            ]
        )

        try:
            msg = await self.bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
                reply_markup=keyboard,
            )
            log.info("inverse_approval_sent", request_id=request_id, ticker=trade.ticker)
            return msg.message_id
        except Exception as e:
            log.error("inverse_approval_failed", error=str(e))
            return None

    async def wait_for_inverse_approval(
        self,
        request_id: str,
        timeout_seconds: int = 300,
    ) -> str:
        """Poll for user's inverse trade approval response.

        Args:
            request_id: The request ID to match in callback_data.
            timeout_seconds: Max seconds to wait.

        Returns:
            "approve" or "reject". Defaults to "reject" on timeout (conservative).
        """
        elapsed = 0
        poll_interval = 2
        last_update_id: int | None = None

        while elapsed < timeout_seconds:
            try:
                kwargs: dict = {"timeout": 1}
                if last_update_id is not None:
                    kwargs["offset"] = last_update_id + 1

                updates = await self.bot.get_updates(**kwargs)
                for update in updates:
                    last_update_id = update.update_id
                    if update.callback_query and update.callback_query.data:
                        cb = update.callback_query
                        cb_chat = str(
                            getattr(cb.message, "chat_id", None)
                            or getattr(getattr(cb.message, "chat", None), "id", None)
                            or ""
                        )
                        if cb_chat != self._chat_id:
                            continue
                        data = cb.data
                        if data.startswith(f"{request_id}:"):
                            choice = data.split(":", 1)[1]
                            if choice in ("approve", "reject"):
                                log.info(
                                    "inverse_approval_received",
                                    request_id=request_id,
                                    choice=choice,
                                )
                                return choice
            except Exception as e:
                log.warning("inverse_approval_poll_error", error=str(e))

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        log.info("inverse_approval_timeout", request_id=request_id)
        return "reject"

    async def send_large_trade_approval(
        self,
        trade: TradeSchema,
        trade_pct: float,
        approval_reason: str,
        request_id: str,
    ) -> int | None:
        """Send a large trade approval request with inline keyboard.

        Args:
            trade: The proposed trade exceeding the threshold.
            trade_pct: Trade value as % of portfolio.
            approval_reason: Human-readable reason for the approval request.
            request_id: Unique ID to match callback responses.

        Returns:
            Message ID if sent successfully, None otherwise.
        """
        if not self._chat_id:
            log.warning("telegram_no_chat_id")
            return None

        text = (
            f"⚠️ <b>Large Trade Approval</b>\n\n"
            f"Portfolio {trade.portfolio.value} | "
            f"{trade.side.value} {trade.shares:.0f} <b>{_escape_html(trade.ticker)}</b>\n"
            f"Price: ~${trade.price:.2f} | Value: ${trade.total:,.0f}\n"
            f"Portfolio weight: {trade_pct:.1f}%\n\n"
            f"Reason: {_escape_html(trade.reason[:200])}\n\n"
            f"⏰ Auto-reject in {settings.trade_approval_timeout_s // 60} minutes"
        )

        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("Approve", callback_data=f"{request_id}:approve"),
                    InlineKeyboardButton("Reject", callback_data=f"{request_id}:reject"),
                ]
            ]
        )

        try:
            msg = await self.bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
                reply_markup=keyboard,
            )
            log.info("large_trade_approval_sent", request_id=request_id, ticker=trade.ticker)
            return msg.message_id
        except Exception as e:
            log.error("large_trade_approval_failed", error=str(e))
            return None

    async def wait_for_large_trade_approval(
        self,
        request_id: str,
        timeout_seconds: int = 300,
    ) -> str:
        """Poll for user's large trade approval response.

        Args:
            request_id: The request ID to match in callback_data.
            timeout_seconds: Max seconds to wait.

        Returns:
            "approve" or "reject". Defaults to "reject" on timeout.
        """
        elapsed = 0
        poll_interval = 2
        last_update_id: int | None = None

        while elapsed < timeout_seconds:
            try:
                kwargs: dict = {"timeout": 1}
                if last_update_id is not None:
                    kwargs["offset"] = last_update_id + 1

                updates = await self.bot.get_updates(**kwargs)
                for update in updates:
                    last_update_id = update.update_id
                    if update.callback_query and update.callback_query.data:
                        cb = update.callback_query
                        cb_chat = str(
                            getattr(cb.message, "chat_id", None)
                            or getattr(getattr(cb.message, "chat", None), "id", None)
                            or ""
                        )
                        if cb_chat != self._chat_id:
                            log.warning(
                                "large_trade_approval_rejected_wrong_chat",
                                expected=self._chat_id,
                                got=cb_chat,
                            )
                            continue
                        data = cb.data
                        if data.startswith(f"{request_id}:"):
                            choice = data.split(":", 1)[1]
                            if choice in ("approve", "reject"):
                                log.info(
                                    "large_trade_approval_received",
                                    request_id=request_id,
                                    choice=choice,
                                )
                                return choice
            except Exception as e:
                log.warning("large_trade_approval_poll_error", error=str(e))

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        log.info("large_trade_approval_timeout", request_id=request_id)
        return "reject"

    async def send_ticker_proposal(
        self,
        ticker_row: "DiscoveredTickerRow",
        request_id: str,
    ) -> int | None:
        """Send a ticker discovery approval request with inline keyboard.

        Args:
            ticker_row: DiscoveredTickerRow with ticker info.
            request_id: Unique ID to match callback responses.

        Returns:
            Message ID if sent successfully, None otherwise.
        """
        if not self._chat_id:
            log.warning("telegram_no_chat_id")
            return None

        text = format_ticker_proposal(ticker_row)
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("Approve", callback_data=f"{request_id}:approve"),
                    InlineKeyboardButton("Reject", callback_data=f"{request_id}:reject"),
                ]
            ]
        )

        try:
            msg = await self.bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
                reply_markup=keyboard,
            )
            log.info("ticker_proposal_sent", request_id=request_id, ticker=ticker_row.ticker)
            return msg.message_id
        except Exception as e:
            log.error("ticker_proposal_failed", error=str(e))
            return None

    async def wait_for_ticker_approval(
        self,
        request_id: str,
        timeout_seconds: int = 300,
    ) -> str:
        """Poll for user's ticker approval response.

        Args:
            request_id: The request ID to match in callback_data.
            timeout_seconds: Max seconds to wait.

        Returns:
            "approve" or "reject". Defaults to "reject" on timeout.
        """
        elapsed = 0
        poll_interval = 2
        last_update_id: int | None = None

        while elapsed < timeout_seconds:
            try:
                kwargs: dict = {"timeout": 1}
                if last_update_id is not None:
                    kwargs["offset"] = last_update_id + 1

                updates = await self.bot.get_updates(**kwargs)
                for update in updates:
                    last_update_id = update.update_id
                    if update.callback_query and update.callback_query.data:
                        cb = update.callback_query
                        cb_chat = str(
                            getattr(cb.message, "chat_id", None)
                            or getattr(getattr(cb.message, "chat", None), "id", None)
                            or ""
                        )
                        if cb_chat != self._chat_id:
                            log.warning(
                                "ticker_approval_rejected_wrong_chat",
                                expected=self._chat_id,
                                got=cb_chat,
                            )
                            continue
                        data = cb.data
                        if data.startswith(f"{request_id}:"):
                            choice = data.split(":", 1)[1]
                            if choice in ("approve", "reject"):
                                log.info(
                                    "ticker_approval_received",
                                    request_id=request_id,
                                    choice=choice,
                                )
                                return choice
            except Exception as e:
                log.warning("ticker_approval_poll_error", error=str(e))

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        log.info("ticker_approval_timeout", request_id=request_id)
        return "reject"


# ── Message formatting ───────────────────────────────────────────────────────


def format_daily_brief(
    brief_date: date,
    regime: str | None,
    portfolio_a: dict,
    portfolio_b: dict,
    proposed_trades: list[TradeSchema],
    commentary: str = "",
    session: str = "",
    strategy_mode: str = "conservative",
    run_portfolio_a: bool = True,
    run_portfolio_b: bool = True,
    trailing_stop_alerts: list[dict] | None = None,
    agent_tool_summary: dict | None = None,
    inverse_exposure: dict | None = None,
    inverse_hold_alerts: list[dict] | None = None,
) -> str:
    """Format the full daily brief as HTML.

    Args:
        brief_date: Date of the brief.
        regime: Current regime string.
        portfolio_a: Portfolio A summary dict.
        portfolio_b: Portfolio B summary dict.
        proposed_trades: Today's proposed trades.
        commentary: AI commentary text.
        session: Run label (e.g. "Morning", "Midday", "Closing").
        strategy_mode: Active strategy persona.
        run_portfolio_a: Whether Portfolio A is enabled.
        run_portfolio_b: Whether Portfolio B is enabled.

    Returns:
        HTML-formatted message string.
    """
    # Header
    session_label = f" ({session})" if session else ""
    strategy_label = STRATEGY_LABELS.get(strategy_mode, strategy_mode)
    regime_icons = {
        "BULL": "\U0001f7e2",  # green circle
        "BEAR": "\U0001f534",  # red circle
        "CORRECTION": "\U0001f7e0",  # orange circle
        "CRISIS": "\u26a0\ufe0f",  # warning sign
        "CONSOLIDATION": "\U0001f7e1",  # yellow circle
    }
    regime_label = ""
    if regime:
        icon = regime_icons.get(regime.upper(), "")
        regime_label = f" | Regime: {icon} {regime.upper()}"
    lines = [
        f"<b>Kukulkan Daily Brief — {brief_date.isoformat()}{session_label}</b>",
        f"Strategy: {strategy_label}{regime_label}",
        "",
    ]

    # Portfolio A (only if enabled)
    if run_portfolio_a:
        a_ret = portfolio_a.get("daily_return_pct")
        a_ret_str = f"{a_ret:+.2f}%" if a_ret is not None else "N/A"
        lines.extend(
            [
                "<b>Portfolio A</b> (Momentum)",
                f"  Value: ${portfolio_a.get('total_value', 0):,.0f} ({a_ret_str})",
                f"  Holding: {portfolio_a.get('top_ticker', 'cash')}",
                "",
            ]
        )

    # Portfolio B (only if enabled)
    if run_portfolio_b:
        b_ret = portfolio_b.get("daily_return_pct")
        b_ret_str = f"{b_ret:+.2f}%" if b_ret is not None else "N/A"
        lines.extend(
            [
                "<b>Portfolio B</b> (AI Autonomy)",
                f"  Value: ${portfolio_b.get('total_value', 0):,.0f} ({b_ret_str})",
                f"  AI: {_escape_html(portfolio_b.get('reasoning', 'N/A')[:150])}",
            ]
        )
        if agent_tool_summary:
            tools = agent_tool_summary.get("tools_used", 0)
            turns = agent_tool_summary.get("turns", 0)
            dur = agent_tool_summary.get("duration_ms", 0)
            dur_s = f" ({dur / 1000:.0f}s)" if dur else ""
            lines.append(f"  🤖 Investigation: {tools} tools across {turns} turns{dur_s}")
            posture = agent_tool_summary.get("declared_posture")
            if posture:
                lines.append(f"  🎯 Posture: {posture.capitalize()}")
        lines.append("")

    # Inverse exposure section
    if inverse_exposure and inverse_exposure.get("positions"):
        lines.append("<b>Inverse Exposure</b>")
        for pos in inverse_exposure["positions"]:
            hedge_tag = " [equity hedge]" if pos.get("equity_hedge") else " [rate hedge]"
            lines.append(f"  🛡️ {pos['ticker']}: ${pos['value']:,.0f} ({pos['pct']:.1f}%){hedge_tag}")
        net_eq = inverse_exposure.get("net_equity_pct", 0)
        lines.append(f"  Net equity exposure: {net_eq:.1f}%")
        lines.append("")

    # Inverse hold time alerts
    if inverse_hold_alerts:
        lines.append("<b>Inverse Hold Alerts</b>")
        for alert in inverse_hold_alerts:
            icon = "🔴" if alert["alert_level"] == "review" else "🟡"
            lines.append(f"  {icon} {_escape_html(alert['message'])}")
        lines.append("")

    # Total (only sum active portfolios)
    total = 0.0
    if run_portfolio_a:
        total += portfolio_a.get("total_value", 0)
    if run_portfolio_b:
        total += portfolio_b.get("total_value", 0)
    initial = 100_000.0  # Alpaca paper account starting balance
    total_ret = ((total - initial) / initial) * 100 if initial > 0 else 0
    lines.extend(
        [
            f"<b>Combined:</b> ${total:,.0f} ({total_ret:+.2f}%)",
            "",
        ]
    )

    # Proposed trades
    if proposed_trades:
        lines.append(f"<b>Proposed Trades ({len(proposed_trades)})</b>")
        for t in proposed_trades:
            emoji = "🟢" if t.side.value == "BUY" else "🔴"
            lines.append(f"  {emoji} {t.side.value} {t.shares:.0f}x {t.ticker} @ ${t.price:.2f} [{t.portfolio.value}]")
        lines.append("")
    else:
        a_reason = portfolio_a.get("reason", "")
        b_reason = portfolio_b.get("reasoning", "")
        lines.append("<b>No Trades Today</b>")
        if a_reason:
            lines.append(f"  A: {_escape_html(a_reason)}")
        if b_reason:
            lines.append(f"  B: {_escape_html(b_reason[:100])}")
        lines.append("")

    # Trailing stop alerts (before proposed trades)
    if trailing_stop_alerts:
        lines.append(f"<b>Trailing Stops Triggered ({len(trailing_stop_alerts)})</b>")
        for alert in trailing_stop_alerts:
            lines.append(f"  {alert['ticker']}: sold @ ${alert['price']:.2f}")
            lines.append(f"    Entry: ${alert['entry']:.2f} → Peak: ${alert['peak']:.2f}")
        lines.append("")

    # Commentary
    if commentary:
        lines.extend(
            [
                "<b>Market Commentary</b>",
                _escape_html(commentary),
            ]
        )

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
            lines.append(f"  {emoji} {t.side.value} {t.shares:.0f}x {t.ticker} @ ${t.price:.2f} = ${t.total:,.0f}")
            if t.reason:
                lines.append(f"     ↳ {_escape_html(t.reason)}")
        lines.append("")

    total_buy = sum(t.total for t in trades if t.side.value == "BUY")
    total_sell = sum(t.total for t in trades if t.side.value == "SELL")
    lines.append(f"Total bought: ${total_buy:,.0f} | Total sold: ${total_sell:,.0f}")

    return "\n".join(lines)


def format_ticker_proposal(ticker_row: "DiscoveredTickerRow") -> str:
    """Format a ticker discovery proposal as HTML.

    Args:
        ticker_row: DiscoveredTickerRow with ticker info.

    Returns:
        HTML-formatted message string.
    """
    mcap_str = f"${ticker_row.market_cap / 1e9:.1f}B" if ticker_row.market_cap else "N/A"
    lines = [
        "🔍 <b>New Ticker Proposal</b>",
        "",
        f"Ticker: <b>{_escape_html(ticker_row.ticker)}</b>",
        f"Sector: {_escape_html(ticker_row.sector or 'Unknown')}",
        f"Market Cap: {mcap_str}",
        f"Source: {_escape_html(ticker_row.source)}",
        f"Expires: {ticker_row.expires_at.isoformat()}",
        "",
        f"<b>Rationale:</b> {_escape_html(ticker_row.rationale or 'N/A')}",
        "",
        "Add to Portfolio B universe?",
    ]
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
