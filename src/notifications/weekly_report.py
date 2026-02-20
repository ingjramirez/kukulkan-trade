"""Weekly performance report sent via Telegram every Friday after market close."""

from datetime import date, timedelta

import structlog

from src.notifications.telegram_bot import TelegramNotifier
from src.storage.database import Database
from src.utils.allocations import DEFAULT_ALLOCATIONS, TenantAllocations

log = structlog.get_logger()


class WeeklyReporter:
    """Generates and sends weekly performance summaries."""

    def __init__(
        self,
        db: Database,
        notifier: TelegramNotifier,
        tenant_id: str = "default",
        allocations: TenantAllocations | None = None,
        run_portfolio_a: bool = True,
        run_portfolio_b: bool = True,
    ) -> None:
        self._db = db
        self._notifier = notifier
        self._tenant_id = tenant_id
        self._alloc = allocations or DEFAULT_ALLOCATIONS
        self._run_portfolio_a = run_portfolio_a
        self._run_portfolio_b = run_portfolio_b

    async def generate_and_send(self, report_date: date | None = None) -> str:
        """Generate weekly report and send via Telegram.

        Args:
            report_date: The Friday to report on. Defaults to today.

        Returns:
            The formatted report text.
        """
        report_date = report_date or date.today()
        week_start = report_date - timedelta(days=4)  # Monday

        log.info(
            "weekly_report_generating",
            week_start=str(week_start),
            week_end=str(report_date),
            tenant_id=self._tenant_id,
        )

        sections: list[str] = []
        sections.append("Weekly Report")
        sections.append(f"{week_start.strftime('%b %d')} - {report_date.strftime('%b %d, %Y')}")
        sections.append("")

        # Portfolio summaries (only enabled portfolios)
        active = self._active_portfolios()
        for portfolio_name in active:
            section = await self._portfolio_summary(
                portfolio_name,
                week_start,
                report_date,
            )
            sections.append(section)

        # Trades of the week
        trades_section = await self._trades_summary(week_start, report_date)
        sections.append(trades_section)

        # Claude's decisions summary
        agent_section = await self._agent_summary(week_start, report_date)
        sections.append(agent_section)

        # Drawdown status
        drawdown_section = await self._drawdown_status()
        sections.append(drawdown_section)

        report = "\n".join(sections)
        await self._notifier.send_message(report, parse_mode=None)
        log.info("weekly_report_sent")
        return report

    async def _portfolio_summary(
        self,
        portfolio_name: str,
        week_start: date,
        week_end: date,
    ) -> str:
        """Summarize a portfolio's weekly performance."""
        snapshots = await self._db.get_snapshots(
            portfolio_name,
            tenant_id=self._tenant_id,
        )
        if not snapshots:
            return f"Portfolio {portfolio_name}: No data yet"

        week_snapshots = [s for s in snapshots if week_start <= s.date <= week_end]

        if len(week_snapshots) < 2:
            latest = snapshots[-1]
            return f"Portfolio {portfolio_name}: ${latest.total_value:,.0f} (insufficient data for weekly return)"

        start_val = week_snapshots[0].total_value
        end_val = week_snapshots[-1].total_value
        week_return = ((end_val - start_val) / start_val) * 100

        # SPY benchmark for the same week
        spy_text = ""
        try:
            import yfinance as yf

            spy_data = yf.download(
                "SPY",
                start=str(week_start),
                end=str(week_end + timedelta(days=1)),
                progress=False,
            )
            if len(spy_data) >= 2:
                close = spy_data["Close"]
                if hasattr(close, "columns"):
                    close = close.iloc[:, 0]  # flatten multi-level column
                spy_start_price = float(close.iloc[0])
                spy_end_price = float(close.iloc[-1])
                spy_week_return = ((spy_end_price - spy_start_price) / spy_start_price) * 100
                alpha = week_return - spy_week_return
                spy_text = f"\n  vs SPY: {spy_week_return:+.2f}% | Alpha: {alpha:+.2f}%"
        except (ValueError, KeyError, IndexError, IOError) as e:
            log.warning("weekly_spy_benchmark_failed", error=str(e))

        # Count trades this week
        trades = await self._db.get_trades(portfolio_name, tenant_id=self._tenant_id)
        week_trades = [t for t in trades if week_start <= t.executed_at.date() <= week_end]

        return (
            f"Portfolio {portfolio_name}\n"
            f"  Value: ${end_val:,.0f} ({week_return:+.2f}% this week)"
            f"{spy_text}\n"
            f"  Trades: {len(week_trades)} this week\n"
            f"  Snapshots: {len(week_snapshots)} days"
        )

    def _active_portfolios(self) -> list[str]:
        """Return list of enabled portfolio names."""
        active: list[str] = []
        if self._run_portfolio_a:
            active.append("A")
        if self._run_portfolio_b:
            active.append("B")
        return active or ["A", "B"]  # fallback: show both if none set

    async def _trades_summary(
        self,
        week_start: date,
        week_end: date,
    ) -> str:
        """Find best and worst trades of the week across all portfolios."""
        lines = ["\nTrades of the Week"]

        all_trades = []
        for pname in self._active_portfolios():
            trades = await self._db.get_trades(pname, tenant_id=self._tenant_id)
            week_trades = [t for t in trades if week_start <= t.executed_at.date() <= week_end]
            all_trades.extend(week_trades)

        if not all_trades:
            lines.append("  No trades executed this week")
            return "\n".join(lines)

        lines.append(f"  Total: {len(all_trades)} trades")

        buys = [t for t in all_trades if t.side == "BUY"]
        sells = [t for t in all_trades if t.side == "SELL"]
        lines.append(f"  Buys: {len(buys)} | Sells: {len(sells)}")

        # Biggest trade by dollar amount
        biggest = max(all_trades, key=lambda t: t.shares * t.price)
        lines.append(
            f"  Biggest: {biggest.side} {biggest.shares:.0f}x {biggest.ticker} (${biggest.shares * biggest.price:,.0f})"
        )

        return "\n".join(lines)

    async def _agent_summary(
        self,
        week_start: date,
        week_end: date,
    ) -> str:
        """Summarize Claude's decisions for Portfolio B."""
        lines = ["\nAI Decisions (Portfolio B)"]

        decisions = await self._db.get_agent_decisions(
            limit=50,
            tenant_id=self._tenant_id,
        )
        week_decisions = [d for d in decisions if week_start <= d.date <= week_end]

        if not week_decisions:
            lines.append("  No AI decisions this week")
            return "\n".join(lines)

        lines.append(f"  Decisions: {len(week_decisions)}")

        return "\n".join(lines)

    async def _drawdown_status(self) -> str:
        """Current drawdown from peak for each portfolio."""
        lines = ["\nDrawdown Status"]

        for pname in self._active_portfolios():
            snapshots = await self._db.get_snapshots(pname, tenant_id=self._tenant_id)
            if not snapshots:
                continue

            values = [s.total_value for s in snapshots]
            peak = max(values)
            current = values[-1]
            drawdown = ((peak - current) / peak) * 100

            initial = self._alloc.for_portfolio(pname)
            total_return = ((current - initial) / initial) * 100

            lines.append(f"  Portfolio {pname}: {drawdown:.1f}% from peak ({total_return:+.1f}% total return)")

        return "\n".join(lines)
