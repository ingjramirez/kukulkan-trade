"""Tests for weekly performance report."""

from datetime import date, datetime
from unittest.mock import AsyncMock

import pytest

from src.notifications.weekly_report import WeeklyReporter
from src.storage.database import Database
from src.storage.models import (
    AgentDecisionRow,
    DailySnapshotRow,
    PortfolioRow,
    TradeRow,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
async def db():
    """In-memory database for testing."""
    test_db = Database(url="sqlite+aiosqlite:///:memory:")
    await test_db.init_db()
    await test_db.ensure_tenant("tenant-1")
    await test_db.ensure_tenant("tenant-2")
    await test_db.ensure_tenant("tenant-abc")
    yield test_db
    await test_db.close()


@pytest.fixture
def mock_notifier():
    """Mock TelegramNotifier that captures messages."""
    notifier = AsyncMock()
    notifier.send_message = AsyncMock(return_value=True)
    return notifier


@pytest.fixture
async def seeded_db(db):
    """Database with a week of realistic data."""
    async with db.session() as s:
        # Portfolios
        s.add(PortfolioRow(name="A", cash=30000.0, total_value=33450.0))
        s.add(PortfolioRow(name="B", cash=55000.0, total_value=65200.0))

        # Snapshots for the week (Mon-Fri)
        for i, d in enumerate(
            [
                date(2026, 2, 2),
                date(2026, 2, 3),
                date(2026, 2, 4),
                date(2026, 2, 5),
                date(2026, 2, 6),
            ]
        ):
            s.add(
                DailySnapshotRow(
                    portfolio="A",
                    date=d,
                    total_value=33000 + (i * 100),
                    cash=30000.0,
                    positions_value=3000 + (i * 100),
                    daily_return_pct=0.3,
                )
            )
            s.add(
                DailySnapshotRow(
                    portfolio="B",
                    date=d,
                    total_value=66000 - (i * 200),
                    cash=55000.0,
                    positions_value=11000 - (i * 200),
                    daily_return_pct=-0.3,
                )
            )

        # Trades
        s.add(
            TradeRow(
                portfolio="A",
                ticker="XLK",
                side="BUY",
                shares=50,
                price=200.0,
                total=10000.0,
                reason="Momentum",
                executed_at=datetime(2026, 2, 3, 15, 0),
            )
        )
        s.add(
            TradeRow(
                portfolio="B",
                ticker="MSFT",
                side="BUY",
                shares=20,
                price=400.0,
                total=8000.0,
                reason="AI: tech thesis",
                executed_at=datetime(2026, 2, 4, 10, 0),
            )
        )
        s.add(
            TradeRow(
                portfolio="B",
                ticker="GLD",
                side="SELL",
                shares=30,
                price=180.0,
                total=5400.0,
                reason="AI exit: gold weakness",
                executed_at=datetime(2026, 2, 5, 10, 0),
            )
        )

        # Agent decisions
        s.add(
            AgentDecisionRow(
                date=date(2026, 2, 3),
                reasoning="Tech rotation thesis",
                model_used="claude-sonnet-4-6",
                tokens_used=3000,
            )
        )
        s.add(
            AgentDecisionRow(
                date=date(2026, 2, 5),
                reasoning="Reducing gold exposure",
                model_used="claude-sonnet-4-6",
                tokens_used=2800,
            )
        )

        await s.commit()
    return db


# ── Tests ─────────────────────────────────────────────────────────────────────


async def test_generate_and_send_with_data(seeded_db, mock_notifier):
    """Full report generates with seeded data and sends via notifier."""
    reporter = WeeklyReporter(seeded_db, mock_notifier)
    report = await reporter.generate_and_send(date(2026, 2, 6))

    mock_notifier.send_message.assert_called_once()
    assert "Weekly Report" in report
    assert "Portfolio A" in report
    assert "Portfolio B" in report
    assert "Trades of the Week" in report
    assert "AI Decisions" in report
    assert "Drawdown Status" in report


async def test_report_calculates_return(seeded_db, mock_notifier):
    """Weekly return is calculated from first to last snapshot."""
    reporter = WeeklyReporter(seeded_db, mock_notifier)
    report = await reporter.generate_and_send(date(2026, 2, 6))

    # Portfolio A: 33000 -> 33400, ~+1.21%
    assert "+1.21%" in report
    # Portfolio B: 66000 -> 65200, ~-1.21%
    assert "-1.21%" in report


async def test_report_shows_trade_count(seeded_db, mock_notifier):
    """Trade count is shown per portfolio."""
    reporter = WeeklyReporter(seeded_db, mock_notifier)
    report = await reporter.generate_and_send(date(2026, 2, 6))

    assert "Total: 3 trades" in report
    assert "Buys: 2 | Sells: 1" in report


async def test_report_shows_biggest_trade(seeded_db, mock_notifier):
    """Biggest trade by dollar amount is highlighted."""
    reporter = WeeklyReporter(seeded_db, mock_notifier)
    report = await reporter.generate_and_send(date(2026, 2, 6))

    # XLK: 50 * 200 = $10,000 is the biggest
    assert "XLK" in report
    assert "$10,000" in report


async def test_report_shows_agent_decisions(seeded_db, mock_notifier):
    """Agent decision count and token usage appear in report."""
    reporter = WeeklyReporter(seeded_db, mock_notifier)
    report = await reporter.generate_and_send(date(2026, 2, 6))

    assert "Decisions: 2" in report
    assert "sonnet-4-6" in report
    assert "5,800" in report  # 3000 + 2800 tokens


async def test_report_shows_drawdown(seeded_db, mock_notifier):
    """Drawdown from peak and total return are shown."""
    reporter = WeeklyReporter(seeded_db, mock_notifier)
    report = await reporter.generate_and_send(date(2026, 2, 6))

    assert "from peak" in report
    assert "total return" in report


async def test_report_handles_no_snapshots(db, mock_notifier):
    """Report handles portfolios with no data gracefully."""
    reporter = WeeklyReporter(db, mock_notifier)
    report = await reporter.generate_and_send(date(2026, 2, 6))

    assert "No data yet" in report


async def test_report_handles_no_trades(db, mock_notifier):
    """Report handles weeks with no trades."""
    # Add minimal snapshots but no trades
    async with db.session() as s:
        for d in [date(2026, 2, 2), date(2026, 2, 6)]:
            s.add(
                DailySnapshotRow(
                    portfolio="A",
                    date=d,
                    total_value=33000.0,
                    cash=33000.0,
                    positions_value=0.0,
                )
            )
        await s.commit()

    reporter = WeeklyReporter(db, mock_notifier)
    report = await reporter.generate_and_send(date(2026, 2, 6))

    assert "No trades executed this week" in report


async def test_report_handles_no_agent_decisions(db, mock_notifier):
    """Report handles weeks with no AI decisions."""
    reporter = WeeklyReporter(db, mock_notifier)
    report = await reporter.generate_and_send(date(2026, 2, 6))

    assert "No AI decisions this week" in report


# ── Multi-Tenant Tests ──────────────────────────────────────────────────────


async def test_tenant_scoped_report(db, mock_notifier):
    """WeeklyReporter with tenant_id only sees that tenant's data."""
    tenant_id = "tenant-abc"

    async with db.session() as s:
        # Seed data for tenant-abc
        s.add(
            DailySnapshotRow(
                portfolio="A",
                date=date(2026, 2, 2),
                tenant_id=tenant_id,
                total_value=33000.0,
                cash=30000.0,
                positions_value=3000.0,
            )
        )
        s.add(
            DailySnapshotRow(
                portfolio="A",
                date=date(2026, 2, 6),
                tenant_id=tenant_id,
                total_value=33500.0,
                cash=30000.0,
                positions_value=3500.0,
            )
        )
        s.add(
            TradeRow(
                portfolio="A",
                ticker="XLK",
                side="BUY",
                shares=10,
                price=200.0,
                total=2000.0,
                reason="Momentum",
                executed_at=datetime(2026, 2, 3, 15, 0),
                tenant_id=tenant_id,
            )
        )
        await s.commit()

    reporter = WeeklyReporter(db, mock_notifier, tenant_id=tenant_id)
    report = await reporter.generate_and_send(date(2026, 2, 6))

    assert "Portfolio A" in report
    assert "XLK" in report or "1 this week" in report


async def test_report_skips_disabled_portfolio_a(seeded_db, mock_notifier):
    """WeeklyReporter with run_portfolio_a=False skips Portfolio A section."""
    reporter = WeeklyReporter(
        seeded_db,
        mock_notifier,
        run_portfolio_a=False,
        run_portfolio_b=True,
    )
    report = await reporter.generate_and_send(date(2026, 2, 6))

    assert "Portfolio B" in report
    assert "Portfolio A" not in report


async def test_report_skips_disabled_portfolio_b(seeded_db, mock_notifier):
    """WeeklyReporter with run_portfolio_b=False skips Portfolio B section."""
    reporter = WeeklyReporter(
        seeded_db,
        mock_notifier,
        run_portfolio_a=True,
        run_portfolio_b=False,
    )
    report = await reporter.generate_and_send(date(2026, 2, 6))

    assert "Portfolio A" in report
    # Portfolio B section header should not appear
    # (B trades and drawdowns are also skipped)
    lines = report.split("\n")
    portfolio_b_lines = [ln for ln in lines if ln.strip().startswith("Portfolio B")]
    assert len(portfolio_b_lines) == 0


async def test_tenant_isolation_in_reports(db, mock_notifier):
    """Data from one tenant does not leak into another tenant's report."""
    async with db.session() as s:
        # Seed data for tenant-1
        s.add(
            DailySnapshotRow(
                portfolio="A",
                date=date(2026, 2, 2),
                tenant_id="tenant-1",
                total_value=33000.0,
                cash=30000.0,
                positions_value=3000.0,
            )
        )
        s.add(
            DailySnapshotRow(
                portfolio="A",
                date=date(2026, 2, 6),
                tenant_id="tenant-1",
                total_value=34000.0,
                cash=30000.0,
                positions_value=4000.0,
            )
        )
        s.add(
            TradeRow(
                portfolio="B",
                ticker="NVDA",
                side="BUY",
                shares=5,
                price=800.0,
                total=4000.0,
                reason="AI: GPU demand",
                executed_at=datetime(2026, 2, 4, 10, 0),
                tenant_id="tenant-1",
            )
        )

        # Seed data for tenant-2 (should NOT appear in tenant-1 report)
        s.add(
            DailySnapshotRow(
                portfolio="A",
                date=date(2026, 2, 2),
                tenant_id="tenant-2",
                total_value=50000.0,
                cash=45000.0,
                positions_value=5000.0,
            )
        )
        s.add(
            DailySnapshotRow(
                portfolio="A",
                date=date(2026, 2, 6),
                tenant_id="tenant-2",
                total_value=48000.0,
                cash=45000.0,
                positions_value=3000.0,
            )
        )
        s.add(
            TradeRow(
                portfolio="B",
                ticker="AAPL",
                side="SELL",
                shares=100,
                price=150.0,
                total=15000.0,
                reason="AI: exit",
                executed_at=datetime(2026, 2, 5, 10, 0),
                tenant_id="tenant-2",
            )
        )
        await s.commit()

    # Generate report for tenant-1
    reporter1 = WeeklyReporter(db, mock_notifier, tenant_id="tenant-1")
    report1 = await reporter1.generate_and_send(date(2026, 2, 6))

    # tenant-1 should see NVDA trade but NOT AAPL
    assert "NVDA" in report1
    assert "AAPL" not in report1

    # tenant-1 portfolio value should be ~$34K, not $48K-$50K
    assert "$34,000" in report1
    assert "$48,000" not in report1
    assert "$50,000" not in report1
