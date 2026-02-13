"""Tests for portfolio toggle rebalance lifecycle."""

from unittest.mock import AsyncMock

import pandas as pd
import pytest

from src.notifications.telegram_bot import TelegramNotifier
from src.orchestrator import Orchestrator
from src.storage.database import Database
from src.storage.models import TenantRow, TradeSchema

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


@pytest.fixture
def mock_notifier():
    notifier = AsyncMock(spec=TelegramNotifier)
    notifier._token = None
    notifier._chat_id = None
    notifier.send_message = AsyncMock(return_value=True)
    notifier.send_daily_brief = AsyncMock(return_value=True)
    return notifier


@pytest.fixture
def closes() -> pd.DataFrame:
    """Minimal closes DataFrame with a few tickers."""
    dates = pd.date_range("2026-02-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "AAPL": [180.0, 182.0, 181.0, 183.0, 185.0],
            "MSFT": [400.0, 402.0, 401.0, 403.0, 405.0],
            "TSLA": [250.0, 252.0, 251.0, 253.0, 255.0],
            "SPY": [500.0, 501.0, 500.5, 502.0, 503.0],
        },
        index=dates,
    )


def _make_mock_executor(db: Database, tenant_id: str = "test-tenant"):
    """Create a mock executor that simulates sells with tenant-scoped DB ops."""
    executor = AsyncMock()

    async def _execute_trades(trades: list[TradeSchema], **kwargs) -> list[TradeSchema]:
        executed = []
        for trade in trades:
            pname = trade.portfolio.value
            if trade.side.value == "SELL":
                portfolio = await db.get_portfolio(pname, tenant_id=tenant_id)
                if portfolio:
                    await db.upsert_position(
                        pname,
                        trade.ticker,
                        0,
                        trade.price,
                        tenant_id=tenant_id,
                    )
                    await db.upsert_portfolio(
                        pname,
                        cash=portfolio.cash + trade.total,
                        total_value=portfolio.total_value,
                        tenant_id=tenant_id,
                    )
                    # Re-read portfolio to get updated cash for next trade
                    executed.append(trade)
        return executed

    executor.execute_trades = AsyncMock(side_effect=_execute_trades)
    return executor


async def _create_tenant(
    db: Database,
    *,
    run_a: bool = True,
    run_b: bool = True,
    pending_rebalance: bool = True,
    initial_equity: float = 100_000.0,
    a_pct: float = 33.33,
    b_pct: float = 66.67,
) -> TenantRow:
    """Helper to create a tenant with portfolios and optional positions."""
    tenant = TenantRow(
        id="test-tenant",
        name="Test Tenant",
        is_active=True,
        strategy_mode="conservative",
        run_portfolio_a=run_a,
        run_portfolio_b=run_b,
        pending_rebalance=pending_rebalance,
        initial_equity=initial_equity,
        portfolio_a_pct=a_pct,
        portfolio_b_pct=b_pct,
        portfolio_a_cash=initial_equity * a_pct / 100,
        portfolio_b_cash=initial_equity * b_pct / 100,
        alpaca_api_key_enc="enc_key",
        alpaca_api_secret_enc="enc_secret",
        telegram_bot_token_enc="enc_token",
        telegram_chat_id_enc="enc_chat",
    )
    await db.create_tenant(tenant)
    return tenant


async def _seed_positions(
    db: Database,
    tenant_id: str = "test-tenant",
) -> None:
    """Add positions to both portfolios."""
    # Portfolio A: 50 shares AAPL @ $170 = $8,500 + $20K cash = $28,500
    await db.upsert_portfolio("A", cash=20_000.0, total_value=28_500.0, tenant_id=tenant_id)
    await db.upsert_position("A", "AAPL", shares=50, avg_price=170.0, tenant_id=tenant_id)

    # Portfolio B: 40 MSFT @ $390 + 20 TSLA @ $240 = $20,400 + $40K cash = $60,400
    await db.upsert_portfolio("B", cash=40_000.0, total_value=60_400.0, tenant_id=tenant_id)
    await db.upsert_position("B", "MSFT", shares=40, avg_price=390.0, tenant_id=tenant_id)
    await db.upsert_position("B", "TSLA", shares=20, avg_price=240.0, tenant_id=tenant_id)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestHandleRebalance:
    async def test_disable_b_liquidates_and_transfers_cash_to_a(
        self,
        db,
        mock_notifier,
        closes,
    ):
        """Disabling B sells B positions, gives all cash to A."""
        await _create_tenant(db, run_a=True, run_b=False)
        await _seed_positions(db)

        executor = _make_mock_executor(db)
        orch = Orchestrator(db, notifier=mock_notifier, executor=executor)
        result = await orch._handle_rebalance(
            "test-tenant",
            closes,
            run_portfolio_a=True,
            run_portfolio_b=False,
        )

        assert result is not None

        # B should be liquidated — 0 positions
        positions_b = await db.get_positions("B", tenant_id="test-tenant")
        active_b = [p for p in positions_b if p.shares > 0]
        assert len(active_b) == 0

        # All cash should be in A, none in B
        portfolio_a = await db.get_portfolio("A", tenant_id="test-tenant")
        portfolio_b = await db.get_portfolio("B", tenant_id="test-tenant")
        assert portfolio_b.cash == 0.0
        assert portfolio_a.cash > 0

    async def test_disable_a_liquidates_and_transfers_cash_to_b(
        self,
        db,
        mock_notifier,
        closes,
    ):
        """Disabling A sells A positions, gives all cash to B."""
        await _create_tenant(db, run_a=False, run_b=True)
        await _seed_positions(db)

        executor = _make_mock_executor(db)
        orch = Orchestrator(db, notifier=mock_notifier, executor=executor)
        result = await orch._handle_rebalance(
            "test-tenant",
            closes,
            run_portfolio_a=False,
            run_portfolio_b=True,
        )

        assert result is not None

        # A should be liquidated
        positions_a = await db.get_positions("A", tenant_id="test-tenant")
        active_a = [p for p in positions_a if p.shares > 0]
        assert len(active_a) == 0

        # All cash in B
        portfolio_a = await db.get_portfolio("A", tenant_id="test-tenant")
        portfolio_b = await db.get_portfolio("B", tenant_id="test-tenant")
        assert portfolio_a.cash == 0.0
        assert portfolio_b.cash > 0

    async def test_disable_both_liquidates_all(
        self,
        db,
        mock_notifier,
        closes,
    ):
        """Disabling both portfolios sells everything, zeros out cash."""
        await _create_tenant(db, run_a=False, run_b=False)
        await _seed_positions(db)

        executor = _make_mock_executor(db)
        orch = Orchestrator(db, notifier=mock_notifier, executor=executor)
        result = await orch._handle_rebalance(
            "test-tenant",
            closes,
            run_portfolio_a=False,
            run_portfolio_b=False,
        )

        assert result is not None

        # All positions liquidated
        for pname in ("A", "B"):
            positions = await db.get_positions(pname, tenant_id="test-tenant")
            active = [p for p in positions if p.shares > 0]
            assert len(active) == 0

        # Cash is zeroed
        portfolio_a = await db.get_portfolio("A", tenant_id="test-tenant")
        portfolio_b = await db.get_portfolio("B", tenant_id="test-tenant")
        assert portfolio_a.cash == 0.0
        assert portfolio_b.cash == 0.0

    async def test_enable_both_fresh_start(
        self,
        db,
        mock_notifier,
        closes,
    ):
        """Enabling both portfolios sells everything and splits cash by pct."""
        await _create_tenant(db, run_a=True, run_b=True)
        await _seed_positions(db)

        executor = _make_mock_executor(db)
        orch = Orchestrator(db, notifier=mock_notifier, executor=executor)
        result = await orch._handle_rebalance(
            "test-tenant",
            closes,
            run_portfolio_a=True,
            run_portfolio_b=True,
        )

        assert result is not None

        # All positions liquidated (fresh start)
        for pname in ("A", "B"):
            positions = await db.get_positions(pname, tenant_id="test-tenant")
            active = [p for p in positions if p.shares > 0]
            assert len(active) == 0

        # Cash split by percentage
        portfolio_a = await db.get_portfolio("A", tenant_id="test-tenant")
        portfolio_b = await db.get_portfolio("B", tenant_id="test-tenant")
        total = portfolio_a.cash + portfolio_b.cash
        assert total > 0
        # A should be ~33.33% and B ~66.67% of total
        assert abs(portfolio_a.cash / total - 0.3333) < 0.01
        assert abs(portfolio_b.cash / total - 0.6667) < 0.01

    async def test_enable_one_from_both_disabled(
        self,
        db,
        mock_notifier,
        closes,
    ):
        """Enabling one portfolio from both-off gives it all cash."""
        # Start with no positions (both were disabled), just cash
        await _create_tenant(db, run_a=False, run_b=True)
        await db.upsert_portfolio(
            "A",
            cash=50_000.0,
            total_value=50_000.0,
            tenant_id="test-tenant",
        )
        await db.upsert_portfolio(
            "B",
            cash=50_000.0,
            total_value=50_000.0,
            tenant_id="test-tenant",
        )

        executor = _make_mock_executor(db)
        orch = Orchestrator(db, notifier=mock_notifier, executor=executor)
        result = await orch._handle_rebalance(
            "test-tenant",
            closes,
            run_portfolio_a=False,
            run_portfolio_b=True,
        )

        assert result is not None
        portfolio_a = await db.get_portfolio("A", tenant_id="test-tenant")
        portfolio_b = await db.get_portfolio("B", tenant_id="test-tenant")
        assert portfolio_a.cash == 0.0
        assert portfolio_b.cash == 100_000.0

    async def test_no_rebalance_when_flag_false(
        self,
        db,
        mock_notifier,
        closes,
    ):
        """No rebalance when pending_rebalance=False."""
        await _create_tenant(db, run_a=True, run_b=True, pending_rebalance=False)
        await _seed_positions(db)

        executor = _make_mock_executor(db)
        orch = Orchestrator(db, notifier=mock_notifier, executor=executor)
        result = await orch._handle_rebalance(
            "test-tenant",
            closes,
            run_portfolio_a=True,
            run_portfolio_b=True,
        )

        assert result is None

        # Positions untouched
        positions_b = await db.get_positions("B", tenant_id="test-tenant")
        active_b = [p for p in positions_b if p.shares > 0]
        assert len(active_b) == 2

    async def test_no_rebalance_for_default_tenant(
        self,
        db,
        mock_notifier,
        closes,
    ):
        """Default tenant skips rebalance (no TenantRow for 'default')."""
        executor = _make_mock_executor(db)
        orch = Orchestrator(db, notifier=mock_notifier, executor=executor)
        result = await orch._handle_rebalance(
            "default",
            closes,
            run_portfolio_a=True,
            run_portfolio_b=True,
        )
        assert result is None

    async def test_rebalance_clears_flag(
        self,
        db,
        mock_notifier,
        closes,
    ):
        """After rebalance, pending_rebalance is set to False."""
        await _create_tenant(db, run_a=True, run_b=False)
        await _seed_positions(db)

        executor = _make_mock_executor(db)
        orch = Orchestrator(db, notifier=mock_notifier, executor=executor)
        await orch._handle_rebalance(
            "test-tenant",
            closes,
            run_portfolio_a=True,
            run_portfolio_b=False,
        )

        tenant = await db.get_tenant("test-tenant")
        assert tenant.pending_rebalance is False

    async def test_rebalance_updates_allocations(
        self,
        db,
        mock_notifier,
        closes,
    ):
        """Rebalance updates tenant initial_equity and cash fields."""
        await _create_tenant(db, run_a=True, run_b=False)
        await _seed_positions(db)

        executor = _make_mock_executor(db)
        orch = Orchestrator(db, notifier=mock_notifier, executor=executor)
        new_alloc = await orch._handle_rebalance(
            "test-tenant",
            closes,
            run_portfolio_a=True,
            run_portfolio_b=False,
        )

        assert new_alloc is not None
        tenant = await db.get_tenant("test-tenant")
        # All cash goes to A
        assert tenant.portfolio_b_cash == 0.0
        assert tenant.portfolio_a_cash > 0
        assert tenant.initial_equity > 0

    async def test_rebalance_telegram_notification(
        self,
        db,
        closes,
    ):
        """Telegram is notified of the rebalance when notifier is available."""
        notifier = AsyncMock(spec=TelegramNotifier)
        notifier._token = "test-token"
        notifier._chat_id = "test-chat"
        notifier.send_message = AsyncMock(return_value=True)

        await _create_tenant(db, run_a=True, run_b=False)
        await _seed_positions(db)

        executor = _make_mock_executor(db)
        orch = Orchestrator(db, notifier=notifier, executor=executor)
        await orch._handle_rebalance(
            "test-tenant",
            closes,
            run_portfolio_a=True,
            run_portfolio_b=False,
        )

        notifier.send_message.assert_called_once()
        msg = notifier.send_message.call_args[0][0]
        assert "rebalance" in msg.lower()
        assert "Portfolio B" in msg
