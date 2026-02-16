"""Tests for the ticker discovery pipeline.

Tests validation, DB persistence, dynamic universe, expiry, tenant isolation,
and Telegram approval.
"""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.universe import FULL_UNIVERSE, get_dynamic_universe
from src.agent.ticker_discovery import (
    MAX_DYNAMIC_TICKERS,
    TickerDiscovery,
)
from src.notifications.telegram_bot import (
    TelegramNotifier,
    format_ticker_proposal,
)
from src.storage.database import Database
from src.storage.models import DiscoveredTickerRow

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
async def db():
    """In-memory database for testing."""
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    await database.ensure_tenant("tenant-1")
    await database.ensure_tenant("tenant-2")
    await database.ensure_tenant("t-a")
    await database.ensure_tenant("t-b")
    yield database
    await database.close()


@pytest.fixture
async def discovery(db: Database):
    """TickerDiscovery instance with in-memory DB."""
    return TickerDiscovery(db)


def _make_yf_info(
    market_cap: float = 50e9,
    avg_volume: float = 5_000_000,
    sector: str = "Technology",
    name: str = "Test Corp",
) -> dict:
    """Build a mock yfinance info dict."""
    return {
        "regularMarketPrice": 150.0,
        "marketCap": market_cap,
        "averageVolume": avg_volume,
        "sector": sector,
        "shortName": name,
    }


# ── Validation Tests ────────────────────────────────────────────────────────


class TestTickerValidation:
    def test_rejects_ticker_in_universe(self, discovery: TickerDiscovery) -> None:
        result = discovery.validate_ticker("XLK")
        assert not result.valid
        assert "Already in universe" in result.reason

    @patch("src.agent.ticker_discovery.yf.Ticker")
    def test_accepts_valid_ticker(self, mock_yf_cls, discovery: TickerDiscovery) -> None:
        mock_ticker = MagicMock()
        mock_ticker.info = _make_yf_info()
        mock_yf_cls.return_value = mock_ticker

        result = discovery.validate_ticker("PLTR")
        assert result.valid
        assert result.sector == "Technology"
        assert result.market_cap == 50e9

    @patch("src.agent.ticker_discovery.yf.Ticker")
    def test_rejects_low_market_cap(self, mock_yf_cls, discovery: TickerDiscovery) -> None:
        mock_ticker = MagicMock()
        mock_ticker.info = _make_yf_info(market_cap=500e6)
        mock_yf_cls.return_value = mock_ticker

        result = discovery.validate_ticker("SMALL")
        assert not result.valid
        assert "Market cap" in result.reason

    @patch("src.agent.ticker_discovery.yf.Ticker")
    def test_rejects_low_volume(self, mock_yf_cls, discovery: TickerDiscovery) -> None:
        mock_ticker = MagicMock()
        mock_ticker.info = _make_yf_info(avg_volume=50_000)
        mock_yf_cls.return_value = mock_ticker

        result = discovery.validate_ticker("LOWVOL")
        assert not result.valid
        assert "volume" in result.reason.lower()

    @patch("src.agent.ticker_discovery.yf.Ticker")
    def test_rejects_not_found(self, mock_yf_cls, discovery: TickerDiscovery) -> None:
        mock_ticker = MagicMock()
        mock_ticker.info = {"regularMarketPrice": None}
        mock_yf_cls.return_value = mock_ticker

        result = discovery.validate_ticker("FAKE")
        assert not result.valid
        assert "not found" in result.reason.lower()

    @patch("src.agent.ticker_discovery.yf.Ticker")
    def test_handles_yfinance_exception(self, mock_yf_cls, discovery: TickerDiscovery) -> None:
        mock_yf_cls.side_effect = Exception("Network error")
        result = discovery.validate_ticker("ERROR")
        assert not result.valid
        assert "Validation error" in result.reason

    def test_normalizes_ticker_case(self, discovery: TickerDiscovery) -> None:
        result = discovery.validate_ticker("xlk")
        assert not result.valid
        assert "Already in universe" in result.reason


# ── Proposal Tests ──────────────────────────────────────────────────────────


class TestProposeTicker:
    @patch("src.agent.ticker_discovery.yf.Ticker")
    async def test_propose_valid_ticker(self, mock_yf_cls, discovery: TickerDiscovery, db: Database) -> None:
        mock_ticker = MagicMock()
        mock_ticker.info = _make_yf_info(sector="Industrials", name="Palantir")
        mock_yf_cls.return_value = mock_ticker

        row = await discovery.propose_ticker(
            "PLTR",
            "Defense AI growth",
            today=date(2026, 2, 5),
        )
        assert row is not None
        assert row.ticker == "PLTR"
        assert row.status == "proposed"
        assert row.sector == "Industrials"
        assert row.expires_at == date(2026, 3, 7)
        assert row.tenant_id == "default"

        # Verify persisted
        stored = await db.get_discovered_ticker("PLTR")
        assert stored is not None
        assert stored.rationale == "Defense AI growth"

    @patch("src.agent.ticker_discovery.yf.Ticker")
    async def test_propose_with_tenant_id(self, mock_yf_cls, discovery: TickerDiscovery, db: Database) -> None:
        mock_ticker = MagicMock()
        mock_ticker.info = _make_yf_info()
        mock_yf_cls.return_value = mock_ticker

        row = await discovery.propose_ticker(
            "PLTR",
            "AI play",
            today=date(2026, 2, 5),
            tenant_id="tenant-1",
        )
        assert row is not None
        assert row.tenant_id == "tenant-1"

        # Not visible to default tenant
        default_row = await db.get_discovered_ticker("PLTR", tenant_id="default")
        assert default_row is None

        # Visible to tenant-1
        tenant_row = await db.get_discovered_ticker("PLTR", tenant_id="tenant-1")
        assert tenant_row is not None

    @patch("src.agent.ticker_discovery.yf.Ticker")
    async def test_rejects_duplicate_proposal(self, mock_yf_cls, discovery: TickerDiscovery) -> None:
        mock_ticker = MagicMock()
        mock_ticker.info = _make_yf_info()
        mock_yf_cls.return_value = mock_ticker

        await discovery.propose_ticker("PLTR", "first", today=date(2026, 2, 5))
        second = await discovery.propose_ticker("PLTR", "duplicate", today=date(2026, 2, 5))
        assert second is None

    async def test_rejects_static_universe_ticker(self, discovery: TickerDiscovery) -> None:
        row = await discovery.propose_ticker("XLK", "already in universe")
        assert row is None

    @patch("src.agent.ticker_discovery.yf.Ticker")
    async def test_rejects_when_limit_reached(self, mock_yf_cls, discovery: TickerDiscovery, db: Database) -> None:
        mock_ticker = MagicMock()
        mock_ticker.info = _make_yf_info()
        mock_yf_cls.return_value = mock_ticker

        # Fill up to the limit with approved tickers
        for i in range(MAX_DYNAMIC_TICKERS):
            await db.save_discovered_ticker(
                DiscoveredTickerRow(
                    tenant_id="default",
                    ticker=f"DYN{i}",
                    source="test",
                    rationale="test",
                    status="approved",
                    proposed_at=date(2026, 2, 1),
                    expires_at=date(2026, 3, 1),
                )
            )

        row = await discovery.propose_ticker("OVERFLOW", "too many", today=date(2026, 2, 5))
        assert row is None

    @patch("src.agent.ticker_discovery.yf.Ticker")
    async def test_tenant_isolation_same_ticker(self, mock_yf_cls, discovery: TickerDiscovery, db: Database) -> None:
        """Same ticker can be proposed independently for different tenants."""
        mock_ticker = MagicMock()
        mock_ticker.info = _make_yf_info()
        mock_yf_cls.return_value = mock_ticker

        row1 = await discovery.propose_ticker(
            "PLTR",
            "tenant A reason",
            today=date(2026, 2, 5),
            tenant_id="t-a",
        )
        row2 = await discovery.propose_ticker(
            "PLTR",
            "tenant B reason",
            today=date(2026, 2, 5),
            tenant_id="t-b",
        )
        assert row1 is not None
        assert row2 is not None
        assert row1.tenant_id == "t-a"
        assert row2.tenant_id == "t-b"

    @patch("src.agent.ticker_discovery.yf.Ticker")
    async def test_per_tenant_limit(self, mock_yf_cls, discovery: TickerDiscovery, db: Database) -> None:
        """Limit is per-tenant, not global."""
        mock_ticker = MagicMock()
        mock_ticker.info = _make_yf_info()
        mock_yf_cls.return_value = mock_ticker

        # Fill tenant-1 to the limit
        for i in range(MAX_DYNAMIC_TICKERS):
            await db.save_discovered_ticker(
                DiscoveredTickerRow(
                    tenant_id="tenant-1",
                    ticker=f"T1_{i}",
                    source="test",
                    rationale="test",
                    status="approved",
                    proposed_at=date(2026, 2, 1),
                    expires_at=date(2026, 3, 1),
                )
            )

        # tenant-1 is at limit
        row = await discovery.propose_ticker(
            "OVER",
            "too many",
            today=date(2026, 2, 5),
            tenant_id="tenant-1",
        )
        assert row is None

        # tenant-2 still has room
        row2 = await discovery.propose_ticker(
            "OVER",
            "has room",
            today=date(2026, 2, 5),
            tenant_id="tenant-2",
        )
        assert row2 is not None


# ── Active Tickers / Dynamic Universe ───────────────────────────────────────


class TestActiveTickers:
    async def test_empty_by_default(self, discovery: TickerDiscovery) -> None:
        tickers = await discovery.get_active_tickers()
        assert tickers == []

    async def test_returns_approved_only(self, discovery: TickerDiscovery, db: Database) -> None:
        for status in ("proposed", "approved", "rejected", "expired"):
            await db.save_discovered_ticker(
                DiscoveredTickerRow(
                    tenant_id="default",
                    ticker=f"T_{status.upper()}",
                    source="test",
                    rationale="test",
                    status=status,
                    proposed_at=date(2026, 2, 1),
                    expires_at=date(2026, 3, 1),
                )
            )

        tickers = await discovery.get_active_tickers()
        assert tickers == ["T_APPROVED"]

    async def test_returns_tenant_scoped(self, discovery: TickerDiscovery, db: Database) -> None:
        await db.save_discovered_ticker(
            DiscoveredTickerRow(
                tenant_id="t-a",
                ticker="PLTR",
                source="test",
                rationale="test",
                status="approved",
                proposed_at=date(2026, 2, 1),
                expires_at=date(2026, 3, 1),
            )
        )
        await db.save_discovered_ticker(
            DiscoveredTickerRow(
                tenant_id="t-b",
                ticker="ORCL",
                source="test",
                rationale="test",
                status="approved",
                proposed_at=date(2026, 2, 1),
                expires_at=date(2026, 3, 1),
            )
        )

        assert await discovery.get_active_tickers(tenant_id="t-a") == ["PLTR"]
        assert await discovery.get_active_tickers(tenant_id="t-b") == ["ORCL"]
        assert await discovery.get_active_tickers(tenant_id="default") == []

    async def test_dynamic_universe_includes_all_tenants(self, db: Database) -> None:
        """get_dynamic_universe merges approved tickers from ALL tenants."""
        await db.save_discovered_ticker(
            DiscoveredTickerRow(
                tenant_id="t-a",
                ticker="PLTR",
                source="agent",
                rationale="test",
                status="approved",
                proposed_at=date(2026, 2, 1),
                expires_at=date(2026, 3, 1),
            )
        )
        await db.save_discovered_ticker(
            DiscoveredTickerRow(
                tenant_id="t-b",
                ticker="ORCL",
                source="agent",
                rationale="test",
                status="approved",
                proposed_at=date(2026, 2, 1),
                expires_at=date(2026, 3, 1),
            )
        )

        universe = await get_dynamic_universe(db)
        assert "PLTR" in universe
        assert "ORCL" in universe
        assert "XLK" in universe

    async def test_dynamic_universe_no_extras(self, db: Database) -> None:
        from config.universe import BENCHMARK_TICKERS

        universe = await get_dynamic_universe(db)
        expected = sorted(set(FULL_UNIVERSE + BENCHMARK_TICKERS))
        assert universe == expected


# ── Expiry Tests ────────────────────────────────────────────────────────────


class TestExpiry:
    async def test_expires_old_tickers(self, discovery: TickerDiscovery, db: Database) -> None:
        await db.save_discovered_ticker(
            DiscoveredTickerRow(
                tenant_id="default",
                ticker="OLD",
                source="test",
                rationale="test",
                status="approved",
                proposed_at=date(2026, 1, 1),
                expires_at=date(2026, 2, 1),
            )
        )

        count = await discovery.expire_old(today=date(2026, 2, 5))
        assert count == 1

        row = await db.get_discovered_ticker("OLD")
        assert row.status == "expired"

    async def test_does_not_expire_future_tickers(self, discovery: TickerDiscovery, db: Database) -> None:
        await db.save_discovered_ticker(
            DiscoveredTickerRow(
                tenant_id="default",
                ticker="FRESH",
                source="test",
                rationale="test",
                status="approved",
                proposed_at=date(2026, 2, 1),
                expires_at=date(2026, 3, 15),
            )
        )

        count = await discovery.expire_old(today=date(2026, 2, 5))
        assert count == 0

        row = await db.get_discovered_ticker("FRESH")
        assert row.status == "approved"

    async def test_per_tenant_expiry(self, discovery: TickerDiscovery, db: Database) -> None:
        """Expiry only affects the specified tenant."""
        await db.save_discovered_ticker(
            DiscoveredTickerRow(
                tenant_id="t-a",
                ticker="OLD_A",
                source="test",
                rationale="test",
                status="approved",
                proposed_at=date(2026, 1, 1),
                expires_at=date(2026, 2, 1),
            )
        )
        await db.save_discovered_ticker(
            DiscoveredTickerRow(
                tenant_id="t-b",
                ticker="OLD_B",
                source="test",
                rationale="test",
                status="approved",
                proposed_at=date(2026, 1, 1),
                expires_at=date(2026, 2, 1),
            )
        )

        count = await discovery.expire_old(today=date(2026, 2, 5), tenant_id="t-a")
        assert count == 1

        row_a = await db.get_discovered_ticker("OLD_A", tenant_id="t-a")
        assert row_a.status == "expired"

        row_b = await db.get_discovered_ticker("OLD_B", tenant_id="t-b")
        assert row_b.status == "approved"  # untouched


# ── DB CRUD Tests ───────────────────────────────────────────────────────────


class TestDatabaseCRUD:
    async def test_save_and_get(self, db: Database) -> None:
        row = DiscoveredTickerRow(
            tenant_id="default",
            ticker="PLTR",
            source="agent",
            rationale="AI defense play",
            status="proposed",
            proposed_at=date(2026, 2, 5),
            expires_at=date(2026, 3, 7),
            sector="Technology",
            market_cap=50e9,
        )
        await db.save_discovered_ticker(row)

        stored = await db.get_discovered_ticker("PLTR")
        assert stored is not None
        assert stored.ticker == "PLTR"
        assert stored.market_cap == 50e9

    async def test_update_status(self, db: Database) -> None:
        await db.save_discovered_ticker(
            DiscoveredTickerRow(
                tenant_id="default",
                ticker="PLTR",
                source="agent",
                rationale="test",
                status="proposed",
                proposed_at=date(2026, 2, 5),
                expires_at=date(2026, 3, 7),
            )
        )

        await db.update_discovered_ticker_status("PLTR", "approved")
        row = await db.get_discovered_ticker("PLTR")
        assert row.status == "approved"

    async def test_get_approved_tickers(self, db: Database) -> None:
        for status in ("proposed", "approved", "rejected"):
            await db.save_discovered_ticker(
                DiscoveredTickerRow(
                    tenant_id="default",
                    ticker=f"T{status[0].upper()}",
                    source="test",
                    rationale="test",
                    status=status,
                    proposed_at=date(2026, 2, 5),
                    expires_at=date(2026, 3, 7),
                )
            )

        approved = await db.get_approved_tickers()
        assert len(approved) == 1
        assert approved[0].ticker == "TA"

    async def test_get_all_approved_all_tenants(self, db: Database) -> None:
        await db.save_discovered_ticker(
            DiscoveredTickerRow(
                tenant_id="t-a",
                ticker="PLTR",
                source="test",
                rationale="test",
                status="approved",
                proposed_at=date(2026, 2, 5),
                expires_at=date(2026, 3, 7),
            )
        )
        await db.save_discovered_ticker(
            DiscoveredTickerRow(
                tenant_id="t-b",
                ticker="ORCL",
                source="test",
                rationale="test",
                status="approved",
                proposed_at=date(2026, 2, 5),
                expires_at=date(2026, 3, 7),
            )
        )
        await db.save_discovered_ticker(
            DiscoveredTickerRow(
                tenant_id="t-b",
                ticker="REJECTED",
                source="test",
                rationale="test",
                status="rejected",
                proposed_at=date(2026, 2, 5),
                expires_at=date(2026, 3, 7),
            )
        )

        all_approved = await db.get_all_approved_tickers_all_tenants()
        tickers = {r.ticker for r in all_approved}
        assert tickers == {"PLTR", "ORCL"}

    async def test_get_all_discovered_tickers(self, db: Database) -> None:
        for status in ("proposed", "approved", "rejected"):
            await db.save_discovered_ticker(
                DiscoveredTickerRow(
                    tenant_id="default",
                    ticker=f"T{status[0].upper()}",
                    source="test",
                    rationale="test",
                    status=status,
                    proposed_at=date(2026, 2, 5),
                    expires_at=date(2026, 3, 7),
                )
            )

        # All statuses
        all_rows = await db.get_all_discovered_tickers()
        assert len(all_rows) == 3

        # Filter by status
        proposed = await db.get_all_discovered_tickers(status="proposed")
        assert len(proposed) == 1
        assert proposed[0].ticker == "TP"

    async def test_get_all_discovered_tenant_scoped(self, db: Database) -> None:
        await db.save_discovered_ticker(
            DiscoveredTickerRow(
                tenant_id="t-a",
                ticker="PLTR",
                source="test",
                rationale="test",
                status="proposed",
                proposed_at=date(2026, 2, 5),
                expires_at=date(2026, 3, 7),
            )
        )
        await db.save_discovered_ticker(
            DiscoveredTickerRow(
                tenant_id="t-b",
                ticker="ORCL",
                source="test",
                rationale="test",
                status="proposed",
                proposed_at=date(2026, 2, 5),
                expires_at=date(2026, 3, 7),
            )
        )

        t_a = await db.get_all_discovered_tickers(tenant_id="t-a")
        assert len(t_a) == 1
        assert t_a[0].ticker == "PLTR"

        t_b = await db.get_all_discovered_tickers(tenant_id="t-b")
        assert len(t_b) == 1
        assert t_b[0].ticker == "ORCL"


# ── Telegram Ticker Proposal Tests ──────────────────────────────────────────


class TestFormatTickerProposal:
    def test_contains_ticker(self) -> None:
        row = DiscoveredTickerRow(
            tenant_id="default",
            ticker="PLTR",
            source="agent",
            rationale="Defense AI growth",
            status="proposed",
            proposed_at=date(2026, 2, 5),
            expires_at=date(2026, 3, 7),
            sector="Technology",
            market_cap=50e9,
        )
        msg = format_ticker_proposal(row)
        assert "PLTR" in msg
        assert "Technology" in msg
        assert "$50.0B" in msg
        assert "Defense AI growth" in msg

    def test_handles_missing_sector(self) -> None:
        row = DiscoveredTickerRow(
            tenant_id="default",
            ticker="TEST",
            source="news",
            rationale="trending",
            status="proposed",
            proposed_at=date(2026, 2, 5),
            expires_at=date(2026, 3, 7),
            sector=None,
            market_cap=None,
        )
        msg = format_ticker_proposal(row)
        assert "Unknown" in msg
        assert "N/A" in msg

    def test_html_escapes_rationale(self) -> None:
        row = DiscoveredTickerRow(
            tenant_id="default",
            ticker="TEST",
            source="agent",
            rationale="Price < $100 & growing",
            status="proposed",
            proposed_at=date(2026, 2, 5),
            expires_at=date(2026, 3, 7),
        )
        msg = format_ticker_proposal(row)
        assert "&lt;" in msg
        assert "&amp;" in msg


class TestSendTickerProposal:
    async def test_sends_with_keyboard(self) -> None:
        notifier = TelegramNotifier(bot_token="test-token", chat_id="12345")
        mock_bot = AsyncMock()
        mock_msg = MagicMock()
        mock_msg.message_id = 42
        mock_bot.send_message.return_value = mock_msg
        notifier._bot = mock_bot

        row = DiscoveredTickerRow(
            tenant_id="default",
            ticker="PLTR",
            source="agent",
            rationale="test",
            status="proposed",
            proposed_at=date(2026, 2, 5),
            expires_at=date(2026, 3, 7),
            sector="Tech",
            market_cap=50e9,
        )

        result = await notifier.send_ticker_proposal(row, "req123")
        assert result == 42

        call_kwargs = mock_bot.send_message.call_args[1]
        keyboard = call_kwargs["reply_markup"]
        buttons = keyboard.inline_keyboard[0]
        assert len(buttons) == 2
        assert buttons[0].callback_data == "req123:approve"
        assert buttons[1].callback_data == "req123:reject"

    async def test_returns_none_without_chat_id(self) -> None:
        notifier = TelegramNotifier(bot_token="test-token", chat_id="")
        row = DiscoveredTickerRow(
            tenant_id="default",
            ticker="PLTR",
            source="agent",
            rationale="test",
            status="proposed",
            proposed_at=date(2026, 2, 5),
            expires_at=date(2026, 3, 7),
        )
        result = await notifier.send_ticker_proposal(row, "req123")
        assert result is None


class TestWaitForTickerApproval:
    async def test_receives_approve(self) -> None:
        notifier = TelegramNotifier(bot_token="test-token", chat_id="12345")
        mock_bot = AsyncMock()
        mock_update = MagicMock()
        mock_update.update_id = 1
        mock_update.callback_query.data = "req123:approve"
        mock_update.callback_query.message.chat.id = 12345
        mock_update.callback_query.message.chat_id = None
        mock_bot.get_updates.return_value = [mock_update]
        notifier._bot = mock_bot

        result = await notifier.wait_for_ticker_approval("req123", timeout_seconds=5)
        assert result == "approve"

    async def test_timeout_defaults_to_reject(self) -> None:
        notifier = TelegramNotifier(bot_token="test-token", chat_id="12345")
        mock_bot = AsyncMock()
        mock_bot.get_updates.return_value = []
        notifier._bot = mock_bot

        result = await notifier.wait_for_ticker_approval("req123", timeout_seconds=1)
        assert result == "reject"


# ── Portfolio B Extra Tickers Integration ────────────────────────────────────


class TestPortfolioBExtraTickers:
    def test_extra_tickers_accepted_in_trades(self) -> None:
        """Agent can trade dynamically approved tickers."""
        import pandas as pd

        from src.agent.claude_agent import ClaudeAgent
        from src.strategies.portfolio_b import AIAutonomyStrategy

        strategy = AIAutonomyStrategy(agent=ClaudeAgent(api_key="fake-key"))
        prices = pd.Series({"XLK": 200.0, "PLTR": 75.0})

        response = {
            "trades": [
                {"ticker": "PLTR", "side": "BUY", "weight": 0.10, "reason": "AI play"},
            ],
        }

        # Without extra_tickers, PLTR is rejected
        trades_without = strategy.agent_response_to_trades(
            response,
            total_value=66_000.0,
            current_positions={},
            latest_prices=prices,
        )
        assert len(trades_without) == 0

        # With extra_tickers, PLTR is accepted
        trades_with = strategy.agent_response_to_trades(
            response,
            total_value=66_000.0,
            current_positions={},
            latest_prices=prices,
            extra_tickers=["PLTR"],
        )
        assert len(trades_with) == 1
        assert trades_with[0].ticker == "PLTR"
