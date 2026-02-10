"""Tests for the SQLite database layer."""

from datetime import date

import pytest

from src.storage.database import Database


@pytest.fixture
async def db():
    """Create an in-memory test database."""
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


class TestPortfolioCRUD:
    async def test_upsert_and_get_portfolio(self, db: Database) -> None:
        await db.upsert_portfolio("A", cash=33_333.0, total_value=33_333.0)
        portfolio = await db.get_portfolio("A")
        assert portfolio is not None
        assert portfolio.name == "A"
        assert portfolio.cash == 33_333.0

    async def test_update_existing_portfolio(self, db: Database) -> None:
        await db.upsert_portfolio("B", cash=33_333.0, total_value=33_333.0)
        await db.upsert_portfolio("B", cash=30_000.0, total_value=35_000.0)
        portfolio = await db.get_portfolio("B")
        assert portfolio is not None
        assert portfolio.cash == 30_000.0
        assert portfolio.total_value == 35_000.0

    async def test_get_nonexistent_portfolio(self, db: Database) -> None:
        result = await db.get_portfolio("X")
        assert result is None


class TestPositionCRUD:
    async def test_create_position(self, db: Database) -> None:
        await db.upsert_position("A", "XLK", shares=100.0, avg_price=200.0)
        positions = await db.get_positions("A")
        assert len(positions) == 1
        assert positions[0].ticker == "XLK"
        assert positions[0].shares == 100.0

    async def test_update_position(self, db: Database) -> None:
        await db.upsert_position("A", "XLK", shares=100.0, avg_price=200.0)
        await db.upsert_position("A", "XLK", shares=150.0, avg_price=195.0)
        positions = await db.get_positions("A")
        assert len(positions) == 1
        assert positions[0].shares == 150.0

    async def test_delete_position_on_zero_shares(self, db: Database) -> None:
        await db.upsert_position("A", "XLK", shares=100.0, avg_price=200.0)
        await db.upsert_position("A", "XLK", shares=0, avg_price=0)
        positions = await db.get_positions("A")
        assert len(positions) == 0

    async def test_multiple_positions(self, db: Database) -> None:
        await db.upsert_position("A", "XLK", shares=50.0, avg_price=200.0)
        await db.upsert_position("A", "XLF", shares=75.0, avg_price=40.0)
        positions = await db.get_positions("A")
        assert len(positions) == 2


class TestUpdatePositionPrices:
    async def test_updates_current_price_and_market_value(self, db: Database) -> None:
        await db.upsert_position("A", "XLK", shares=10.0, avg_price=200.0)
        await db.upsert_position("A", "AAPL", shares=5.0, avg_price=150.0)

        await db.update_position_prices("A", {"XLK": 210.0, "AAPL": 160.0})

        positions = await db.get_positions("A")
        by_ticker = {p.ticker: p for p in positions}
        assert by_ticker["XLK"].current_price == 210.0
        assert by_ticker["XLK"].market_value == 2100.0
        assert by_ticker["AAPL"].current_price == 160.0
        assert by_ticker["AAPL"].market_value == 800.0

    async def test_skips_tickers_not_in_prices(self, db: Database) -> None:
        await db.upsert_position("A", "XLK", shares=10.0, avg_price=200.0)

        await db.update_position_prices("A", {})

        positions = await db.get_positions("A")
        assert positions[0].current_price is None
        assert positions[0].market_value is None

    async def test_scoped_to_portfolio(self, db: Database) -> None:
        await db.upsert_position("A", "XLK", shares=10.0, avg_price=200.0)
        await db.upsert_position("B", "XLK", shares=20.0, avg_price=200.0)

        await db.update_position_prices("A", {"XLK": 210.0})

        pos_a = await db.get_positions("A")
        pos_b = await db.get_positions("B")
        assert pos_a[0].current_price == 210.0
        assert pos_b[0].current_price is None


class TestTradeLog:
    async def test_log_and_retrieve_trade(self, db: Database) -> None:
        await db.log_trade("A", "XLK", "BUY", shares=100.0, price=200.0, reason="momentum")
        trades = await db.get_trades("A")
        assert len(trades) == 1
        assert trades[0].ticker == "XLK"
        assert trades[0].side == "BUY"
        assert trades[0].total == 20_000.0

    async def test_filter_trades_by_date(self, db: Database) -> None:
        await db.log_trade("A", "XLK", "BUY", shares=100.0, price=200.0)
        # Filtering by a future date should return nothing
        trades = await db.get_trades("A", since=date(2099, 1, 1))
        assert len(trades) == 0


class TestDailySnapshots:
    async def test_save_and_get_snapshot(self, db: Database) -> None:
        await db.save_snapshot(
            portfolio="A",
            snapshot_date=date(2026, 2, 5),
            total_value=34_000.0,
            cash=4_000.0,
            positions_value=30_000.0,
            daily_return_pct=2.0,
        )
        snapshots = await db.get_snapshots("A")
        assert len(snapshots) == 1
        assert snapshots[0].total_value == 34_000.0
