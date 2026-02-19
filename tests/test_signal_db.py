"""Tests for signal engine DB operations."""

from datetime import datetime, timedelta, timezone

import pytest

from src.storage.database import Database
from src.storage.models import TickerSignalRow


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


def _make_signal_row(
    tenant_id: str = "default",
    ticker: str = "XLK",
    rank: int = 1,
    scored_at: datetime | None = None,
) -> TickerSignalRow:
    return TickerSignalRow(
        tenant_id=tenant_id,
        ticker=ticker,
        composite_score=80.0,
        rank=rank,
        prev_rank=None,
        rank_velocity=0.0,
        momentum_20d=0.05,
        momentum_63d=0.10,
        rsi=55.0,
        macd_histogram=0.3,
        sma_trend_score=2.0,
        bollinger_pct_b=0.6,
        volume_ratio=1.2,
        alerts="[]",
        scored_at=scored_at or datetime.now(timezone.utc),
    )


class TestSaveSignalBatch:
    async def test_save_and_retrieve(self, db: Database) -> None:
        await db.ensure_tenant("default")
        now = datetime.now(timezone.utc)
        rows = [
            _make_signal_row(ticker="XLK", rank=1, scored_at=now),
            _make_signal_row(ticker="XLF", rank=2, scored_at=now),
        ]
        await db.save_signal_batch(rows)
        signals = await db.get_latest_signals("default")
        assert len(signals) == 2
        assert signals[0].rank == 1
        assert signals[1].rank == 2

    async def test_save_empty_batch(self, db: Database) -> None:
        await db.ensure_tenant("default")
        await db.save_signal_batch([])
        signals = await db.get_latest_signals("default")
        assert signals == []


class TestGetLatestSignals:
    async def test_returns_only_latest_batch(self, db: Database) -> None:
        """Only the most recent scored_at batch is returned."""
        await db.ensure_tenant("default")
        old_time = datetime(2026, 2, 19, 10, 0)
        new_time = datetime(2026, 2, 19, 10, 10)
        old_rows = [_make_signal_row(ticker="XLK", rank=1, scored_at=old_time)]
        new_rows = [
            _make_signal_row(ticker="XLK", rank=2, scored_at=new_time),
            _make_signal_row(ticker="XLF", rank=1, scored_at=new_time),
        ]
        await db.save_signal_batch(old_rows)
        await db.save_signal_batch(new_rows)
        signals = await db.get_latest_signals("default")
        assert len(signals) == 2
        # All returned rows should be from the latest batch (new_time)
        tickers = {s.ticker for s in signals}
        assert tickers == {"XLK", "XLF"}

    async def test_returns_empty_for_unknown_tenant(self, db: Database) -> None:
        signals = await db.get_latest_signals("nonexistent")
        assert signals == []

    async def test_sorted_by_rank(self, db: Database) -> None:
        await db.ensure_tenant("default")
        now = datetime.now(timezone.utc)
        rows = [
            _make_signal_row(ticker="XLE", rank=3, scored_at=now),
            _make_signal_row(ticker="XLK", rank=1, scored_at=now),
            _make_signal_row(ticker="XLF", rank=2, scored_at=now),
        ]
        await db.save_signal_batch(rows)
        signals = await db.get_latest_signals("default")
        assert [s.ticker for s in signals] == ["XLK", "XLF", "XLE"]

    async def test_tenant_isolation(self, db: Database) -> None:
        """Signals from different tenants don't mix."""
        await db.ensure_tenant("t1")
        await db.ensure_tenant("t2")
        now = datetime.now(timezone.utc)
        await db.save_signal_batch([_make_signal_row(tenant_id="t1", ticker="XLK", scored_at=now)])
        await db.save_signal_batch([_make_signal_row(tenant_id="t2", ticker="GLD", scored_at=now)])
        t1_signals = await db.get_latest_signals("t1")
        t2_signals = await db.get_latest_signals("t2")
        assert len(t1_signals) == 1
        assert t1_signals[0].ticker == "XLK"
        assert len(t2_signals) == 1
        assert t2_signals[0].ticker == "GLD"


class TestCleanupOldSignals:
    async def test_cleanup_removes_old(self, db: Database) -> None:
        await db.ensure_tenant("default")
        old_time = datetime.now(timezone.utc) - timedelta(hours=48)
        recent_time = datetime.now(timezone.utc) - timedelta(hours=1)
        await db.save_signal_batch([_make_signal_row(ticker="XLK", scored_at=old_time)])
        await db.save_signal_batch([_make_signal_row(ticker="XLF", scored_at=recent_time)])
        deleted = await db.cleanup_old_signals("default", keep_hours=24)
        assert deleted == 1
        remaining = await db.get_latest_signals("default")
        assert len(remaining) == 1
        assert remaining[0].ticker == "XLF"

    async def test_cleanup_preserves_recent(self, db: Database) -> None:
        await db.ensure_tenant("default")
        recent_time = datetime.now(timezone.utc) - timedelta(hours=1)
        await db.save_signal_batch([_make_signal_row(ticker="XLK", scored_at=recent_time)])
        deleted = await db.cleanup_old_signals("default", keep_hours=24)
        assert deleted == 0
        remaining = await db.get_latest_signals("default")
        assert len(remaining) == 1


class TestGetCachedClosesAndVolumes:
    async def test_returns_dataframes(self, db: Database) -> None:
        """Cached closes and volumes are returned as DataFrames."""
        from datetime import date as d

        from src.storage.models import MarketDataRow

        await db.save_market_data([
            MarketDataRow(ticker="XLK", date=d(2026, 2, 3), open=200, high=205, low=198, close=203, volume=1000000),
            MarketDataRow(ticker="XLK", date=d(2026, 2, 4), open=203, high=207, low=201, close=205, volume=1100000),
            MarketDataRow(ticker="XLF", date=d(2026, 2, 3), open=40, high=41, low=39.5, close=40.5, volume=500000),
            MarketDataRow(ticker="XLF", date=d(2026, 2, 4), open=40.5, high=42, low=40, close=41.5, volume=600000),
        ])
        closes, volumes = await db.get_cached_closes_and_volumes(tickers=["XLK", "XLF"])
        assert "XLK" in closes.columns
        assert "XLF" in closes.columns
        assert len(closes) == 2
        assert closes.loc[d(2026, 2, 4), "XLK"] == 205
        assert volumes.loc[d(2026, 2, 3), "XLF"] == 500000

    async def test_returns_empty_when_no_data(self, db: Database) -> None:
        closes, volumes = await db.get_cached_closes_and_volumes(tickers=["FAKE"])
        assert closes.empty
        assert volumes.empty
