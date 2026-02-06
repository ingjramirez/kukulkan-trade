"""SQLite database setup and CRUD operations."""

from datetime import date, datetime
from pathlib import Path

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.storage.models import (
    Base,
    DailySnapshotRow,
    MarketDataRow,
    MomentumRankingRow,
    PortfolioRow,
    PositionRow,
    TradeRow,
)

log = structlog.get_logger()


class Database:
    """Async SQLite database manager."""

    def __init__(self, url: str = "sqlite+aiosqlite:///data/atlas.db") -> None:
        self._url = url
        self._engine = create_async_engine(url, echo=False)
        self._session_factory = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_db(self) -> None:
        """Create all tables if they don't exist."""
        db_path = self._url.replace("sqlite+aiosqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        log.info("database_initialized", url=self._url)

    def session(self) -> AsyncSession:
        """Create a new async session."""
        return self._session_factory()

    async def close(self) -> None:
        """Dispose of the engine."""
        await self._engine.dispose()

    # ── Portfolio CRUD ───────────────────────────────────────────────────

    async def get_portfolio(self, name: str) -> PortfolioRow | None:
        """Get portfolio by name (A, B, or C)."""
        async with self.session() as s:
            result = await s.execute(
                select(PortfolioRow).where(PortfolioRow.name == name)
            )
            return result.scalar_one_or_none()

    async def upsert_portfolio(self, name: str, cash: float, total_value: float) -> None:
        """Create or update a portfolio."""
        async with self.session() as s:
            existing = (
                await s.execute(select(PortfolioRow).where(PortfolioRow.name == name))
            ).scalar_one_or_none()

            if existing:
                existing.cash = cash
                existing.total_value = total_value
                existing.updated_at = datetime.utcnow()
            else:
                s.add(PortfolioRow(
                    name=name, cash=cash, total_value=total_value
                ))
            await s.commit()

    # ── Position CRUD ────────────────────────────────────────────────────

    async def get_positions(self, portfolio: str) -> list[PositionRow]:
        """Get all open positions for a portfolio."""
        async with self.session() as s:
            result = await s.execute(
                select(PositionRow).where(PositionRow.portfolio == portfolio)
            )
            return list(result.scalars().all())

    async def upsert_position(
        self,
        portfolio: str,
        ticker: str,
        shares: float,
        avg_price: float,
    ) -> None:
        """Create or update a position. Deletes if shares == 0."""
        async with self.session() as s:
            existing = (
                await s.execute(
                    select(PositionRow).where(
                        PositionRow.portfolio == portfolio,
                        PositionRow.ticker == ticker,
                    )
                )
            ).scalar_one_or_none()

            if shares == 0 and existing:
                await s.delete(existing)
            elif existing:
                existing.shares = shares
                existing.avg_price = avg_price
                existing.updated_at = datetime.utcnow()
            elif shares > 0:
                s.add(PositionRow(
                    portfolio=portfolio,
                    ticker=ticker,
                    shares=shares,
                    avg_price=avg_price,
                ))
            await s.commit()

    # ── Trade Log ────────────────────────────────────────────────────────

    async def log_trade(
        self,
        portfolio: str,
        ticker: str,
        side: str,
        shares: float,
        price: float,
        reason: str = "",
    ) -> None:
        """Record an executed trade."""
        async with self.session() as s:
            s.add(TradeRow(
                portfolio=portfolio,
                ticker=ticker,
                side=side,
                shares=shares,
                price=price,
                total=shares * price,
                reason=reason,
            ))
            await s.commit()
        log.info(
            "trade_logged",
            portfolio=portfolio,
            ticker=ticker,
            side=side,
            shares=shares,
            price=price,
        )

    async def get_trades(
        self, portfolio: str, since: date | None = None
    ) -> list[TradeRow]:
        """Get trades for a portfolio, optionally filtered by date."""
        async with self.session() as s:
            stmt = select(TradeRow).where(TradeRow.portfolio == portfolio)
            if since:
                stmt = stmt.where(TradeRow.executed_at >= datetime.combine(since, datetime.min.time()))
            stmt = stmt.order_by(TradeRow.executed_at.desc())
            result = await s.execute(stmt)
            return list(result.scalars().all())

    # ── Daily Snapshots ──────────────────────────────────────────────────

    async def save_snapshot(
        self,
        portfolio: str,
        snapshot_date: date,
        total_value: float,
        cash: float,
        positions_value: float,
        daily_return_pct: float | None = None,
        cumulative_return_pct: float | None = None,
    ) -> None:
        """Save end-of-day portfolio snapshot."""
        async with self.session() as s:
            s.add(DailySnapshotRow(
                portfolio=portfolio,
                date=snapshot_date,
                total_value=total_value,
                cash=cash,
                positions_value=positions_value,
                daily_return_pct=daily_return_pct,
                cumulative_return_pct=cumulative_return_pct,
            ))
            await s.commit()

    async def get_snapshots(
        self, portfolio: str, since: date | None = None
    ) -> list[DailySnapshotRow]:
        """Get snapshots for a portfolio."""
        async with self.session() as s:
            stmt = select(DailySnapshotRow).where(
                DailySnapshotRow.portfolio == portfolio
            )
            if since:
                stmt = stmt.where(DailySnapshotRow.date >= since)
            stmt = stmt.order_by(DailySnapshotRow.date)
            result = await s.execute(stmt)
            return list(result.scalars().all())

    # ── Momentum Rankings ────────────────────────────────────────────────

    async def save_momentum_rankings(
        self, rankings: list[MomentumRankingRow]
    ) -> None:
        """Save a batch of momentum rankings."""
        async with self.session() as s:
            s.add_all(rankings)
            await s.commit()

    async def get_latest_momentum_rankings(self) -> list[MomentumRankingRow]:
        """Get the most recent momentum rankings."""
        async with self.session() as s:
            # Find the latest date
            latest = await s.execute(
                select(MomentumRankingRow.date)
                .order_by(MomentumRankingRow.date.desc())
                .limit(1)
            )
            latest_date = latest.scalar_one_or_none()
            if not latest_date:
                return []

            result = await s.execute(
                select(MomentumRankingRow)
                .where(MomentumRankingRow.date == latest_date)
                .order_by(MomentumRankingRow.rank)
            )
            return list(result.scalars().all())

    # ── Market Data Cache ────────────────────────────────────────────────

    async def save_market_data(self, rows: list[MarketDataRow]) -> None:
        """Bulk insert market data rows (ignores duplicates)."""
        async with self.session() as s:
            for row in rows:
                existing = (
                    await s.execute(
                        select(MarketDataRow).where(
                            MarketDataRow.ticker == row.ticker,
                            MarketDataRow.date == row.date,
                        )
                    )
                ).scalar_one_or_none()
                if not existing:
                    s.add(row)
            await s.commit()

    async def get_market_data(
        self, ticker: str, since: date | None = None
    ) -> list[MarketDataRow]:
        """Get cached OHLCV data for a ticker."""
        async with self.session() as s:
            stmt = select(MarketDataRow).where(MarketDataRow.ticker == ticker)
            if since:
                stmt = stmt.where(MarketDataRow.date >= since)
            stmt = stmt.order_by(MarketDataRow.date)
            result = await s.execute(stmt)
            return list(result.scalars().all())
