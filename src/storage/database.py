"""SQLite database setup and CRUD operations."""

from datetime import date, datetime
from pathlib import Path

import structlog
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.storage.models import (
    AgentDecisionRow,
    AgentMemoryRow,
    Base,
    DailySnapshotRow,
    DiscoveredTickerRow,
    MarketDataRow,
    MomentumRankingRow,
    PortfolioRow,
    PositionRow,
    TradeRow,
)

log = structlog.get_logger()


class Database:
    """Async SQLite database manager."""

    def __init__(self, url: str = "sqlite+aiosqlite:///data/kukulkan.db") -> None:
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
                since_dt = datetime.combine(since, datetime.min.time())
                stmt = stmt.where(TradeRow.executed_at >= since_dt)
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
        """Save end-of-day portfolio snapshot (replaces if exists)."""
        async with self.session() as s:
            # Delete existing snapshot for this portfolio+date if re-running
            await s.execute(
                delete(DailySnapshotRow).where(
                    DailySnapshotRow.portfolio == portfolio,
                    DailySnapshotRow.date == snapshot_date,
                )
            )
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
        """Save a batch of momentum rankings (replaces if exists for same date)."""
        if not rankings:
            return
        async with self.session() as s:
            # Delete existing rankings for this date if re-running
            ranking_date = rankings[0].date
            await s.execute(
                delete(MomentumRankingRow).where(
                    MomentumRankingRow.date == ranking_date
                )
            )
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

    # ── Discovered Tickers ────────────────────────────────────────────────

    async def get_approved_tickers(self) -> list[DiscoveredTickerRow]:
        """Get all approved (non-expired) discovered tickers."""
        async with self.session() as s:
            result = await s.execute(
                select(DiscoveredTickerRow).where(
                    DiscoveredTickerRow.status == "approved"
                )
            )
            return list(result.scalars().all())

    async def get_discovered_ticker(self, ticker: str) -> DiscoveredTickerRow | None:
        """Get a discovered ticker by symbol."""
        async with self.session() as s:
            result = await s.execute(
                select(DiscoveredTickerRow).where(
                    DiscoveredTickerRow.ticker == ticker
                )
            )
            return result.scalar_one_or_none()

    async def save_discovered_ticker(self, row: DiscoveredTickerRow) -> None:
        """Save a new discovered ticker."""
        async with self.session() as s:
            s.add(row)
            await s.commit()

    async def update_discovered_ticker_status(
        self, ticker: str, status: str
    ) -> None:
        """Update the status of a discovered ticker."""
        async with self.session() as s:
            row = (
                await s.execute(
                    select(DiscoveredTickerRow).where(
                        DiscoveredTickerRow.ticker == ticker
                    )
                )
            ).scalar_one_or_none()
            if row:
                row.status = status
                await s.commit()

    async def expire_old_tickers(self, today: date) -> int:
        """Mark approved tickers past their expiry date as expired.

        Returns:
            Number of tickers expired.
        """
        async with self.session() as s:
            result = await s.execute(
                select(DiscoveredTickerRow).where(
                    DiscoveredTickerRow.status == "approved",
                    DiscoveredTickerRow.expires_at <= today,
                )
            )
            expired = list(result.scalars().all())
            for row in expired:
                row.status = "expired"
            await s.commit()
            return len(expired)

    # ── API Query Methods ──────────────────────────────────────────────

    async def get_all_portfolios(self) -> list[PortfolioRow]:
        """Get all portfolios."""
        async with self.session() as s:
            result = await s.execute(select(PortfolioRow))
            return list(result.scalars().all())

    async def get_all_trades(
        self,
        portfolio: str | None = None,
        side: str | None = None,
        limit: int = 100,
    ) -> list[TradeRow]:
        """Get trades with optional filters."""
        async with self.session() as s:
            stmt = select(TradeRow)
            if portfolio:
                stmt = stmt.where(TradeRow.portfolio == portfolio)
            if side:
                stmt = stmt.where(TradeRow.side == side)
            stmt = stmt.order_by(TradeRow.executed_at.desc()).limit(limit)
            result = await s.execute(stmt)
            return list(result.scalars().all())

    async def get_all_snapshots(
        self,
        portfolio: str | None = None,
        since: date | None = None,
    ) -> list[DailySnapshotRow]:
        """Get snapshots with optional filters."""
        async with self.session() as s:
            stmt = select(DailySnapshotRow)
            if portfolio:
                stmt = stmt.where(DailySnapshotRow.portfolio == portfolio)
            if since:
                stmt = stmt.where(DailySnapshotRow.date >= since)
            stmt = stmt.order_by(DailySnapshotRow.date)
            result = await s.execute(stmt)
            return list(result.scalars().all())

    async def get_agent_decisions(self, limit: int = 10) -> list[AgentDecisionRow]:
        """Get recent agent decisions."""
        async with self.session() as s:
            result = await s.execute(
                select(AgentDecisionRow)
                .order_by(AgentDecisionRow.date.desc())
                .limit(limit)
            )
            return list(result.scalars().all())

    # ── Agent Memory ──────────────────────────────────────────────────

    async def get_agent_memories(self, category: str) -> list[AgentMemoryRow]:
        """Get all memories for a given category, ordered by created_at."""
        async with self.session() as s:
            result = await s.execute(
                select(AgentMemoryRow)
                .where(AgentMemoryRow.category == category)
                .order_by(AgentMemoryRow.created_at)
            )
            return list(result.scalars().all())

    async def upsert_agent_memory(
        self,
        category: str,
        key: str,
        content: str,
        expires_at: datetime | None = None,
    ) -> None:
        """Create or update an agent memory entry."""
        async with self.session() as s:
            existing = (
                await s.execute(
                    select(AgentMemoryRow).where(
                        AgentMemoryRow.category == category,
                        AgentMemoryRow.key == key,
                    )
                )
            ).scalar_one_or_none()

            if existing:
                existing.content = content
                existing.expires_at = expires_at
                existing.created_at = datetime.utcnow()
            else:
                s.add(AgentMemoryRow(
                    category=category,
                    key=key,
                    content=content,
                    expires_at=expires_at,
                ))
            await s.commit()

    async def delete_expired_memories(self) -> int:
        """Delete memories past their expires_at. Returns count deleted."""
        async with self.session() as s:
            result = await s.execute(
                select(AgentMemoryRow).where(
                    AgentMemoryRow.expires_at.isnot(None),
                    AgentMemoryRow.expires_at <= datetime.utcnow(),
                )
            )
            expired = list(result.scalars().all())
            for row in expired:
                await s.delete(row)
            await s.commit()
            return len(expired)

    async def get_all_agent_memory_context(self) -> dict:
        """Return all 3 memory tiers as a dict.

        Returns:
            Dict with keys: short_term, weekly_summary, agent_note.
            Each value is a list of AgentMemoryRow.
        """
        return {
            "short_term": await self.get_agent_memories("short_term"),
            "weekly_summary": await self.get_agent_memories("weekly_summary"),
            "agent_note": await self.get_agent_memories("agent_note"),
        }
