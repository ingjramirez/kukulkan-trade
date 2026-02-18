"""SQLite database setup and CRUD operations."""

import json
from datetime import date, datetime, timezone
from pathlib import Path

import structlog
from sqlalchemy import delete, event, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.storage.models import (
    AgentBudgetLogRow,
    AgentDecisionRow,
    AgentMemoryRow,
    Base,
    ConvictionCalibrationRow,
    DailySnapshotRow,
    DiscoveredTickerRow,
    EarningsCalendarRow,
    ImprovementSnapshotRow,
    IntradaySnapshotRow,
    MarketDataRow,
    MomentumRankingRow,
    ParameterChangelogRow,
    PlaybookSnapshotRow,
    PortfolioRow,
    PositionRow,
    PostureHistoryRow,
    SentinelActionRow,
    TenantRow,
    ToolCallLogRow,
    TradeRow,
    TrailingStopRow,
    WatchlistRow,
)

log = structlog.get_logger()


class Database:
    """Async SQLite database manager."""

    def __init__(self, url: str = "sqlite+aiosqlite:///data/kukulkan.db") -> None:
        self._url = url
        self._engine = create_async_engine(url, echo=False)
        self._session_factory = sessionmaker(self._engine, class_=AsyncSession, expire_on_commit=False)

        # Enable foreign key enforcement for SQLite connections
        @event.listens_for(self._engine.sync_engine, "connect")
        def _set_sqlite_fk(dbapi_connection, connection_record):  # noqa: ARG001
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    async def init_db(self) -> None:
        """Create all tables if they don't exist."""
        db_path = self._url.replace("sqlite+aiosqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Ensure default tenant exists (FK anchor for tenant_id columns)
        await self.ensure_tenant("default", name="Default")

        db_type = self._url.split(":")[0].split("+")[0]
        log.info("database_initialized", type=db_type)

    async def ensure_tenant(self, tenant_id: str, *, name: str | None = None) -> None:
        """Create a tenant row if it doesn't exist (idempotent).

        Args:
            tenant_id: Tenant UUID to ensure exists.
            name: Display name (defaults to tenant_id).
        """
        async with self.session() as s:
            existing = await s.get(TenantRow, tenant_id)
            if not existing:
                s.add(TenantRow(id=tenant_id, name=name or tenant_id))
                await s.commit()

    def session(self) -> AsyncSession:
        """Create a new async session."""
        return self._session_factory()

    async def close(self) -> None:
        """Dispose of the engine."""
        await self._engine.dispose()

    # ── Portfolio CRUD ───────────────────────────────────────────────────

    async def get_portfolio(
        self,
        name: str,
        tenant_id: str = "default",
    ) -> PortfolioRow | None:
        """Get portfolio by name and tenant."""
        async with self.session() as s:
            result = await s.execute(
                select(PortfolioRow).where(
                    PortfolioRow.tenant_id == tenant_id,
                    PortfolioRow.name == name,
                )
            )
            return result.scalar_one_or_none()

    async def upsert_portfolio(
        self,
        name: str,
        cash: float,
        total_value: float,
        tenant_id: str = "default",
    ) -> None:
        """Create or update a portfolio."""
        async with self.session() as s:
            existing = (
                await s.execute(
                    select(PortfolioRow).where(
                        PortfolioRow.tenant_id == tenant_id,
                        PortfolioRow.name == name,
                    )
                )
            ).scalar_one_or_none()

            if existing:
                existing.cash = cash
                existing.total_value = total_value
                existing.updated_at = datetime.now(timezone.utc)
            else:
                s.add(
                    PortfolioRow(
                        tenant_id=tenant_id,
                        name=name,
                        cash=cash,
                        total_value=total_value,
                    )
                )
            await s.commit()

    # ── Position CRUD ────────────────────────────────────────────────────

    async def get_positions(
        self,
        portfolio: str,
        tenant_id: str = "default",
    ) -> list[PositionRow]:
        """Get all open positions for a portfolio."""
        async with self.session() as s:
            result = await s.execute(
                select(PositionRow).where(
                    PositionRow.tenant_id == tenant_id,
                    PositionRow.portfolio == portfolio,
                )
            )
            return list(result.scalars().all())

    async def upsert_position(
        self,
        portfolio: str,
        ticker: str,
        shares: float,
        avg_price: float,
        tenant_id: str = "default",
    ) -> None:
        """Create or update a position. Deletes if shares == 0."""
        async with self.session() as s:
            existing = (
                await s.execute(
                    select(PositionRow).where(
                        PositionRow.tenant_id == tenant_id,
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
                existing.updated_at = datetime.now(timezone.utc)
            elif shares > 0:
                s.add(
                    PositionRow(
                        tenant_id=tenant_id,
                        portfolio=portfolio,
                        ticker=ticker,
                        shares=shares,
                        avg_price=avg_price,
                    )
                )
            await s.commit()

    async def update_position_prices(
        self,
        portfolio: str,
        prices: dict[str, float],
        tenant_id: str = "default",
    ) -> None:
        """Update current_price and market_value for all positions in a portfolio."""
        async with self.session() as s:
            positions = (
                (
                    await s.execute(
                        select(PositionRow).where(
                            PositionRow.tenant_id == tenant_id,
                            PositionRow.portfolio == portfolio,
                        )
                    )
                )
                .scalars()
                .all()
            )
            for p in positions:
                price = prices.get(p.ticker)
                if price is not None:
                    p.current_price = price
                    p.market_value = p.shares * price
                    p.updated_at = datetime.now(timezone.utc)
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
        tenant_id: str = "default",
    ) -> None:
        """Record an executed trade."""
        async with self.session() as s:
            s.add(
                TradeRow(
                    tenant_id=tenant_id,
                    portfolio=portfolio,
                    ticker=ticker,
                    side=side,
                    shares=shares,
                    price=price,
                    total=shares * price,
                    reason=reason,
                )
            )
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
        self,
        portfolio: str,
        since: date | None = None,
        tenant_id: str = "default",
    ) -> list[TradeRow]:
        """Get trades for a portfolio, optionally filtered by date."""
        async with self.session() as s:
            stmt = select(TradeRow).where(
                TradeRow.tenant_id == tenant_id,
                TradeRow.portfolio == portfolio,
            )
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
        tenant_id: str = "default",
    ) -> None:
        """Save end-of-day portfolio snapshot (replaces if exists)."""
        async with self.session() as s:
            # Delete existing snapshot for this tenant+portfolio+date if re-running
            await s.execute(
                delete(DailySnapshotRow).where(
                    DailySnapshotRow.tenant_id == tenant_id,
                    DailySnapshotRow.portfolio == portfolio,
                    DailySnapshotRow.date == snapshot_date,
                )
            )
            s.add(
                DailySnapshotRow(
                    tenant_id=tenant_id,
                    portfolio=portfolio,
                    date=snapshot_date,
                    total_value=total_value,
                    cash=cash,
                    positions_value=positions_value,
                    daily_return_pct=daily_return_pct,
                    cumulative_return_pct=cumulative_return_pct,
                )
            )
            await s.commit()

    async def get_snapshots(
        self,
        portfolio: str,
        since: date | None = None,
        tenant_id: str = "default",
    ) -> list[DailySnapshotRow]:
        """Get snapshots for a portfolio."""
        async with self.session() as s:
            stmt = select(DailySnapshotRow).where(
                DailySnapshotRow.tenant_id == tenant_id,
                DailySnapshotRow.portfolio == portfolio,
            )
            if since:
                stmt = stmt.where(DailySnapshotRow.date >= since)
            stmt = stmt.order_by(DailySnapshotRow.date)
            result = await s.execute(stmt)
            return list(result.scalars().all())

    # ── Momentum Rankings ────────────────────────────────────────────────

    async def save_momentum_rankings(self, rankings: list[MomentumRankingRow]) -> None:
        """Save a batch of momentum rankings (replaces if exists for same date)."""
        if not rankings:
            return
        async with self.session() as s:
            # Delete existing rankings for this date if re-running
            ranking_date = rankings[0].date
            await s.execute(delete(MomentumRankingRow).where(MomentumRankingRow.date == ranking_date))
            s.add_all(rankings)
            await s.commit()

    async def get_latest_momentum_rankings(self) -> list[MomentumRankingRow]:
        """Get the most recent momentum rankings."""
        async with self.session() as s:
            # Find the latest date
            latest = await s.execute(select(MomentumRankingRow.date).order_by(MomentumRankingRow.date.desc()).limit(1))
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

    async def get_market_data(self, ticker: str, since: date | None = None) -> list[MarketDataRow]:
        """Get cached OHLCV data for a ticker."""
        async with self.session() as s:
            stmt = select(MarketDataRow).where(MarketDataRow.ticker == ticker)
            if since:
                stmt = stmt.where(MarketDataRow.date >= since)
            stmt = stmt.order_by(MarketDataRow.date)
            result = await s.execute(stmt)
            return list(result.scalars().all())

    # ── Discovered Tickers ────────────────────────────────────────────────

    async def get_approved_tickers(
        self,
        tenant_id: str = "default",
    ) -> list[DiscoveredTickerRow]:
        """Get all approved discovered tickers for a tenant."""
        async with self.session() as s:
            result = await s.execute(
                select(DiscoveredTickerRow).where(
                    DiscoveredTickerRow.tenant_id == tenant_id,
                    DiscoveredTickerRow.status == "approved",
                )
            )
            return list(result.scalars().all())

    async def get_all_approved_tickers_all_tenants(self) -> list[DiscoveredTickerRow]:
        """Get all approved discovered tickers across all tenants.

        Used for global market data fetching (need prices for all tickers).
        """
        async with self.session() as s:
            result = await s.execute(
                select(DiscoveredTickerRow).where(
                    DiscoveredTickerRow.status == "approved",
                )
            )
            return list(result.scalars().all())

    async def get_discovered_ticker(
        self,
        ticker: str,
        tenant_id: str = "default",
    ) -> DiscoveredTickerRow | None:
        """Get a discovered ticker by symbol and tenant."""
        async with self.session() as s:
            result = await s.execute(
                select(DiscoveredTickerRow).where(
                    DiscoveredTickerRow.tenant_id == tenant_id,
                    DiscoveredTickerRow.ticker == ticker,
                )
            )
            return result.scalar_one_or_none()

    async def save_discovered_ticker(self, row: DiscoveredTickerRow) -> None:
        """Save a new discovered ticker."""
        async with self.session() as s:
            s.add(row)
            await s.commit()

    async def update_discovered_ticker_status(
        self,
        ticker: str,
        status: str,
        tenant_id: str = "default",
    ) -> None:
        """Update the status of a discovered ticker."""
        async with self.session() as s:
            row = (
                await s.execute(
                    select(DiscoveredTickerRow).where(
                        DiscoveredTickerRow.tenant_id == tenant_id,
                        DiscoveredTickerRow.ticker == ticker,
                    )
                )
            ).scalar_one_or_none()
            if row:
                row.status = status
                await s.commit()

    async def expire_old_tickers(
        self,
        today: date,
        tenant_id: str = "default",
    ) -> int:
        """Mark approved tickers past their expiry date as expired for a tenant.

        Returns:
            Number of tickers expired.
        """
        async with self.session() as s:
            result = await s.execute(
                select(DiscoveredTickerRow).where(
                    DiscoveredTickerRow.tenant_id == tenant_id,
                    DiscoveredTickerRow.status == "approved",
                    DiscoveredTickerRow.expires_at <= today,
                )
            )
            expired = list(result.scalars().all())
            for row in expired:
                row.status = "expired"
            await s.commit()
            return len(expired)

    async def get_all_discovered_tickers(
        self,
        tenant_id: str = "default",
        status: str | None = None,
    ) -> list[DiscoveredTickerRow]:
        """Get all discovered tickers for a tenant, optionally filtered by status.

        Args:
            tenant_id: Tenant UUID.
            status: Optional status filter (proposed, approved, rejected, expired).

        Returns:
            List of DiscoveredTickerRow ordered by proposed_at desc.
        """
        async with self.session() as s:
            stmt = select(DiscoveredTickerRow).where(
                DiscoveredTickerRow.tenant_id == tenant_id,
            )
            if status:
                stmt = stmt.where(DiscoveredTickerRow.status == status)
            stmt = stmt.order_by(DiscoveredTickerRow.proposed_at.desc())
            result = await s.execute(stmt)
            return list(result.scalars().all())

    # ── API Query Methods ──────────────────────────────────────────────

    async def get_all_portfolios(
        self,
        tenant_id: str = "default",
    ) -> list[PortfolioRow]:
        """Get all portfolios for a tenant."""
        async with self.session() as s:
            result = await s.execute(
                select(PortfolioRow).where(
                    PortfolioRow.tenant_id == tenant_id,
                )
            )
            return list(result.scalars().all())

    async def get_all_trades(
        self,
        portfolio: str | None = None,
        side: str | None = None,
        limit: int = 100,
        tenant_id: str = "default",
    ) -> list[TradeRow]:
        """Get trades with optional filters."""
        async with self.session() as s:
            stmt = select(TradeRow).where(TradeRow.tenant_id == tenant_id)
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
        tenant_id: str = "default",
    ) -> list[DailySnapshotRow]:
        """Get snapshots with optional filters."""
        async with self.session() as s:
            stmt = select(DailySnapshotRow).where(
                DailySnapshotRow.tenant_id == tenant_id,
            )
            if portfolio:
                stmt = stmt.where(DailySnapshotRow.portfolio == portfolio)
            if since:
                stmt = stmt.where(DailySnapshotRow.date >= since)
            stmt = stmt.order_by(DailySnapshotRow.date)
            result = await s.execute(stmt)
            return list(result.scalars().all())

    async def get_agent_decisions(
        self,
        limit: int = 10,
        tenant_id: str = "default",
    ) -> list[AgentDecisionRow]:
        """Get recent agent decisions."""
        async with self.session() as s:
            result = await s.execute(
                select(AgentDecisionRow)
                .where(AgentDecisionRow.tenant_id == tenant_id)
                .order_by(AgentDecisionRow.date.desc())
                .limit(limit)
            )
            return list(result.scalars().all())

    # ── Agent Memory ──────────────────────────────────────────────────

    async def get_agent_memories(
        self,
        category: str,
        tenant_id: str = "default",
    ) -> list[AgentMemoryRow]:
        """Get all memories for a given category, ordered by created_at."""
        async with self.session() as s:
            result = await s.execute(
                select(AgentMemoryRow)
                .where(
                    AgentMemoryRow.tenant_id == tenant_id,
                    AgentMemoryRow.category == category,
                )
                .order_by(AgentMemoryRow.created_at)
            )
            return list(result.scalars().all())

    async def upsert_agent_memory(
        self,
        category: str,
        key: str,
        content: str,
        expires_at: datetime | None = None,
        tenant_id: str = "default",
    ) -> None:
        """Create or update an agent memory entry."""
        async with self.session() as s:
            existing = (
                await s.execute(
                    select(AgentMemoryRow).where(
                        AgentMemoryRow.tenant_id == tenant_id,
                        AgentMemoryRow.category == category,
                        AgentMemoryRow.key == key,
                    )
                )
            ).scalar_one_or_none()

            if existing:
                existing.content = content
                existing.expires_at = expires_at
                existing.created_at = datetime.now(timezone.utc)
            else:
                s.add(
                    AgentMemoryRow(
                        tenant_id=tenant_id,
                        category=category,
                        key=key,
                        content=content,
                        expires_at=expires_at,
                    )
                )
            await s.commit()

    async def delete_expired_memories(self) -> int:
        """Delete memories past their expires_at. Returns count deleted."""
        async with self.session() as s:
            result = await s.execute(
                select(AgentMemoryRow).where(
                    AgentMemoryRow.expires_at.isnot(None),
                    AgentMemoryRow.expires_at <= datetime.now(timezone.utc),
                )
            )
            expired = list(result.scalars().all())
            for row in expired:
                await s.delete(row)
            await s.commit()
            return len(expired)

    async def get_all_agent_memory_context(
        self,
        tenant_id: str = "default",
    ) -> dict:
        """Return all 3 memory tiers as a dict.

        Returns:
            Dict with keys: short_term, weekly_summary, agent_note.
            Each value is a list of AgentMemoryRow.
        """
        return {
            "short_term": await self.get_agent_memories("short_term", tenant_id),
            "weekly_summary": await self.get_agent_memories("weekly_summary", tenant_id),
            "agent_note": await self.get_agent_memories("agent_note", tenant_id),
        }

    # ── Trailing Stops ─────────────────────────────────────────────

    async def create_trailing_stop(
        self,
        tenant_id: str,
        portfolio: str,
        ticker: str,
        entry_price: float,
        trail_pct: float,
    ) -> TrailingStopRow:
        """Create or replace a trailing stop for a position."""
        peak_price = entry_price
        stop_price = peak_price * (1 - trail_pct)
        now = datetime.now(timezone.utc)
        async with self.session() as s:
            # Delete existing stop for same tenant/portfolio/ticker (unique constraint)
            existing = (
                await s.execute(
                    select(TrailingStopRow).where(
                        TrailingStopRow.tenant_id == tenant_id,
                        TrailingStopRow.portfolio == portfolio,
                        TrailingStopRow.ticker == ticker,
                    )
                )
            ).scalar_one_or_none()
            if existing:
                await s.delete(existing)
                await s.flush()

            row = TrailingStopRow(
                tenant_id=tenant_id,
                portfolio=portfolio,
                ticker=ticker,
                entry_price=entry_price,
                peak_price=peak_price,
                trail_pct=trail_pct,
                stop_price=stop_price,
                is_active=True,
                created_at=now,
                updated_at=now,
            )
            s.add(row)
            await s.commit()
            await s.refresh(row)
        return row

    async def get_active_trailing_stops(
        self,
        tenant_id: str,
        portfolio: str | None = None,
    ) -> list[TrailingStopRow]:
        """Get all active trailing stops, optionally filtered by portfolio."""
        async with self.session() as s:
            stmt = select(TrailingStopRow).where(
                TrailingStopRow.tenant_id == tenant_id,
                TrailingStopRow.is_active.is_(True),
            )
            if portfolio:
                stmt = stmt.where(TrailingStopRow.portfolio == portfolio)
            result = await s.execute(stmt)
            return list(result.scalars().all())

    async def update_trailing_stop(
        self,
        stop_id: int,
        *,
        peak_price: float | None = None,
        stop_price: float | None = None,
        is_active: bool | None = None,
    ) -> None:
        """Update a trailing stop's peak, stop_price, or active status."""
        async with self.session() as s:
            row = (await s.execute(select(TrailingStopRow).where(TrailingStopRow.id == stop_id))).scalar_one_or_none()
            if row is None:
                return
            if peak_price is not None:
                row.peak_price = peak_price
            if stop_price is not None:
                row.stop_price = stop_price
            if is_active is not None:
                row.is_active = is_active
            row.updated_at = datetime.now(timezone.utc)
            await s.commit()

    async def deactivate_trailing_stop(self, stop_id: int) -> None:
        """Deactivate a single trailing stop."""
        await self.update_trailing_stop(stop_id, is_active=False)

    async def deactivate_trailing_stops_for_ticker(
        self,
        tenant_id: str,
        portfolio: str,
        ticker: str,
    ) -> None:
        """Deactivate all trailing stops for a tenant/portfolio/ticker."""
        async with self.session() as s:
            result = await s.execute(
                select(TrailingStopRow).where(
                    TrailingStopRow.tenant_id == tenant_id,
                    TrailingStopRow.portfolio == portfolio,
                    TrailingStopRow.ticker == ticker,
                    TrailingStopRow.is_active.is_(True),
                )
            )
            for row in result.scalars().all():
                row.is_active = False
                row.updated_at = datetime.now(timezone.utc)
            await s.commit()

    async def update_trailing_stop_pct(
        self,
        tenant_id: str,
        portfolio: str,
        ticker: str,
        trail_pct: float,
    ) -> None:
        """Update trail_pct on the active trailing stop for a tenant/portfolio/ticker.

        Recalculates stop_price based on current peak and new trail_pct.
        """
        async with self.session() as s:
            row = (
                await s.execute(
                    select(TrailingStopRow).where(
                        TrailingStopRow.tenant_id == tenant_id,
                        TrailingStopRow.portfolio == portfolio,
                        TrailingStopRow.ticker == ticker,
                        TrailingStopRow.is_active.is_(True),
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                return
            row.trail_pct = trail_pct
            row.stop_price = round(row.peak_price * (1 - trail_pct), 2)
            row.updated_at = datetime.now(timezone.utc)
            await s.commit()

    # ── Earnings Calendar ────────────────────────────────────────

    async def upsert_earnings(
        self,
        ticker: str,
        earnings_date: date,
        source: str = "yfinance",
    ) -> None:
        """Insert or update an earnings date for a ticker."""
        async with self.session() as s:
            existing = (
                await s.execute(
                    select(EarningsCalendarRow).where(
                        EarningsCalendarRow.ticker == ticker,
                        EarningsCalendarRow.earnings_date == earnings_date,
                    )
                )
            ).scalar_one_or_none()
            if existing:
                existing.source = source
                existing.fetched_at = datetime.now(timezone.utc)
            else:
                s.add(
                    EarningsCalendarRow(
                        ticker=ticker,
                        earnings_date=earnings_date,
                        source=source,
                        fetched_at=datetime.now(timezone.utc),
                    )
                )
            await s.commit()

    async def get_upcoming_earnings(
        self,
        tickers: list[str],
        days_ahead: int = 14,
    ) -> list[EarningsCalendarRow]:
        """Get earnings within N days for given tickers."""
        from datetime import timedelta

        today = date.today()
        end_date = today + timedelta(days=days_ahead)
        async with self.session() as s:
            result = await s.execute(
                select(EarningsCalendarRow)
                .where(
                    EarningsCalendarRow.ticker.in_(tickers),
                    EarningsCalendarRow.earnings_date >= today,
                    EarningsCalendarRow.earnings_date <= end_date,
                )
                .order_by(EarningsCalendarRow.earnings_date)
            )
            return list(result.scalars().all())

    async def get_latest_earnings_fetch(self) -> datetime | None:
        """Get the most recent fetched_at timestamp from earnings_calendar."""
        async with self.session() as s:
            from sqlalchemy import func

            result = await s.execute(select(func.max(EarningsCalendarRow.fetched_at)))
            return result.scalar_one_or_none()

    async def cleanup_past_earnings(self) -> int:
        """Delete earnings rows where earnings_date < today. Returns count."""
        today = date.today()
        async with self.session() as s:
            result = await s.execute(
                select(EarningsCalendarRow).where(
                    EarningsCalendarRow.earnings_date < today,
                )
            )
            rows = list(result.scalars().all())
            for row in rows:
                await s.delete(row)
            await s.commit()
            return len(rows)

    # ── Watchlist ─────────────────────────────────────────────────

    async def upsert_watchlist_item(
        self,
        tenant_id: str,
        ticker: str,
        reason: str,
        conviction: str = "medium",
        target_entry: float | None = None,
        expires_at: date | None = None,
        portfolio: str = "B",
    ) -> None:
        """Create or update a watchlist item."""
        today = date.today()
        from datetime import timedelta

        exp = expires_at or (today + timedelta(days=14))
        async with self.session() as s:
            existing = (
                await s.execute(
                    select(WatchlistRow).where(
                        WatchlistRow.tenant_id == tenant_id,
                        WatchlistRow.ticker == ticker,
                    )
                )
            ).scalar_one_or_none()
            if existing:
                existing.reason = reason
                existing.conviction = conviction
                existing.target_entry = target_entry
                existing.expires_at = exp
                existing.portfolio = portfolio
            else:
                s.add(
                    WatchlistRow(
                        tenant_id=tenant_id,
                        portfolio=portfolio,
                        ticker=ticker,
                        reason=reason,
                        conviction=conviction,
                        target_entry=target_entry,
                        added_date=today,
                        expires_at=exp,
                    )
                )
            await s.commit()

    async def get_watchlist(
        self,
        tenant_id: str,
        portfolio: str = "B",
    ) -> list[WatchlistRow]:
        """Get all watchlist items for a tenant/portfolio."""
        async with self.session() as s:
            result = await s.execute(
                select(WatchlistRow)
                .where(
                    WatchlistRow.tenant_id == tenant_id,
                    WatchlistRow.portfolio == portfolio,
                )
                .order_by(WatchlistRow.added_date.desc())
            )
            return list(result.scalars().all())

    async def remove_watchlist_item(
        self,
        tenant_id: str,
        ticker: str,
    ) -> None:
        """Remove a watchlist item."""
        async with self.session() as s:
            row = (
                await s.execute(
                    select(WatchlistRow).where(
                        WatchlistRow.tenant_id == tenant_id,
                        WatchlistRow.ticker == ticker,
                    )
                )
            ).scalar_one_or_none()
            if row:
                await s.delete(row)
                await s.commit()

    async def cleanup_expired_watchlist(self, tenant_id: str) -> int:
        """Delete watchlist items past their expires_at. Returns count."""
        today = date.today()
        async with self.session() as s:
            result = await s.execute(
                select(WatchlistRow).where(
                    WatchlistRow.tenant_id == tenant_id,
                    WatchlistRow.expires_at < today,
                )
            )
            rows = list(result.scalars().all())
            for row in rows:
                await s.delete(row)
            await s.commit()
            return len(rows)

    async def remove_watchlist_if_traded(
        self,
        tenant_id: str,
        ticker: str,
    ) -> None:
        """Auto-remove a ticker from watchlist when it's actually traded."""
        await self.remove_watchlist_item(tenant_id, ticker)

    # ── Intraday Snapshots ─────────────────────────────────────────

    async def save_intraday_snapshot(
        self,
        tenant_id: str,
        portfolio: str,
        timestamp: datetime,
        total_value: float,
        cash: float,
        positions_value: float,
        is_extended_hours: bool = False,
        market_phase: str = "market",
    ) -> None:
        """Save an intraday portfolio snapshot (replaces if exists)."""
        async with self.session() as s:
            await s.execute(
                delete(IntradaySnapshotRow).where(
                    IntradaySnapshotRow.tenant_id == tenant_id,
                    IntradaySnapshotRow.portfolio == portfolio,
                    IntradaySnapshotRow.timestamp == timestamp,
                )
            )
            s.add(
                IntradaySnapshotRow(
                    tenant_id=tenant_id,
                    portfolio=portfolio,
                    timestamp=timestamp,
                    total_value=total_value,
                    cash=cash,
                    positions_value=positions_value,
                    is_extended_hours=is_extended_hours,
                    market_phase=market_phase,
                )
            )
            await s.commit()

    async def get_intraday_snapshots(
        self,
        tenant_id: str,
        portfolio: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[IntradaySnapshotRow]:
        """Get intraday snapshots, optionally filtered by portfolio and time range."""
        async with self.session() as s:
            stmt = select(IntradaySnapshotRow).where(
                IntradaySnapshotRow.tenant_id == tenant_id,
            )
            if portfolio:
                stmt = stmt.where(IntradaySnapshotRow.portfolio == portfolio)
            if since:
                stmt = stmt.where(IntradaySnapshotRow.timestamp >= since)
            if until:
                stmt = stmt.where(IntradaySnapshotRow.timestamp <= until)
            stmt = stmt.order_by(IntradaySnapshotRow.timestamp)
            result = await s.execute(stmt)
            return list(result.scalars().all())

    async def purge_old_intraday_snapshots(self, days: int = 90) -> int:
        """Delete intraday snapshots older than N days. Returns count deleted."""
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        async with self.session() as s:
            result = await s.execute(
                select(IntradaySnapshotRow).where(
                    IntradaySnapshotRow.timestamp < cutoff,
                )
            )
            rows = list(result.scalars().all())
            for row in rows:
                await s.delete(row)
            await s.commit()
            return len(rows)

    # ── Tool Call Logs ────────────────────────────────────────────────

    async def save_tool_call_logs(
        self,
        logs: list[dict],
        session_date: date,
        session_label: str | None = None,
        tenant_id: str = "default",
    ) -> list[int]:
        """Save a batch of tool call log entries.

        Args:
            logs: List of dicts with turn, tool_name, tool_input, etc.
            session_date: Date of the agent session.
            session_label: Session label (e.g. "Morning").
            tenant_id: Tenant UUID.

        Returns:
            List of saved row IDs.
        """
        if not logs:
            return []
        rows: list[ToolCallLogRow] = []
        async with self.session() as s:
            for log_entry in logs:
                # Serialize tool_input dict to JSON string for TEXT column
                raw_input = log_entry.get("tool_input")
                input_str = json.dumps(raw_input, default=str) if isinstance(raw_input, dict) else raw_input
                row = ToolCallLogRow(
                    tenant_id=tenant_id,
                    session_date=session_date,
                    session_label=session_label,
                    turn=log_entry.get("turn", 0),
                    tool_name=log_entry.get("tool_name", ""),
                    tool_input=input_str,
                    tool_output_preview=log_entry.get("tool_output_preview"),
                    success=log_entry.get("success", True),
                    error=log_entry.get("error"),
                )
                s.add(row)
                rows.append(row)
            await s.flush()
            saved_ids = [r.id for r in rows]
            await s.commit()
        return saved_ids

    async def update_tool_call_influenced(
        self,
        log_ids: list[int],
    ) -> None:
        """Mark tool call logs as having influenced the final decision.

        Args:
            log_ids: List of ToolCallLogRow IDs to mark as influential.
        """
        if not log_ids:
            return
        async with self.session() as s:
            result = await s.execute(select(ToolCallLogRow).where(ToolCallLogRow.id.in_(log_ids)))
            for row in result.scalars().all():
                row.influenced_decision = True
            await s.commit()

    async def get_tool_call_logs(
        self,
        tenant_id: str = "default",
        session_date: date | None = None,
        limit: int = 100,
    ) -> list[ToolCallLogRow]:
        """Get tool call logs, optionally filtered by date.

        Args:
            tenant_id: Tenant UUID.
            session_date: Optional date filter.
            limit: Max rows to return.

        Returns:
            List of ToolCallLogRow ordered by created_at desc.
        """
        async with self.session() as s:
            stmt = select(ToolCallLogRow).where(
                ToolCallLogRow.tenant_id == tenant_id,
            )
            if session_date:
                stmt = stmt.where(ToolCallLogRow.session_date == session_date)
            stmt = stmt.order_by(ToolCallLogRow.created_at.desc()).limit(limit)
            result = await s.execute(stmt)
            return list(result.scalars().all())

    # ── Posture History ──────────────────────────────────────────────

    async def save_posture(
        self,
        tenant_id: str,
        session_date: date,
        session_label: str | None,
        posture: str,
        effective_posture: str,
        reason: str | None = None,
    ) -> None:
        """Save a posture declaration for this session."""
        async with self.session() as s:
            s.add(
                PostureHistoryRow(
                    tenant_id=tenant_id,
                    session_date=session_date,
                    session_label=session_label,
                    posture=posture,
                    effective_posture=effective_posture,
                    reason=reason,
                )
            )
            await s.commit()

    async def get_current_posture(
        self,
        tenant_id: str = "default",
    ) -> PostureHistoryRow | None:
        """Get the most recent posture declaration for a tenant."""
        async with self.session() as s:
            result = await s.execute(
                select(PostureHistoryRow)
                .where(PostureHistoryRow.tenant_id == tenant_id)
                .order_by(PostureHistoryRow.created_at.desc())
                .limit(1)
            )
            return result.scalar_one_or_none()

    async def get_posture_history(
        self,
        tenant_id: str = "default",
        limit: int = 30,
    ) -> list[PostureHistoryRow]:
        """Get posture history for a tenant, most recent first."""
        async with self.session() as s:
            result = await s.execute(
                select(PostureHistoryRow)
                .where(PostureHistoryRow.tenant_id == tenant_id)
                .order_by(PostureHistoryRow.created_at.desc())
                .limit(limit)
            )
            return list(result.scalars().all())

    # ── Playbook Snapshots ────────────────────────────────────────────

    async def save_playbook_snapshot(
        self,
        cells: list[dict],
        tenant_id: str = "default",
    ) -> None:
        """Save a playbook snapshot (list of regime×sector cells).

        Args:
            cells: List of dicts with regime, sector, total_trades, wins, losses,
                   win_rate_pct, avg_pnl_pct, recommendation.
            tenant_id: Tenant UUID.
        """
        if not cells:
            return
        now = datetime.now(timezone.utc)
        async with self.session() as s:
            for cell in cells:
                s.add(
                    PlaybookSnapshotRow(
                        tenant_id=tenant_id,
                        generated_at=now,
                        regime=cell["regime"],
                        sector=cell["sector"],
                        total_trades=cell["total_trades"],
                        wins=cell["wins"],
                        losses=cell["losses"],
                        win_rate_pct=cell["win_rate_pct"],
                        avg_pnl_pct=cell["avg_pnl_pct"],
                        recommendation=cell["recommendation"],
                    )
                )
            await s.commit()

    async def get_latest_playbook(
        self,
        tenant_id: str = "default",
    ) -> list[PlaybookSnapshotRow]:
        """Get all cells from the most recent playbook snapshot."""
        from sqlalchemy import func

        async with self.session() as s:
            # Find the max generated_at for this tenant
            max_dt = await s.execute(
                select(func.max(PlaybookSnapshotRow.generated_at)).where(
                    PlaybookSnapshotRow.tenant_id == tenant_id,
                )
            )
            latest = max_dt.scalar_one_or_none()
            if not latest:
                return []

            result = await s.execute(
                select(PlaybookSnapshotRow).where(
                    PlaybookSnapshotRow.tenant_id == tenant_id,
                    PlaybookSnapshotRow.generated_at == latest,
                )
            )
            return list(result.scalars().all())

    # ── Conviction Calibration ────────────────────────────────────────

    async def save_conviction_calibration(
        self,
        buckets: list[dict],
        tenant_id: str = "default",
    ) -> None:
        """Save conviction calibration buckets.

        Args:
            buckets: List of dicts with conviction_level, total_trades, wins, losses,
                     win_rate_pct, avg_pnl_pct, assessment, suggested_multiplier.
            tenant_id: Tenant UUID.
        """
        if not buckets:
            return
        now = datetime.now(timezone.utc)
        async with self.session() as s:
            for bucket in buckets:
                s.add(
                    ConvictionCalibrationRow(
                        tenant_id=tenant_id,
                        generated_at=now,
                        conviction_level=bucket["conviction_level"],
                        total_trades=bucket["total_trades"],
                        wins=bucket["wins"],
                        losses=bucket["losses"],
                        win_rate_pct=bucket["win_rate_pct"],
                        avg_pnl_pct=bucket["avg_pnl_pct"],
                        assessment=bucket["assessment"],
                        suggested_multiplier=bucket["suggested_multiplier"],
                    )
                )
            await s.commit()

    async def get_latest_calibration(
        self,
        tenant_id: str = "default",
    ) -> list[ConvictionCalibrationRow]:
        """Get all buckets from the most recent calibration snapshot."""
        from sqlalchemy import func

        async with self.session() as s:
            max_dt = await s.execute(
                select(func.max(ConvictionCalibrationRow.generated_at)).where(
                    ConvictionCalibrationRow.tenant_id == tenant_id,
                )
            )
            latest = max_dt.scalar_one_or_none()
            if not latest:
                return []

            result = await s.execute(
                select(ConvictionCalibrationRow).where(
                    ConvictionCalibrationRow.tenant_id == tenant_id,
                    ConvictionCalibrationRow.generated_at == latest,
                )
            )
            return list(result.scalars().all())

    # ── Budget Log CRUD ─────────────────────────────────────────────

    async def save_budget_log(
        self,
        tenant_id: str,
        session_date: date,
        session_label: str,
        session_id: str | None,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_creation_tokens: int,
        cost_usd: float,
        session_profile: str | None = None,
    ) -> None:
        """Save a single session's cost record."""
        async with self.session() as s:
            s.add(
                AgentBudgetLogRow(
                    tenant_id=tenant_id,
                    session_date=session_date,
                    session_label=session_label,
                    session_id=session_id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_creation_tokens=cache_creation_tokens,
                    cost_usd=cost_usd,
                    session_profile=session_profile,
                )
            )
            await s.commit()

    async def get_daily_spend(
        self,
        tenant_id: str,
        target_date: date,
    ) -> float:
        """Get total spend for a tenant on a given date."""
        from sqlalchemy import func

        async with self.session() as s:
            result = await s.execute(
                select(func.coalesce(func.sum(AgentBudgetLogRow.cost_usd), 0.0)).where(
                    AgentBudgetLogRow.tenant_id == tenant_id,
                    AgentBudgetLogRow.session_date == target_date,
                )
            )
            return float(result.scalar_one())

    async def get_monthly_spend(
        self,
        tenant_id: str,
        year: int,
        month: int,
    ) -> float:
        """Get total spend for a tenant in a given month."""
        from sqlalchemy import extract, func

        async with self.session() as s:
            result = await s.execute(
                select(func.coalesce(func.sum(AgentBudgetLogRow.cost_usd), 0.0)).where(
                    AgentBudgetLogRow.tenant_id == tenant_id,
                    extract("year", AgentBudgetLogRow.session_date) == year,
                    extract("month", AgentBudgetLogRow.session_date) == month,
                )
            )
            return float(result.scalar_one())

    # ── Improvement Snapshot CRUD ────────────────────────────────────

    async def save_improvement_snapshot(
        self,
        tenant_id: str,
        week_start: date,
        week_end: date,
        total_trades: int,
        win_rate_pct: float | None,
        avg_pnl_pct: float | None,
        avg_alpha_vs_spy: float | None,
        total_cost_usd: float,
        strategy_mode: str | None,
        trailing_stop_multiplier: float | None,
        proposal_json: str | None,
        applied_changes: str | None,
        report_text: str | None,
    ) -> int:
        """Save a weekly improvement snapshot. Returns the snapshot ID."""
        async with self.session() as s:
            row = ImprovementSnapshotRow(
                tenant_id=tenant_id,
                week_start=week_start,
                week_end=week_end,
                total_trades=total_trades,
                win_rate_pct=win_rate_pct,
                avg_pnl_pct=avg_pnl_pct,
                avg_alpha_vs_spy=avg_alpha_vs_spy,
                total_cost_usd=total_cost_usd,
                strategy_mode=strategy_mode,
                trailing_stop_multiplier=trailing_stop_multiplier,
                proposal_json=proposal_json,
                applied_changes=applied_changes,
                report_text=report_text,
            )
            s.add(row)
            await s.commit()
            await s.refresh(row)
            return row.id

    async def update_improvement_snapshot_applied(
        self,
        snapshot_id: int,
        applied_changes: str | None,
        report_text: str | None,
    ) -> None:
        """Update applied_changes and report_text on an existing snapshot."""
        async with self.session() as s:
            row = await s.get(ImprovementSnapshotRow, snapshot_id)
            if row:
                row.applied_changes = applied_changes
                row.report_text = report_text
                await s.commit()

    async def get_improvement_snapshots(
        self,
        tenant_id: str = "default",
        limit: int = 20,
    ) -> list[ImprovementSnapshotRow]:
        """Get recent improvement snapshots for a tenant, newest first."""
        async with self.session() as s:
            result = await s.execute(
                select(ImprovementSnapshotRow)
                .where(ImprovementSnapshotRow.tenant_id == tenant_id)
                .order_by(ImprovementSnapshotRow.created_at.desc())
                .limit(limit)
            )
            return list(result.scalars().all())

    async def get_improvement_snapshot(
        self,
        snapshot_id: int,
        tenant_id: str = "default",
    ) -> ImprovementSnapshotRow | None:
        """Get a single improvement snapshot by ID (tenant-scoped)."""
        async with self.session() as s:
            result = await s.execute(
                select(ImprovementSnapshotRow).where(
                    ImprovementSnapshotRow.id == snapshot_id,
                    ImprovementSnapshotRow.tenant_id == tenant_id,
                )
            )
            return result.scalar_one_or_none()

    async def insert_parameter_changelog(
        self,
        tenant_id: str,
        parameter: str,
        old_value: str | None,
        new_value: str | None,
        reason: str | None = None,
        snapshot_id: int | None = None,
    ) -> None:
        """Insert a parameter change audit entry."""
        async with self.session() as s:
            s.add(
                ParameterChangelogRow(
                    tenant_id=tenant_id,
                    snapshot_id=snapshot_id,
                    parameter=parameter,
                    old_value=old_value,
                    new_value=new_value,
                    reason=reason,
                )
            )
            await s.commit()

    async def get_parameter_changelog(
        self,
        tenant_id: str = "default",
        limit: int = 50,
    ) -> list[ParameterChangelogRow]:
        """Get recent parameter changelog entries, newest first."""
        async with self.session() as s:
            result = await s.execute(
                select(ParameterChangelogRow)
                .where(ParameterChangelogRow.tenant_id == tenant_id)
                .order_by(ParameterChangelogRow.applied_at.desc())
                .limit(limit)
            )
            return list(result.scalars().all())

    async def get_parameter_changes_for(
        self,
        tenant_id: str,
        parameter: str,
        weeks: int = 4,
    ) -> list[ParameterChangelogRow]:
        """Get recent changes for a specific parameter (for flip-flop detection)."""
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(weeks=weeks)
        async with self.session() as s:
            result = await s.execute(
                select(ParameterChangelogRow)
                .where(
                    ParameterChangelogRow.tenant_id == tenant_id,
                    ParameterChangelogRow.parameter == parameter,
                    ParameterChangelogRow.applied_at >= cutoff,
                )
                .order_by(ParameterChangelogRow.applied_at.desc())
            )
            return list(result.scalars().all())

    # ── Tenant CRUD ──────────────────────────────────────────────────

    async def create_tenant(self, tenant: TenantRow) -> TenantRow:
        """Insert a new tenant row.

        Args:
            tenant: Fully-populated TenantRow (credentials already encrypted).

        Returns:
            The inserted TenantRow.
        """
        async with self.session() as s:
            s.add(tenant)
            await s.commit()
            await s.refresh(tenant)
        log.info("tenant_created", tenant_id=tenant.id, name=tenant.name)
        return tenant

    async def get_tenant(self, tenant_id: str) -> TenantRow | None:
        """Get a tenant by ID."""
        async with self.session() as s:
            result = await s.execute(select(TenantRow).where(TenantRow.id == tenant_id))
            return result.scalar_one_or_none()

    async def get_active_tenants(self) -> list[TenantRow]:
        """Get all active tenants, ordered by name."""
        async with self.session() as s:
            result = await s.execute(select(TenantRow).where(TenantRow.is_active.is_(True)).order_by(TenantRow.name))
            return list(result.scalars().all())

    async def get_tenant_by_username(self, username: str) -> TenantRow | None:
        """Find an active tenant by dashboard_user (for login)."""
        async with self.session() as s:
            result = await s.execute(
                select(TenantRow).where(TenantRow.dashboard_user == username).where(TenantRow.is_active.is_(True))
            )
            return result.scalar_one_or_none()

    async def get_all_tenants(self) -> list[TenantRow]:
        """Get all tenants (active and inactive)."""
        async with self.session() as s:
            result = await s.execute(select(TenantRow).order_by(TenantRow.name))
            return list(result.scalars().all())

    async def update_tenant(
        self,
        tenant_id: str,
        updates: dict,
    ) -> TenantRow | None:
        """Update a tenant's fields.

        Args:
            tenant_id: Tenant UUID.
            updates: Dict of column_name -> new_value (only non-None).

        Returns:
            Updated TenantRow, or None if not found.
        """
        async with self.session() as s:
            row = (await s.execute(select(TenantRow).where(TenantRow.id == tenant_id))).scalar_one_or_none()
            if row is None:
                return None
            for key, value in updates.items():
                if hasattr(row, key) and value is not None:
                    setattr(row, key, value)
            row.updated_at = datetime.now(timezone.utc)
            await s.commit()
            await s.refresh(row)
        log.info("tenant_updated", tenant_id=tenant_id, fields=list(updates.keys()))
        return row

    async def deactivate_tenant(self, tenant_id: str) -> bool:
        """Soft-delete a tenant by setting is_active=False.

        Returns:
            True if tenant was found and deactivated.
        """
        async with self.session() as s:
            row = (await s.execute(select(TenantRow).where(TenantRow.id == tenant_id))).scalar_one_or_none()
            if row is None:
                return False
            row.is_active = False
            row.updated_at = datetime.now(timezone.utc)
            await s.commit()
        log.info("tenant_deactivated", tenant_id=tenant_id)
        return True

    # ── Sentinel Actions ──────────────────────────────────────────────

    async def save_sentinel_action(
        self,
        tenant_id: str,
        action_type: str,
        ticker: str,
        reason: str,
        source: str,
        alert_level: str,
        status: str = "pending",
    ) -> int:
        """Save a queued sentinel action. Returns the new row ID."""
        async with self.session() as s:
            row = SentinelActionRow(
                tenant_id=tenant_id,
                action_type=action_type,
                ticker=ticker,
                reason=reason,
                source=source,
                alert_level=alert_level,
                status=status,
            )
            s.add(row)
            await s.commit()
            await s.refresh(row)
            return row.id

    async def get_pending_sentinel_actions(self, tenant_id: str) -> list[dict]:
        """Get all pending sentinel actions for a tenant, ordered by creation time."""
        async with self.session() as s:
            result = await s.execute(
                select(SentinelActionRow)
                .where(
                    SentinelActionRow.tenant_id == tenant_id,
                    SentinelActionRow.status == "pending",
                )
                .order_by(SentinelActionRow.created_at)
            )
            rows = list(result.scalars().all())
            return [
                {
                    "id": r.id,
                    "action_type": r.action_type,
                    "ticker": r.ticker,
                    "reason": r.reason,
                    "source": r.source,
                    "alert_level": r.alert_level,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]

    async def resolve_sentinel_action(
        self,
        action_id: int,
        status: str,
        resolved_by: str,
    ) -> bool:
        """Resolve a sentinel action (executed/cancelled). Returns True if found."""
        async with self.session() as s:
            row = await s.get(SentinelActionRow, action_id)
            if row is None:
                return False
            row.status = status
            row.resolved_by = resolved_by
            row.resolved_at = datetime.now(timezone.utc)
            await s.commit()
            return True

    async def get_last_market_hours_snapshot(
        self,
        tenant_id: str,
        portfolio: str | None = None,
    ) -> IntradaySnapshotRow | None:
        """Get the most recent market-hours (not extended) intraday snapshot."""
        async with self.session() as s:
            stmt = select(IntradaySnapshotRow).where(
                IntradaySnapshotRow.tenant_id == tenant_id,
                IntradaySnapshotRow.market_phase == "market",
            )
            if portfolio:
                stmt = stmt.where(IntradaySnapshotRow.portfolio == portfolio)
            stmt = stmt.order_by(IntradaySnapshotRow.timestamp.desc()).limit(1)
            result = await s.execute(stmt)
            return result.scalar_one_or_none()
