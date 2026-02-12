"""SQLite database setup and CRUD operations."""

from datetime import date, datetime, timezone
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
    EarningsCalendarRow,
    MarketDataRow,
    MomentumRankingRow,
    PortfolioRow,
    PositionRow,
    TenantRow,
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
        self._session_factory = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_db(self) -> None:
        """Create all tables if they don't exist."""
        db_path = self._url.replace("sqlite+aiosqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        db_type = self._url.split(":")[0].split("+")[0]
        log.info("database_initialized", type=db_type)

    def session(self) -> AsyncSession:
        """Create a new async session."""
        return self._session_factory()

    async def close(self) -> None:
        """Dispose of the engine."""
        await self._engine.dispose()

    # ── Portfolio CRUD ───────────────────────────────────────────────────

    async def get_portfolio(
        self, name: str, tenant_id: str = "default",
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
        self, name: str, cash: float, total_value: float,
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
                s.add(PortfolioRow(
                    tenant_id=tenant_id, name=name,
                    cash=cash, total_value=total_value,
                ))
            await s.commit()

    # ── Position CRUD ────────────────────────────────────────────────────

    async def get_positions(
        self, portfolio: str, tenant_id: str = "default",
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
                s.add(PositionRow(
                    tenant_id=tenant_id,
                    portfolio=portfolio,
                    ticker=ticker,
                    shares=shares,
                    avg_price=avg_price,
                ))
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
                await s.execute(
                    select(PositionRow).where(
                        PositionRow.tenant_id == tenant_id,
                        PositionRow.portfolio == portfolio,
                    )
                )
            ).scalars().all()
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
            s.add(TradeRow(
                tenant_id=tenant_id,
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
        self, portfolio: str, since: date | None = None,
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
            s.add(DailySnapshotRow(
                tenant_id=tenant_id,
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
        self, portfolio: str, since: date | None = None,
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

    async def get_all_portfolios(
        self, tenant_id: str = "default",
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
        self, limit: int = 10, tenant_id: str = "default",
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
        self, category: str, tenant_id: str = "default",
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
                s.add(AgentMemoryRow(
                    tenant_id=tenant_id,
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
                    AgentMemoryRow.expires_at <= datetime.now(timezone.utc),
                )
            )
            expired = list(result.scalars().all())
            for row in expired:
                await s.delete(row)
            await s.commit()
            return len(expired)

    async def get_all_agent_memory_context(
        self, tenant_id: str = "default",
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
        self, tenant_id: str, portfolio: str | None = None,
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
            row = (
                await s.execute(
                    select(TrailingStopRow).where(TrailingStopRow.id == stop_id)
                )
            ).scalar_one_or_none()
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
        self, tenant_id: str, portfolio: str, ticker: str,
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

    # ── Earnings Calendar ────────────────────────────────────────

    async def upsert_earnings(
        self, ticker: str, earnings_date: date, source: str = "yfinance",
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
                s.add(EarningsCalendarRow(
                    ticker=ticker,
                    earnings_date=earnings_date,
                    source=source,
                    fetched_at=datetime.now(timezone.utc),
                ))
            await s.commit()

    async def get_upcoming_earnings(
        self, tickers: list[str], days_ahead: int = 14,
    ) -> list[EarningsCalendarRow]:
        """Get earnings within N days for given tickers."""
        from datetime import timedelta
        today = date.today()
        end_date = today + timedelta(days=days_ahead)
        async with self.session() as s:
            result = await s.execute(
                select(EarningsCalendarRow).where(
                    EarningsCalendarRow.ticker.in_(tickers),
                    EarningsCalendarRow.earnings_date >= today,
                    EarningsCalendarRow.earnings_date <= end_date,
                ).order_by(EarningsCalendarRow.earnings_date)
            )
            return list(result.scalars().all())

    async def get_latest_earnings_fetch(self) -> datetime | None:
        """Get the most recent fetched_at timestamp from earnings_calendar."""
        async with self.session() as s:
            from sqlalchemy import func
            result = await s.execute(
                select(func.max(EarningsCalendarRow.fetched_at))
            )
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
                s.add(WatchlistRow(
                    tenant_id=tenant_id,
                    portfolio=portfolio,
                    ticker=ticker,
                    reason=reason,
                    conviction=conviction,
                    target_entry=target_entry,
                    added_date=today,
                    expires_at=exp,
                ))
            await s.commit()

    async def get_watchlist(
        self, tenant_id: str, portfolio: str = "B",
    ) -> list[WatchlistRow]:
        """Get all watchlist items for a tenant/portfolio."""
        async with self.session() as s:
            result = await s.execute(
                select(WatchlistRow).where(
                    WatchlistRow.tenant_id == tenant_id,
                    WatchlistRow.portfolio == portfolio,
                ).order_by(WatchlistRow.added_date.desc())
            )
            return list(result.scalars().all())

    async def remove_watchlist_item(
        self, tenant_id: str, ticker: str,
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
        self, tenant_id: str, ticker: str,
    ) -> None:
        """Auto-remove a ticker from watchlist when it's actually traded."""
        await self.remove_watchlist_item(tenant_id, ticker)

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
            result = await s.execute(
                select(TenantRow).where(TenantRow.id == tenant_id)
            )
            return result.scalar_one_or_none()

    async def get_active_tenants(self) -> list[TenantRow]:
        """Get all active tenants, ordered by name."""
        async with self.session() as s:
            result = await s.execute(
                select(TenantRow)
                .where(TenantRow.is_active.is_(True))
                .order_by(TenantRow.name)
            )
            return list(result.scalars().all())

    async def get_tenant_by_username(self, username: str) -> TenantRow | None:
        """Find an active tenant by dashboard_user (for login)."""
        async with self.session() as s:
            result = await s.execute(
                select(TenantRow)
                .where(TenantRow.dashboard_user == username)
                .where(TenantRow.is_active.is_(True))
            )
            return result.scalar_one_or_none()

    async def get_all_tenants(self) -> list[TenantRow]:
        """Get all tenants (active and inactive)."""
        async with self.session() as s:
            result = await s.execute(
                select(TenantRow).order_by(TenantRow.name)
            )
            return list(result.scalars().all())

    async def update_tenant(
        self, tenant_id: str, updates: dict,
    ) -> TenantRow | None:
        """Update a tenant's fields.

        Args:
            tenant_id: Tenant UUID.
            updates: Dict of column_name -> new_value (only non-None).

        Returns:
            Updated TenantRow, or None if not found.
        """
        async with self.session() as s:
            row = (
                await s.execute(
                    select(TenantRow).where(TenantRow.id == tenant_id)
                )
            ).scalar_one_or_none()
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
            row = (
                await s.execute(
                    select(TenantRow).where(TenantRow.id == tenant_id)
                )
            ).scalar_one_or_none()
            if row is None:
                return False
            row.is_active = False
            row.updated_at = datetime.now(timezone.utc)
            await s.commit()
        log.info("tenant_deactivated", tenant_id=tenant_id)
        return True
