"""Ticker discovery pipeline for Portfolio B.

Validates agent-suggested tickers via yfinance, persists proposals
to the database, and manages the dynamic universe extension.
"""

from datetime import date, timedelta

import structlog
import yfinance as yf

from config.universe import FULL_UNIVERSE
from src.storage.database import Database
from src.storage.models import DiscoveredTickerRow

log = structlog.get_logger()

# Guardrails
MAX_DYNAMIC_TICKERS = 10
MIN_MARKET_CAP = 1_000_000_000  # $1B
MIN_AVG_VOLUME = 100_000
EXPIRY_DAYS = 30


class TickerValidationResult:
    """Result of ticker validation check."""

    def __init__(
        self,
        ticker: str,
        valid: bool,
        reason: str = "",
        sector: str = "",
        market_cap: float = 0.0,
        name: str = "",
    ) -> None:
        self.ticker = ticker
        self.valid = valid
        self.reason = reason
        self.sector = sector
        self.market_cap = market_cap
        self.name = name


class TickerDiscovery:
    """Manages discovery, validation, and lifecycle of dynamic tickers."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def validate_ticker(self, ticker: str) -> TickerValidationResult:
        """Validate a ticker via yfinance lookup.

        Checks:
        - Ticker exists and has recent data
        - Market cap > $1B
        - Average volume > 100K
        - Not already in the static universe

        Args:
            ticker: Ticker symbol to validate.

        Returns:
            TickerValidationResult with validation outcome.
        """
        ticker = ticker.upper().strip()

        # Already in static universe
        if ticker in FULL_UNIVERSE:
            return TickerValidationResult(
                ticker=ticker, valid=False, reason="Already in universe"
            )

        try:
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info

            if not info or info.get("regularMarketPrice") is None:
                return TickerValidationResult(
                    ticker=ticker, valid=False, reason="Ticker not found or no market data"
                )

            market_cap = info.get("marketCap", 0) or 0
            avg_volume = info.get("averageVolume", 0) or 0
            sector = info.get("sector", "Unknown")
            name = info.get("shortName", ticker)

            if market_cap < MIN_MARKET_CAP:
                return TickerValidationResult(
                    ticker=ticker,
                    valid=False,
                    reason=(
                        f"Market cap ${market_cap / 1e9:.1f}B"
                        f" < ${MIN_MARKET_CAP / 1e9:.0f}B minimum"
                    ),
                    market_cap=market_cap,
                )

            if avg_volume < MIN_AVG_VOLUME:
                return TickerValidationResult(
                    ticker=ticker,
                    valid=False,
                    reason=f"Avg volume {avg_volume:,.0f} < {MIN_AVG_VOLUME:,.0f} minimum",
                    market_cap=market_cap,
                )

            return TickerValidationResult(
                ticker=ticker,
                valid=True,
                sector=sector,
                market_cap=market_cap,
                name=name,
            )

        except Exception as e:
            log.warning("ticker_validation_failed", ticker=ticker, error=str(e))
            return TickerValidationResult(
                ticker=ticker, valid=False, reason=f"Validation error: {e}"
            )

    async def propose_ticker(
        self,
        ticker: str,
        rationale: str,
        source: str = "agent",
        today: date | None = None,
        tenant_id: str = "default",
    ) -> DiscoveredTickerRow | None:
        """Validate and persist a ticker proposal.

        Args:
            ticker: Ticker symbol.
            rationale: Why this ticker was suggested.
            source: Discovery source ("agent", "news", "screener").
            today: Override date for testing.
            tenant_id: Tenant UUID for scoping.

        Returns:
            The created DiscoveredTickerRow, or None if validation failed or limit reached.
        """
        today = today or date.today()
        ticker = ticker.upper().strip()

        # Check if already discovered (any status) for this tenant
        existing = await self._db.get_discovered_ticker(ticker, tenant_id=tenant_id)
        if existing:
            log.info("ticker_already_discovered", ticker=ticker, status=existing.status)
            return None

        # Check dynamic ticker limit (per-tenant)
        approved = await self._db.get_approved_tickers(tenant_id=tenant_id)
        pending = await self._get_pending_tickers(tenant_id=tenant_id)
        if len(approved) + len(pending) >= MAX_DYNAMIC_TICKERS:
            log.warning("dynamic_ticker_limit_reached", limit=MAX_DYNAMIC_TICKERS)
            return None

        # Validate via yfinance
        validation = self.validate_ticker(ticker)
        if not validation.valid:
            log.info("ticker_proposal_rejected", ticker=ticker, reason=validation.reason)
            return None

        row = DiscoveredTickerRow(
            tenant_id=tenant_id,
            ticker=ticker,
            source=source,
            rationale=rationale,
            status="proposed",
            proposed_at=today,
            expires_at=today + timedelta(days=EXPIRY_DAYS),
            sector=validation.sector,
            market_cap=validation.market_cap,
        )
        await self._db.save_discovered_ticker(row)
        log.info(
            "ticker_proposed",
            ticker=ticker,
            sector=validation.sector,
            market_cap=f"${validation.market_cap / 1e9:.1f}B",
            tenant_id=tenant_id,
        )
        return row

    async def get_active_tickers(
        self, tenant_id: str = "default",
    ) -> list[str]:
        """Get all approved, non-expired dynamic tickers for a tenant.

        Returns:
            List of ticker symbols.
        """
        approved = await self._db.get_approved_tickers(tenant_id=tenant_id)
        return [r.ticker for r in approved]

    async def expire_old(
        self, today: date | None = None, tenant_id: str = "default",
    ) -> int:
        """Expire tickers past their expiry date for a tenant.

        Returns:
            Number of tickers expired.
        """
        today = today or date.today()
        count = await self._db.expire_old_tickers(today, tenant_id=tenant_id)
        if count:
            log.info("tickers_expired", count=count, tenant_id=tenant_id)
        return count

    async def _get_pending_tickers(
        self, tenant_id: str = "default",
    ) -> list[DiscoveredTickerRow]:
        """Get tickers with 'proposed' status for a tenant."""
        from sqlalchemy import select

        async with self._db.session() as s:
            result = await s.execute(
                select(DiscoveredTickerRow).where(
                    DiscoveredTickerRow.tenant_id == tenant_id,
                    DiscoveredTickerRow.status == "proposed",
                )
            )
            return list(result.scalars().all())
