"""Earnings calendar: fetch upcoming earnings dates via yfinance, persist to DB."""

import asyncio
from datetime import date, datetime, timezone

import structlog
import yfinance as yf

from src.storage.database import Database
from src.storage.models import EarningsCalendarRow

log = structlog.get_logger()


class EarningsCalendar:
    """Fetches upcoming earnings dates using yfinance, persists to DB."""

    CACHE_TTL_HOURS = 12
    BATCH_SIZE = 10
    BATCH_DELAY_SECONDS = 1.0

    async def refresh_earnings(
        self,
        db: Database,
        tickers: list[str],
    ) -> int:
        """Fetch earnings dates for tickers, upsert into DB.

        Skips if the most recent fetch is within CACHE_TTL_HOURS.

        Args:
            db: Database instance.
            tickers: List of ticker symbols to check.

        Returns:
            Count of earnings dates found and stored.
        """
        # Check cache freshness
        latest_fetch = await db.get_latest_earnings_fetch()
        if latest_fetch:
            # SQLite returns naive datetimes — compare with naive utcnow
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            fetch_naive = latest_fetch.replace(tzinfo=None)
            hours_since = (now - fetch_naive).total_seconds() / 3600
            if hours_since < self.CACHE_TTL_HOURS:
                log.debug("earnings_cache_fresh", hours_since=round(hours_since, 1))
                return 0

        # Clean up past earnings
        await db.cleanup_past_earnings()

        count = 0
        for i in range(0, len(tickers), self.BATCH_SIZE):
            batch = tickers[i : i + self.BATCH_SIZE]
            for ticker in batch:
                try:
                    earnings_date = await self._fetch_earnings_date(ticker)
                    if earnings_date:
                        await db.upsert_earnings(ticker, earnings_date)
                        count += 1
                except Exception as e:
                    log.debug("earnings_fetch_failed", ticker=ticker, error=str(e))

            # Rate-limit between batches
            if i + self.BATCH_SIZE < len(tickers):
                await asyncio.sleep(self.BATCH_DELAY_SECONDS)

        log.info("earnings_refresh_complete", tickers=len(tickers), found=count)
        return count

    async def get_upcoming(
        self,
        db: Database,
        tickers: list[str],
        days_ahead: int = 14,
    ) -> list[EarningsCalendarRow]:
        """Return earnings within N days for given tickers.

        Args:
            db: Database instance.
            tickers: Tickers to filter by.
            days_ahead: How many days ahead to look.

        Returns:
            List of EarningsCalendarRow sorted by date.
        """
        return await db.get_upcoming_earnings(tickers, days_ahead)

    @staticmethod
    async def _fetch_earnings_date(ticker: str) -> date | None:
        """Fetch next earnings date from yfinance (runs in thread).

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Next earnings date, or None if unavailable.
        """

        def _sync_fetch() -> date | None:
            t = yf.Ticker(ticker)
            cal = t.calendar
            if cal is None:
                return None
            # yfinance returns calendar as a dict or DataFrame
            if isinstance(cal, dict):
                # Look for "Earnings Date" key
                earnings_dates = cal.get("Earnings Date")
                if earnings_dates and len(earnings_dates) > 0:
                    ed = earnings_dates[0]
                    if hasattr(ed, "date"):
                        return ed.date()
                    return ed
            return None

        return await asyncio.to_thread(_sync_fetch)
