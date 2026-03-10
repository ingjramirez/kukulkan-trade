"""Market data fetcher using yfinance.

Provides OHLCV data for the full ticker universe with caching to SQLite.
"""

import asyncio
from datetime import date, datetime, timezone

import pandas as pd
import structlog
import yfinance as yf

from config.universe import FULL_UNIVERSE
from src.storage.database import Database
from src.storage.models import MarketDataRow
from src.utils.retry import retry_market_data

log = structlog.get_logger()


class MarketDataFetcher:
    """Fetches and caches OHLCV market data via yfinance."""

    def __init__(self, db: Database) -> None:
        self._db = db

    @retry_market_data
    async def fetch_ticker(
        self,
        ticker: str,
        period: str = "6mo",
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a single ticker.

        Args:
            ticker: The ticker symbol.
            period: yfinance period string (e.g., '6mo', '1y'). Ignored if start is set.
            start: Start date for data fetch.
            end: End date for data fetch.

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, indexed by date.
        """
        log.debug("fetching_ticker", ticker=ticker, period=period)
        yf_ticker = yf.Ticker(ticker)

        if start:
            df = yf_ticker.history(
                start=start.isoformat(),
                end=(end or date.today()).isoformat(),
            )
        else:
            df = yf_ticker.history(period=period)

        if df.empty:
            log.warning("no_data_returned", ticker=ticker)
            return df

        # Normalize column names and index
        df.index = pd.to_datetime(df.index).date
        df.index.name = "date"
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        return df

    @retry_market_data
    async def fetch_universe(
        self,
        tickers: list[str] | None = None,
        period: str = "6mo",
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV for multiple tickers via yfinance batch download.

        Args:
            tickers: List of tickers. Defaults to FULL_UNIVERSE.
            period: yfinance period string.

        Returns:
            Dict mapping ticker -> DataFrame.
        """
        tickers = tickers or FULL_UNIVERSE

        log.info("fetching_universe", count=len(tickers), period=period)

        results: dict[str, pd.DataFrame] = {}
        # yfinance supports batch download
        data = yf.download(
            tickers,
            period=period,
            group_by="ticker",
            threads=True,
            progress=False,
        )

        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    df = data[["Open", "High", "Low", "Close", "Volume"]].copy()
                else:
                    df = data[ticker][["Open", "High", "Low", "Close", "Volume"]].copy()
                df = df.dropna()
                if not df.empty:
                    df.index = pd.to_datetime(df.index).date
                    df.index.name = "date"
                    results[ticker] = df
            except (KeyError, TypeError):
                log.warning("ticker_fetch_failed", ticker=ticker)

        log.info("universe_fetched", success=len(results), total=len(tickers))
        return results

    async def fetch_and_cache(
        self,
        tickers: list[str] | None = None,
        period: str = "6mo",
    ) -> dict[str, pd.DataFrame]:
        """Fetch data and persist to SQLite cache.

        Args:
            tickers: List of tickers. Defaults to FULL_UNIVERSE.
            period: yfinance period string.

        Returns:
            Dict mapping ticker -> DataFrame.
        """
        data = await self.fetch_universe(tickers=tickers, period=period)

        for ticker, df in data.items():
            rows = [
                MarketDataRow(
                    ticker=ticker,
                    date=row_date,
                    open=row["Open"],
                    high=row["High"],
                    low=row["Low"],
                    close=row["Close"],
                    volume=row["Volume"],
                )
                for row_date, row in df.iterrows()
            ]
            await self._db.save_market_data(rows)

        log.info("market_data_cached", tickers=len(data))
        return data

    async def get_closes(
        self,
        tickers: list[str] | None = None,
        period: str = "6mo",
    ) -> pd.DataFrame:
        """Get a DataFrame of close prices for multiple tickers.

        Args:
            tickers: List of tickers. Defaults to FULL_UNIVERSE.
            period: yfinance period string.

        Returns:
            DataFrame with tickers as columns and dates as index.
        """
        data = await self.fetch_universe(tickers=tickers, period=period)
        closes = pd.DataFrame({ticker: df["Close"] for ticker, df in data.items()})
        return closes.sort_index()

    @staticmethod
    @retry_market_data
    def get_latest_price(ticker: str) -> float | None:
        """Get the most recent price for a ticker (synchronous convenience method).

        Args:
            ticker: The ticker symbol.

        Returns:
            Latest close price, or None if unavailable.
        """
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            log.warning("latest_price_failed", ticker=ticker)
        return None


# ── Extended Hours Prices ─────────────────────────────────────────────────

_price_cache: dict[str, tuple[float, datetime]] = {}
PRICE_CACHE_TTL_S = 300  # 5 minutes


def _clear_price_cache() -> None:
    """Clear price cache (for testing)."""
    _price_cache.clear()


async def get_extended_hours_prices(tickers: list[str]) -> dict[str, float]:
    """Fetch current prices including pre/post market.

    Uses yfinance fast_info with a 5-minute TTL cache to avoid
    excessive API calls during extended hours monitoring.
    """
    now = datetime.now(timezone.utc)
    results: dict[str, float] = {}
    tickers_to_fetch: list[str] = []

    for ticker in tickers:
        if ticker in _price_cache:
            price, cached_at = _price_cache[ticker]
            if (now - cached_at).total_seconds() < PRICE_CACHE_TTL_S:
                results[ticker] = price
                continue
        tickers_to_fetch.append(ticker)

    if tickers_to_fetch:

        def _fetch() -> dict[str, float]:
            prices: dict[str, float] = {}
            for t in tickers_to_fetch:
                try:
                    info = yf.Ticker(t).fast_info
                    last_price = info.get("lastPrice") or info.get("last_price")
                    if last_price is not None:
                        prices[t] = float(last_price)
                except Exception:
                    log.warning("extended_price_fetch_failed", ticker=t)
            return prices

        fetched = await asyncio.to_thread(_fetch)
        fetch_time = datetime.now(timezone.utc)
        for ticker, price in fetched.items():
            results[ticker] = price
            _price_cache[ticker] = (price, fetch_time)

    return results
