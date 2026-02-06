"""Market data fetcher — IBKR first, yfinance fallback.

Provides OHLCV data for the full ticker universe with caching to SQLite.
"""

from datetime import date, timedelta

import pandas as pd
import structlog
import yfinance as yf

from config.universe import FULL_UNIVERSE
from src.storage.database import Database
from src.storage.models import MarketDataRow

log = structlog.get_logger()


class MarketDataFetcher:
    """Fetches and caches OHLCV market data. IBKR-first, yfinance fallback."""

    def __init__(self, db: Database, ibkr_client=None) -> None:
        self._db = db
        self._ibkr_client = ibkr_client

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

    async def fetch_universe(
        self,
        tickers: list[str] | None = None,
        period: str = "6mo",
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV for multiple tickers. Tries IBKR first, falls back to yfinance.

        Args:
            tickers: List of tickers. Defaults to FULL_UNIVERSE.
            period: yfinance period string.

        Returns:
            Dict mapping ticker -> DataFrame.
        """
        tickers = tickers or FULL_UNIVERSE

        if self._ibkr_client and self._ibkr_client.is_connected():
            try:
                return await self._fetch_from_ibkr(tickers, period)
            except Exception as e:
                log.warning("ibkr_data_failed_falling_back_to_yfinance", error=str(e))

        return await self._fetch_from_yfinance(tickers, period)

    async def _fetch_from_ibkr(
        self,
        tickers: list[str],
        period: str,
    ) -> dict[str, pd.DataFrame]:
        """Fetch data from IBKR Gateway.

        Args:
            tickers: List of tickers.
            period: Period string (converted to IB duration).

        Returns:
            Dict mapping ticker -> DataFrame.
        """
        # Convert yfinance period to IB duration
        duration_map = {
            "1mo": "1 M",
            "3mo": "3 M",
            "6mo": "6 M",
            "1y": "1 Y",
            "2y": "2 Y",
        }
        duration = duration_map.get(period, "1 Y")

        log.info("fetching_from_ibkr", count=len(tickers), duration=duration)
        return await self._ibkr_client.fetch_universe_historical(
            tickers, duration=duration
        )

    async def _fetch_from_yfinance(
        self,
        tickers: list[str],
        period: str,
    ) -> dict[str, pd.DataFrame]:
        """Fetch data from yfinance (original logic).

        Args:
            tickers: List of tickers.
            period: yfinance period string.

        Returns:
            Dict mapping ticker -> DataFrame.
        """
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
        closes = pd.DataFrame({
            ticker: df["Close"] for ticker, df in data.items()
        })
        return closes.sort_index()

    @staticmethod
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
