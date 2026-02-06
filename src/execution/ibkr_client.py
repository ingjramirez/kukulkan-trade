"""IBKR Gateway client using ib_insync.

Low-level wrapper for connecting to IB Gateway, fetching market data,
and submitting market orders for paper trading.
"""

import asyncio

import pandas as pd
import structlog
from ib_insync import IB, MarketOrder, Position, Stock, Trade

log = structlog.get_logger()


class IBKRClient:
    """Async wrapper around ib_insync for IB Gateway communication.

    Uses ib_insync's native async API (connectAsync, reqHistoricalDataAsync)
    to integrate cleanly with the application's asyncio event loop.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4002,
        client_id: int = 1,
        timeout: int = 30,
    ) -> None:
        self._host = host
        self._port = port
        self._client_id = client_id
        self._timeout = timeout
        self._ib = IB()

    async def connect(self) -> bool:
        """Connect to IB Gateway using native async API.

        Returns:
            True if connected successfully.
        """
        try:
            await self._ib.connectAsync(
                self._host, self._port, clientId=self._client_id
            )
            log.info(
                "ibkr_connected",
                host=self._host,
                port=self._port,
                client_id=self._client_id,
            )
            return True
        except Exception as e:
            log.warning("ibkr_connection_failed", error=str(e))
            return False

    async def disconnect(self) -> None:
        """Disconnect from IB Gateway."""
        try:
            self._ib.disconnect()
            log.info("ibkr_disconnected")
        except Exception as e:
            log.warning("ibkr_disconnect_error", error=str(e))

    def is_connected(self) -> bool:
        """Check if connected to IB Gateway."""
        return self._ib.isConnected()

    def _make_stock(self, ticker: str) -> Stock:
        """Create a Stock contract for US equities."""
        return Stock(ticker, "SMART", "USD")

    async def fetch_historical(
        self,
        ticker: str,
        duration: str = "1 Y",
        bar_size: str = "1 day",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data for a single ticker.

        Qualifies the contract first, then fetches bars via the async API.

        Args:
            ticker: Ticker symbol.
            duration: IB duration string (e.g., '1 Y', '6 M').
            bar_size: Bar size (e.g., '1 day', '1 hour').

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume.
        """
        contract = self._make_stock(ticker)
        await self._ib.qualifyContractsAsync(contract)

        bars = await self._ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )

        if not bars:
            return pd.DataFrame()

        data = [
            {
                "date": bar.date,
                "Open": bar.open,
                "High": bar.high,
                "Low": bar.low,
                "Close": bar.close,
                "Volume": bar.volume,
            }
            for bar in bars
        ]
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.set_index("date")
        return df

    async def fetch_universe_historical(
        self,
        tickers: list[str],
        duration: str = "1 Y",
        bar_size: str = "1 day",
    ) -> dict[str, pd.DataFrame]:
        """Fetch historical data for multiple tickers sequentially.

        Includes a 0.5s delay between requests to respect IBKR pacing limits.

        Args:
            tickers: List of ticker symbols.
            duration: IB duration string.
            bar_size: Bar size.

        Returns:
            Dict mapping ticker -> DataFrame.
        """
        results: dict[str, pd.DataFrame] = {}
        total = len(tickers)
        for i, ticker in enumerate(tickers):
            try:
                df = await self.fetch_historical(ticker, duration, bar_size)
                if not df.empty:
                    results[ticker] = df
                log.debug(
                    "ibkr_historical_fetched",
                    ticker=ticker,
                    rows=len(df),
                    progress=f"{i + 1}/{total}",
                )
            except Exception as e:
                log.warning("ibkr_historical_failed", ticker=ticker, error=str(e))
            # Respect IBKR pacing (max ~60 requests / 10 min for daily bars)
            await asyncio.sleep(0.5)

        log.info(
            "ibkr_universe_fetched",
            success=len(results),
            total=total,
        )
        return results

    async def get_latest_prices(self, tickers: list[str]) -> dict[str, float]:
        """Get latest prices by fetching the most recent 1-day bar.

        Uses historical data instead of live market data snapshots,
        which avoids needing a market data subscription.

        Args:
            tickers: List of ticker symbols.

        Returns:
            Dict mapping ticker -> latest close price.
        """
        prices: dict[str, float] = {}
        for ticker in tickers:
            try:
                contract = self._make_stock(ticker)
                await self._ib.qualifyContractsAsync(contract)
                bars = await self._ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime="",
                    durationStr="2 D",
                    barSizeSetting="1 day",
                    whatToShow="TRADES",
                    useRTH=True,
                )
                if bars:
                    prices[ticker] = bars[-1].close
            except Exception as e:
                log.warning("ibkr_price_failed", ticker=ticker, error=str(e))
            await asyncio.sleep(0.2)
        return prices

    async def place_market_order(
        self,
        ticker: str,
        side: str,
        shares: int,
        reference: str = "",
    ) -> Trade | None:
        """Place a market order.

        Qualifies the contract, submits the order, and waits up to
        `timeout` seconds for a fill.

        Args:
            ticker: Ticker symbol.
            side: 'BUY' or 'SELL'.
            shares: Number of shares.
            reference: Order reference string for tracking.

        Returns:
            Trade object if placed, None on failure.
        """
        contract = self._make_stock(ticker)
        await self._ib.qualifyContractsAsync(contract)

        order = MarketOrder(side, shares)
        order.tif = "DAY"
        if reference:
            order.orderRef = reference

        try:
            trade = self._ib.placeOrder(contract, order)
            log.info(
                "ibkr_order_placed",
                ticker=ticker,
                side=side,
                shares=shares,
                reference=reference,
            )

            # Wait for fill
            start = asyncio.get_running_loop().time()
            while not trade.isDone():
                await asyncio.sleep(0.5)
                elapsed = asyncio.get_running_loop().time() - start
                if elapsed > self._timeout:
                    log.warning(
                        "ibkr_order_timeout",
                        ticker=ticker,
                        timeout=self._timeout,
                    )
                    break

            if trade.orderStatus.status == "Filled":
                log.info(
                    "ibkr_order_filled",
                    ticker=ticker,
                    fill_price=trade.orderStatus.avgFillPrice,
                    filled_shares=trade.orderStatus.filled,
                )
            return trade
        except Exception as e:
            log.error("ibkr_order_failed", ticker=ticker, error=str(e))
            return None

    async def get_positions(self) -> list[Position]:
        """Get all current positions from IBKR.

        Returns:
            List of Position objects.
        """
        return self._ib.positions()

    async def get_account_summary(self) -> dict:
        """Get account summary (NetLiquidation, TotalCashValue).

        Returns:
            Dict with account values.
        """
        summary: dict = {}
        try:
            values = self._ib.accountSummary()
            for v in values:
                if v.tag in ("NetLiquidation", "TotalCashValue", "GrossPositionValue"):
                    summary[v.tag] = float(v.value)
        except Exception as e:
            log.warning("ibkr_account_summary_failed", error=str(e))
        return summary
