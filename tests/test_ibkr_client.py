"""Tests for IBKR client — all tests mock ib_insync.IB."""

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.execution.ibkr_client import IBKRClient


class TestIBKRClientConnection:
    def test_default_params(self) -> None:
        client = IBKRClient()
        assert client._host == "127.0.0.1"
        assert client._port == 4002
        assert client._client_id == 1

    def test_custom_params(self) -> None:
        client = IBKRClient(host="10.0.0.1", port=7497, client_id=5, timeout=60)
        assert client._host == "10.0.0.1"
        assert client._port == 7497
        assert client._timeout == 60

    def test_is_connected_default(self) -> None:
        client = IBKRClient()
        assert not client.is_connected()

    def test_make_stock(self) -> None:
        client = IBKRClient()
        stock = client._make_stock("AAPL")
        assert stock.symbol == "AAPL"
        assert stock.exchange == "SMART"
        assert stock.currency == "USD"

    async def test_connect_success(self) -> None:
        client = IBKRClient()
        client._ib = MagicMock()
        client._ib.connectAsync = AsyncMock()
        result = await client.connect()
        assert result is True
        client._ib.connectAsync.assert_called_once_with(
            "127.0.0.1", 4002, clientId=1
        )

    async def test_connect_failure(self) -> None:
        client = IBKRClient()
        client._ib = MagicMock()
        client._ib.connectAsync = AsyncMock(side_effect=Exception("Refused"))
        result = await client.connect()
        assert result is False


class TestIBKRClientHistorical:
    async def test_fetch_historical_empty(self) -> None:
        client = IBKRClient()
        client._ib = MagicMock()
        client._ib.qualifyContractsAsync = AsyncMock()
        client._ib.reqHistoricalDataAsync = AsyncMock(return_value=[])

        df = await client.fetch_historical("XLK")
        assert df.empty

    async def test_fetch_historical_with_bars(self) -> None:
        client = IBKRClient()
        client._ib = MagicMock()
        client._ib.qualifyContractsAsync = AsyncMock()

        mock_bar = MagicMock()
        mock_bar.date = "2026-02-05"
        mock_bar.open = 200.0
        mock_bar.high = 205.0
        mock_bar.low = 198.0
        mock_bar.close = 203.0
        mock_bar.volume = 1_000_000
        client._ib.reqHistoricalDataAsync = AsyncMock(return_value=[mock_bar])

        df = await client.fetch_historical("XLK")
        assert not df.empty
        assert len(df) == 1
        assert "Close" in df.columns
        assert df.iloc[0]["Close"] == 203.0

    async def test_fetch_historical_qualifies_contract(self) -> None:
        client = IBKRClient()
        client._ib = MagicMock()
        client._ib.qualifyContractsAsync = AsyncMock()
        client._ib.reqHistoricalDataAsync = AsyncMock(return_value=[])

        await client.fetch_historical("AAPL")
        client._ib.qualifyContractsAsync.assert_called_once()

    async def test_fetch_universe_returns_dict(self) -> None:
        client = IBKRClient()
        client._ib = MagicMock()
        client._ib.qualifyContractsAsync = AsyncMock()

        mock_bar = MagicMock()
        mock_bar.date = "2026-02-05"
        mock_bar.open = mock_bar.high = mock_bar.low = mock_bar.close = 100.0
        mock_bar.volume = 500_000
        client._ib.reqHistoricalDataAsync = AsyncMock(return_value=[mock_bar])

        results = await client.fetch_universe_historical(["XLK", "XLF"])
        assert isinstance(results, dict)
        assert len(results) == 2


class TestIBKRClientOrders:
    async def test_place_market_order_filled(self) -> None:
        client = IBKRClient()
        client._ib = MagicMock()
        client._ib.qualifyContractsAsync = AsyncMock()

        mock_trade = MagicMock()
        mock_trade.isDone.return_value = True
        mock_trade.orderStatus.status = "Filled"
        mock_trade.orderStatus.avgFillPrice = 201.5
        mock_trade.orderStatus.filled = 10
        client._ib.placeOrder.return_value = mock_trade

        trade = await client.place_market_order("XLK", "BUY", 10, "atlas-A-XLK")
        assert trade is not None
        assert trade.orderStatus.status == "Filled"

    async def test_place_market_order_qualifies_contract(self) -> None:
        client = IBKRClient()
        client._ib = MagicMock()
        client._ib.qualifyContractsAsync = AsyncMock()

        mock_trade = MagicMock()
        mock_trade.isDone.return_value = True
        mock_trade.orderStatus.status = "Filled"
        client._ib.placeOrder.return_value = mock_trade

        await client.place_market_order("AAPL", "BUY", 5)
        client._ib.qualifyContractsAsync.assert_called_once()

    async def test_place_market_order_failure(self) -> None:
        client = IBKRClient()
        client._ib = MagicMock()
        client._ib.qualifyContractsAsync = AsyncMock()
        client._ib.placeOrder.side_effect = Exception("Order rejected")

        trade = await client.place_market_order("XLK", "BUY", 10)
        assert trade is None

    async def test_get_positions(self) -> None:
        client = IBKRClient()
        client._ib = MagicMock()
        client._ib.positions.return_value = []
        result = await client.get_positions()
        assert result == []

    async def test_get_account_summary(self) -> None:
        client = IBKRClient()
        client._ib = MagicMock()

        mock_val = MagicMock()
        mock_val.tag = "NetLiquidation"
        mock_val.value = "99000.0"
        client._ib.accountSummary.return_value = [mock_val]

        summary = await client.get_account_summary()
        assert summary["NetLiquidation"] == 99000.0


class TestIBKRClientDisconnect:
    async def test_disconnect_no_error(self) -> None:
        client = IBKRClient()
        client._ib = MagicMock()
        await client.disconnect()
        client._ib.disconnect.assert_called_once()

    async def test_disconnect_handles_exception(self) -> None:
        client = IBKRClient()
        client._ib = MagicMock()
        client._ib.disconnect.side_effect = Exception("Already disconnected")
        # Should not raise
        await client.disconnect()


class TestIBKRClientLatestPrices:
    async def test_get_latest_prices(self) -> None:
        client = IBKRClient()
        client._ib = MagicMock()
        client._ib.qualifyContractsAsync = AsyncMock()

        mock_bar = MagicMock()
        mock_bar.close = 205.0
        client._ib.reqHistoricalDataAsync = AsyncMock(return_value=[mock_bar])

        prices = await client.get_latest_prices(["XLK"])
        assert prices["XLK"] == 205.0

    async def test_get_latest_prices_empty_bars(self) -> None:
        client = IBKRClient()
        client._ib = MagicMock()
        client._ib.qualifyContractsAsync = AsyncMock()
        client._ib.reqHistoricalDataAsync = AsyncMock(return_value=[])

        prices = await client.get_latest_prices(["XLK"])
        assert "XLK" not in prices
