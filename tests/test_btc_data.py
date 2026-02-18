"""Tests for BTC data in market overview and risk rules."""

import numpy as np
import pandas as pd
import pytest

from config.risk_rules import RISK_RULES
from src.agent.tools.market import _get_btc_from_closes, _get_market_overview


class TestBtcFromCloses:
    def _make_closes(self, prices: list[float], ticker: str = "BTC-USD") -> pd.DataFrame:
        dates = pd.bdate_range(end="2026-02-18", periods=len(prices))
        return pd.DataFrame({ticker: prices}, index=dates)

    def test_btc_price_and_1d_change(self) -> None:
        closes = self._make_closes([90000, 95000])
        result = _get_btc_from_closes(closes)

        assert result is not None
        assert result["ticker"] == "BTC-USD"
        assert result["price"] == 95000.0
        assert result["change_1d_pct"] == pytest.approx(5.56, abs=0.01)

    def test_btc_5d_change(self) -> None:
        prices = [85000, 86000, 87000, 88000, 89000, 90000, 95000]
        closes = self._make_closes(prices)
        result = _get_btc_from_closes(closes)

        assert "change_5d_pct" in result
        # 5d change: (95000 - 86000) / 86000 * 100
        expected = ((95000 - 86000) / 86000) * 100
        assert result["change_5d_pct"] == pytest.approx(expected, abs=0.01)

    def test_btc_20d_change(self) -> None:
        prices = list(np.linspace(80000, 95000, 22))
        closes = self._make_closes(prices)
        result = _get_btc_from_closes(closes)

        assert "change_20d_pct" in result

    def test_btc_not_in_closes_returns_none(self) -> None:
        closes = pd.DataFrame({"SPY": [400, 405]}, index=pd.bdate_range(end="2026-02-18", periods=2))
        result = _get_btc_from_closes(closes)
        assert result is None

    def test_insufficient_data_returns_none(self) -> None:
        closes = self._make_closes([95000])
        result = _get_btc_from_closes(closes)
        assert result is None


class TestMarketOverviewBtc:
    async def test_overview_includes_bitcoin_when_available(self) -> None:
        dates = pd.bdate_range(end="2026-02-18", periods=7)
        closes = pd.DataFrame(
            {
                "SPY": [500, 502, 504, 506, 508, 510, 512],
                "BTC-USD": [85000, 86000, 87000, 88000, 89000, 90000, 95000],
            },
            index=dates,
        )
        result = await _get_market_overview(closes, vix=18.5, yield_curve=0.5, regime="BULL")

        assert "bitcoin" in result
        btc = result["bitcoin"]
        assert btc["ticker"] == "BTC-USD"
        assert btc["price"] == 95000.0
        assert "change_1d_pct" in btc
        assert "change_5d_pct" in btc

    async def test_overview_no_bitcoin_when_not_in_data(self) -> None:
        dates = pd.bdate_range(end="2026-02-18", periods=7)
        closes = pd.DataFrame({"SPY": [500, 502, 504, 506, 508, 510, 512]}, index=dates)
        result = await _get_market_overview(closes, vix=18.5, yield_curve=0.5, regime="BULL")

        assert "bitcoin" not in result

    async def test_overview_spy_still_works(self) -> None:
        dates = pd.bdate_range(end="2026-02-18", periods=7)
        closes = pd.DataFrame(
            {
                "SPY": [500, 502, 504, 506, 508, 510, 512],
                "BTC-USD": [85000, 86000, 87000, 88000, 89000, 90000, 95000],
            },
            index=dates,
        )
        result = await _get_market_overview(closes, vix=18.5, yield_curve=0.5, regime="BULL")

        assert "spy_price" in result
        assert result["spy_price"] == 512.0


class TestRiskRulesBtc:
    def test_btc_ticker_field(self) -> None:
        assert RISK_RULES.btc_ticker == "BTC-USD"

    def test_btc_proxy_still_ibit(self) -> None:
        assert RISK_RULES.btc_proxy == "IBIT"

    def test_btc_crash_threshold(self) -> None:
        assert RISK_RULES.btc_crash_threshold_pct == -0.20
