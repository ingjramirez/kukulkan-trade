"""Tests for the search_ticker_info agent tool.

Validates yfinance lookup, universe/discovery checks, minimum thresholds,
and price change / RSI computation.
"""

from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from src.agent.tools.market import _search_ticker_info
from src.storage.database import Database
from src.storage.models import DiscoveredTickerRow


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


def _make_yf_info(
    price: float = 285.0,
    market_cap: float = 45e9,
    avg_volume: float = 2_100_000,
    sector: str = "Technology",
    industry: str = "Computer Networking",
    name: str = "Arista Networks Inc",
    pe: float = 35.2,
    high_52w: float = 310.0,
    low_52w: float = 180.0,
) -> dict:
    return {
        "regularMarketPrice": price,
        "marketCap": market_cap,
        "averageVolume": avg_volume,
        "sector": sector,
        "industry": industry,
        "shortName": name,
        "trailingPE": pe,
        "fiftyTwoWeekHigh": high_52w,
        "fiftyTwoWeekLow": low_52w,
    }


def _make_history(days: int = 60, start_price: float = 270.0) -> pd.DataFrame:
    """Create a mock history DataFrame with Close column."""
    import numpy as np

    dates = pd.bdate_range(end=date.today(), periods=days)
    n = len(dates)
    prices = start_price + np.cumsum(np.random.default_rng(42).normal(0.5, 2, n))
    return pd.DataFrame({"Close": prices}, index=dates)


class TestSearchTickerInfoValid:
    @patch("src.agent.tools.market._yf_lookup")
    async def test_valid_ticker_returns_full_info(self, mock_lookup, db: Database) -> None:
        mock_lookup.return_value = {"found": True, "info": _make_yf_info(), "history": _make_history()}

        result = await _search_ticker_info(db, "default", "ANET")

        assert result["valid"] is True
        assert result["ticker"] == "ANET"
        assert result["name"] == "Arista Networks Inc"
        assert result["sector"] == "Technology"
        assert result["industry"] == "Computer Networking"
        assert result["market_cap"] == 45e9
        assert result["market_cap_display"] == "$45.0B"
        assert result["avg_volume"] == 2_100_000
        assert result["price"] == 285.0
        assert result["pe_ratio"] == 35.2
        assert result["meets_minimums"] is True
        assert result["disqualify_reason"] is None

    @patch("src.agent.tools.market._yf_lookup")
    async def test_invalid_ticker_returns_error(self, mock_lookup, db: Database) -> None:
        mock_lookup.return_value = {"found": False}

        result = await _search_ticker_info(db, "default", "XYZFAKE")

        assert result["valid"] is False
        assert result["ticker"] == "XYZFAKE"
        assert "not found" in result["error"].lower()

    @patch("src.agent.tools.market._yf_lookup")
    async def test_includes_price_changes(self, mock_lookup, db: Database) -> None:
        mock_lookup.return_value = {"found": True, "info": _make_yf_info(), "history": _make_history()}

        result = await _search_ticker_info(db, "default", "ANET")

        assert "change_1d_pct" in result
        assert "change_5d_pct" in result
        assert "change_20d_pct" in result

    @patch("src.agent.tools.market._yf_lookup")
    async def test_includes_rsi(self, mock_lookup, db: Database) -> None:
        mock_lookup.return_value = {"found": True, "info": _make_yf_info(), "history": _make_history(days=60)}

        result = await _search_ticker_info(db, "default", "ANET")

        assert "rsi_14" in result
        assert 0 <= result["rsi_14"] <= 100

    @patch("src.agent.tools.market._yf_lookup")
    async def test_in_universe_flag_true_for_existing(self, mock_lookup, db: Database) -> None:
        """XLK is in FULL_UNIVERSE."""
        mock_lookup.return_value = {"found": True, "info": _make_yf_info(), "history": _make_history()}

        result = await _search_ticker_info(db, "default", "XLK")

        assert result["in_universe"] is True

    @patch("src.agent.tools.market._yf_lookup")
    async def test_in_universe_flag_false_for_new(self, mock_lookup, db: Database) -> None:
        mock_lookup.return_value = {"found": True, "info": _make_yf_info(), "history": _make_history()}

        result = await _search_ticker_info(db, "default", "ANET")

        assert result["in_universe"] is False

    @patch("src.agent.tools.market._yf_lookup")
    async def test_previously_discovered_flag(self, mock_lookup, db: Database) -> None:
        await db.save_discovered_ticker(
            DiscoveredTickerRow(
                tenant_id="default",
                ticker="ANET",
                source="agent",
                rationale="test",
                status="approved",
                proposed_at=date(2026, 2, 10),
                expires_at=date(2026, 3, 10),
            )
        )
        mock_lookup.return_value = {"found": True, "info": _make_yf_info(), "history": _make_history()}

        result = await _search_ticker_info(db, "default", "ANET")

        assert result["previously_discovered"] is True
        assert result["discovery_status"] == "approved"

    @patch("src.agent.tools.market._yf_lookup")
    async def test_meets_minimums_false_low_market_cap(self, mock_lookup, db: Database) -> None:
        mock_lookup.return_value = {
            "found": True,
            "info": _make_yf_info(market_cap=500e6),
            "history": _make_history(),
        }

        result = await _search_ticker_info(db, "default", "SMALL")

        assert result["meets_minimums"] is False
        assert "Market cap" in result["disqualify_reason"]

    @patch("src.agent.tools.market._yf_lookup")
    async def test_meets_minimums_false_low_volume(self, mock_lookup, db: Database) -> None:
        mock_lookup.return_value = {
            "found": True,
            "info": _make_yf_info(avg_volume=50_000),
            "history": _make_history(),
        }

        result = await _search_ticker_info(db, "default", "LOWVOL")

        assert result["meets_minimums"] is False
        assert "volume" in result["disqualify_reason"].lower()

    @patch("src.agent.tools.market._yf_lookup")
    async def test_distance_from_52w_high(self, mock_lookup, db: Database) -> None:
        mock_lookup.return_value = {
            "found": True,
            "info": _make_yf_info(price=285.0, high_52w=310.0),
            "history": _make_history(),
        }

        result = await _search_ticker_info(db, "default", "ANET")

        assert "distance_from_52w_high_pct" in result
        assert result["distance_from_52w_high_pct"] < 0  # Below 52w high

    async def test_empty_ticker_returns_error(self, db: Database) -> None:
        result = await _search_ticker_info(db, "default", "")
        assert result["valid"] is False

    @patch("src.agent.tools.market._yf_lookup")
    async def test_normalizes_ticker_case(self, mock_lookup, db: Database) -> None:
        mock_lookup.return_value = {"found": True, "info": _make_yf_info(), "history": _make_history()}

        result = await _search_ticker_info(db, "default", "anet")

        assert result["ticker"] == "ANET"

    @patch("src.agent.tools.market._yf_lookup")
    async def test_works_without_db(self, mock_lookup) -> None:
        """Tool works even when db=None (just no discovery status)."""
        mock_lookup.return_value = {"found": True, "info": _make_yf_info(), "history": _make_history()}

        result = await _search_ticker_info(None, "default", "ANET")

        assert result["valid"] is True
        assert result["previously_discovered"] is False
        assert result["discovery_status"] is None

    @patch("src.agent.tools.market._yf_lookup")
    async def test_normalizes_alpaca_format_ticker(self, mock_lookup, db: Database) -> None:
        """BTC/USD (Alpaca format) should be normalized to BTC-USD (yfinance format)."""
        mock_lookup.return_value = {"found": True, "info": _make_yf_info(), "history": _make_history()}

        result = await _search_ticker_info(db, "default", "BTC/USD")

        assert result["ticker"] == "BTC-USD"
        mock_lookup.assert_called_once_with("BTC-USD")

    @patch("src.agent.tools.market._yf_lookup")
    async def test_handles_yfinance_exception(self, mock_lookup, db: Database) -> None:
        mock_lookup.side_effect = Exception("Network error")

        result = await _search_ticker_info(db, "default", "ERROR")

        assert result["valid"] is False
        assert "failed" in result["error"].lower()
