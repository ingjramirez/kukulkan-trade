"""Tests for inverse ETF-related data in portfolio and market tools."""

import numpy as np
import pandas as pd
import pytest

from src.agent.tools.market import _get_batch_technicals
from src.agent.tools.portfolio import _get_portfolio_state, _get_risk_assessment
from src.storage.database import Database


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


@pytest.fixture
def closes():
    """Synthetic close price DataFrame with inverse ETFs."""
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    np.random.seed(42)
    data = {
        "XLK": 200 + np.cumsum(np.random.normal(0, 1, 100)),
        "SH": 15 + np.cumsum(np.random.normal(0, 0.1, 100)),
        "PSQ": 12 + np.cumsum(np.random.normal(0, 0.1, 100)),
        "RWM": 20 + np.cumsum(np.random.normal(0, 0.1, 100)),
        "TBF": 18 + np.cumsum(np.random.normal(0, 0.1, 100)),
        "SPY": 450 + np.cumsum(np.random.normal(0, 1.5, 100)),
        "AAPL": 170 + np.cumsum(np.random.normal(0, 1, 100)),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def current_prices(closes: pd.DataFrame) -> dict[str, float]:
    return {t: float(closes[t].iloc[-1]) for t in closes.columns}


async def _seed_portfolio_with_inverse(db: Database, current_prices: dict[str, float]) -> None:
    """Seed a portfolio with normal + inverse positions."""
    total = 100_000.0
    await db.upsert_portfolio("B", cash=50_000.0, total_value=total)
    await db.upsert_position("B", "XLK", shares=100, avg_price=200.0)
    await db.upsert_position("B", "SH", shares=200, avg_price=15.0)
    await db.upsert_position("B", "AAPL", shares=50, avg_price=170.0)


class TestPortfolioStateInstrumentType:
    async def test_positions_include_instrument_type(self, db: Database, current_prices: dict) -> None:
        await _seed_portfolio_with_inverse(db, current_prices)
        result = await _get_portfolio_state(db, "default", current_prices)
        for pos in result["positions"]:
            assert "instrument_type" in pos

    async def test_inverse_etf_type(self, db: Database, current_prices: dict) -> None:
        await _seed_portfolio_with_inverse(db, current_prices)
        result = await _get_portfolio_state(db, "default", current_prices)
        sh_pos = next(p for p in result["positions"] if p["ticker"] == "SH")
        assert sh_pos["instrument_type"] == "inverse_etf"

    async def test_stock_type(self, db: Database, current_prices: dict) -> None:
        await _seed_portfolio_with_inverse(db, current_prices)
        result = await _get_portfolio_state(db, "default", current_prices)
        aapl_pos = next(p for p in result["positions"] if p["ticker"] == "AAPL")
        assert aapl_pos["instrument_type"] == "stock"


class TestRiskAssessmentInverseExposure:
    async def test_includes_inverse_exposure(self, db: Database, current_prices: dict, closes: pd.DataFrame) -> None:
        await _seed_portfolio_with_inverse(db, current_prices)
        result = await _get_risk_assessment(db, "default", current_prices, closes)
        assert "inverse_exposure" in result

    async def test_inverse_exposure_structure(self, db: Database, current_prices: dict, closes: pd.DataFrame) -> None:
        await _seed_portfolio_with_inverse(db, current_prices)
        result = await _get_risk_assessment(db, "default", current_prices, closes)
        ie = result["inverse_exposure"]
        assert "total_value" in ie
        assert "total_pct" in ie
        assert "positions" in ie
        assert "net_equity_pct" in ie

    async def test_inverse_exposure_with_no_inverse_positions(
        self, db: Database, current_prices: dict, closes: pd.DataFrame
    ) -> None:
        await db.upsert_portfolio("B", cash=90_000.0, total_value=100_000.0)
        await db.upsert_position("B", "XLK", shares=50, avg_price=200.0)
        result = await _get_risk_assessment(db, "default", current_prices, closes)
        ie = result["inverse_exposure"]
        assert ie["total_value"] == 0.0
        assert ie["total_pct"] == 0.0
        assert ie["positions"] == []

    async def test_net_equity_pct(self, db: Database, current_prices: dict, closes: pd.DataFrame) -> None:
        await _seed_portfolio_with_inverse(db, current_prices)
        result = await _get_risk_assessment(db, "default", current_prices, closes)
        ie = result["inverse_exposure"]
        # net_equity_pct should be less than equity_invested_pct since we have hedges
        assert ie["net_equity_pct"] < result["equity_invested_pct"]


class TestBatchTechnicalsInstrumentType:
    async def test_includes_instrument_type(self, closes: pd.DataFrame) -> None:
        result = await _get_batch_technicals(closes, ["SH", "AAPL", "XLK"])
        for entry in result["results"]:
            assert "instrument_type" in entry

    async def test_inverse_etf_classified(self, closes: pd.DataFrame) -> None:
        result = await _get_batch_technicals(closes, ["SH"])
        assert result["results"][0]["instrument_type"] == "inverse_etf"

    async def test_stock_classified(self, closes: pd.DataFrame) -> None:
        result = await _get_batch_technicals(closes, ["AAPL"])
        assert result["results"][0]["instrument_type"] == "stock"
