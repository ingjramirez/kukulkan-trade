"""Tests for the GapRiskAnalyzer."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.analysis.gap_risk import GapRiskAnalyzer, GapRiskAssessment


def _make_position(
    ticker: str,
    shares: float = 100,
    current_price: float = 100.0,
    market_value: float | None = None,
) -> MagicMock:
    p = MagicMock()
    p.ticker = ticker
    p.shares = shares
    p.current_price = current_price
    p.avg_price = current_price
    p.market_value = market_value or shares * current_price
    return p


def _make_portfolio(total_value: float = 100000.0) -> MagicMock:
    p = MagicMock()
    p.total_value = total_value
    return p


def _make_db(
    positions: list | None = None,
    portfolio: MagicMock | None = None,
) -> AsyncMock:
    db = AsyncMock()
    db.get_positions = AsyncMock(return_value=positions or [])
    db.get_portfolio = AsyncMock(return_value=portfolio or _make_portfolio())
    return db


class TestGapRiskAnalyzer:
    async def test_earnings_multiplier_applied(self) -> None:
        positions = [_make_position("AAPL", 100, 150.0, market_value=15000.0)]
        db = _make_db(positions=positions)
        analyzer = GapRiskAnalyzer()

        result = await analyzer.analyze(db, "default", earnings_tickers={"AAPL"})

        aapl_risk = result.positions[0]
        assert "Earnings tonight" in aapl_risk.reasons
        # Base weight = 15% * 3.0 (earnings) = 45
        assert aapl_risk.gap_risk_score > 40

    async def test_volatile_sector_multiplier(self) -> None:
        positions = [_make_position("NVDA", 50, 800.0, market_value=40000.0)]
        db = _make_db(positions=positions)
        analyzer = GapRiskAnalyzer()

        with patch.object(analyzer, "_get_sector", return_value="Semiconductors"):
            result = await analyzer.analyze(db, "default", earnings_tickers=set())

        nvda_risk = result.positions[0]
        assert "Volatile sector" in nvda_risk.reasons[0]
        # 40% weight * 1.5 (volatile) = 60
        assert nvda_risk.gap_risk_score > 50

    async def test_concentration_multiplier_above_15pct(self) -> None:
        positions = [_make_position("TSLA", 100, 200.0, market_value=20000.0)]
        db = _make_db(positions=positions)
        analyzer = GapRiskAnalyzer()

        result = await analyzer.analyze(db, "default", earnings_tickers=set())

        tsla_risk = result.positions[0]
        assert any("Large position" in r for r in tsla_risk.reasons)

    async def test_inverse_etf_reduces_risk(self) -> None:
        positions = [_make_position("SH", 100, 50.0, market_value=5000.0)]
        db = _make_db(positions=positions)
        analyzer = GapRiskAnalyzer()

        with patch("config.universe.classify_instrument") as mock_classify:
            from config.universe import InstrumentType

            mock_classify.return_value = InstrumentType.INVERSE_ETF
            result = await analyzer.analyze(db, "default", earnings_tickers=set())

        sh_risk = result.positions[0]
        assert "Inverse ETF" in sh_risk.reasons[0]
        # 5% weight * 0.5 = 2.5
        assert sh_risk.gap_risk_score < 5

    async def test_no_multipliers_base_risk(self) -> None:
        positions = [_make_position("XLK", 100, 100.0, market_value=10000.0)]
        db = _make_db(positions=positions)
        analyzer = GapRiskAnalyzer()

        with patch.object(analyzer, "_get_sector", return_value="Financials"):
            result = await analyzer.analyze(db, "default", earnings_tickers=set())

        xlk_risk = result.positions[0]
        assert xlk_risk.reasons == []
        # 10% weight, no multipliers = 10.0
        assert xlk_risk.gap_risk_score == 10.0

    async def test_rating_low(self) -> None:
        positions = [_make_position("XLK", 10, 100.0, market_value=1000.0)]
        db = _make_db(positions=positions)
        analyzer = GapRiskAnalyzer()

        result = await analyzer.analyze(db, "default", earnings_tickers=set())
        assert result.rating == "LOW"

    async def test_rating_moderate(self) -> None:
        positions = [_make_position("AAPL", 100, 100.0, market_value=10000.0)]
        db = _make_db(positions=positions)
        analyzer = GapRiskAnalyzer()

        with patch.object(analyzer, "_get_sector", return_value="Financials"):
            result = await analyzer.analyze(db, "default", earnings_tickers=set())
        # 10% weight = 10.0 score → MODERATE (5-15)
        assert result.rating == "MODERATE"

    async def test_rating_high(self) -> None:
        positions = [_make_position("AAPL", 100, 200.0, market_value=20000.0)]
        db = _make_db(positions=positions)
        analyzer = GapRiskAnalyzer()

        with patch.object(analyzer, "_get_sector", return_value="Financials"):
            result = await analyzer.analyze(db, "default", earnings_tickers=set())
        # 20% weight * 1.2 (concentration) = 24 → HIGH (15-30)
        assert result.rating == "HIGH"

    async def test_rating_extreme(self) -> None:
        # Multiple large positions with earnings
        positions = [
            _make_position("AAPL", 100, 300.0, market_value=30000.0),
            _make_position("MSFT", 50, 400.0, market_value=20000.0),
        ]
        db = _make_db(positions=positions)
        analyzer = GapRiskAnalyzer()

        result = await analyzer.analyze(db, "default", earnings_tickers={"AAPL", "MSFT"})
        # AAPL: 30% * 3.0 * 1.2 = 108, MSFT: 20% * 3.0 * 1.2 = 72 → total 180 → EXTREME
        assert result.rating == "EXTREME"

    async def test_empty_portfolio_low_risk(self) -> None:
        db = _make_db(positions=[])
        analyzer = GapRiskAnalyzer()

        result = await analyzer.analyze(db, "default", earnings_tickers=set())
        assert result.rating == "LOW"
        assert result.aggregate_risk_score == 0.0

    async def test_recommendation_for_high_score(self) -> None:
        positions = [_make_position("AAPL", 100, 300.0, market_value=30000.0)]
        db = _make_db(positions=positions)
        analyzer = GapRiskAnalyzer()

        result = await analyzer.analyze(db, "default", earnings_tickers=set())
        aapl = result.positions[0]
        # 30% * 1.2 = 36 > 20 → "Consider reducing before close"
        assert aapl.recommendation == "Consider reducing before close"

    async def test_earnings_cross_reference_with_positions(self) -> None:
        positions = [
            _make_position("AAPL", 100, 150.0, market_value=15000.0),
            _make_position("MSFT", 50, 400.0, market_value=20000.0),
        ]
        db = _make_db(positions=positions)
        analyzer = GapRiskAnalyzer()

        # Only AAPL has earnings
        result = await analyzer.analyze(db, "default", earnings_tickers={"AAPL", "GOOGL"})
        assert result.earnings_tonight == ["AAPL"]  # Only held tickers
