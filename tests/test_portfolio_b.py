"""Tests for Portfolio B: Sector Rotation + Composite Scoring + Regime Detection."""

import numpy as np
import pandas as pd
import pytest

from config.risk_rules import RISK_RULES
from src.storage.models import Regime
from src.strategies.portfolio_b import CompositeScorer, RegimeDetector, SectorRotationStrategy


@pytest.fixture
def market_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic closes and volumes for Portfolio B universe."""
    np.random.seed(42)
    dates = pd.bdate_range(end="2026-02-05", periods=250)
    tickers = [
        "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE",
        "QQQ", "SMH", "XBI", "IWM", "EFA", "EEM", "TLT", "HYG", "GDX", "ARKK",
        "SH", "PSQ", "TBF", "GLD", "SLV", "USO",
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    ]

    closes_data = {}
    volumes_data = {}
    for i, t in enumerate(tickers):
        drift = 0.3 - i * 0.02
        closes_data[t] = 100 + np.cumsum(np.random.normal(drift, 1.5, 250))
        volumes_data[t] = np.random.uniform(1e6, 1e8, 250)

    closes = pd.DataFrame(closes_data, index=dates)
    volumes = pd.DataFrame(volumes_data, index=dates)
    return closes, volumes


class TestRegimeDetector:
    def test_bull_regime(self) -> None:
        # SPY trending up, above both SMAs, low VIX
        prices = pd.Series(np.linspace(400, 500, 250))
        detector = RegimeDetector()
        regime = detector.detect(prices, yield_curve=1.5, vix=15)
        assert regime == Regime.BULL

    def test_bear_regime(self) -> None:
        # SPY trending down, below SMA200, inverted yield curve
        prices = pd.Series(np.linspace(500, 380, 250))
        detector = RegimeDetector()
        regime = detector.detect(prices, yield_curve=-0.5, vix=35)
        assert regime == Regime.BEAR

    def test_neutral_with_insufficient_data(self) -> None:
        prices = pd.Series(np.linspace(400, 450, 50))
        detector = RegimeDetector()
        regime = detector.detect(prices)
        assert regime == Regime.NEUTRAL

    def test_rotation_regime(self) -> None:
        # Above 200 SMA but below 50 SMA (pullback from highs)
        prices = list(np.linspace(400, 500, 200))  # long uptrend
        prices += list(np.linspace(500, 470, 50))  # recent pullback
        spy = pd.Series(prices)
        detector = RegimeDetector()
        regime = detector.detect(spy, yield_curve=0.5, vix=22)
        assert regime == Regime.ROTATION


class TestCompositeScorer:
    def test_composite_returns_all_tickers(
        self, market_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        closes, volumes = market_data
        scorer = CompositeScorer()
        result = scorer.compute_composite(closes, volumes, regime=Regime.NEUTRAL)
        assert len(result) > 0
        assert "composite_score" in result.columns
        assert "ticker" in result.columns

    def test_scores_between_0_and_1(
        self, market_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        closes, volumes = market_data
        scorer = CompositeScorer()
        result = scorer.compute_composite(closes, volumes)
        # Composite is weighted sum of [0,1] scores, so should be in [0,1]
        assert (result["composite_score"] >= 0).all()
        assert (result["composite_score"] <= 1).all()

    def test_sorted_descending(
        self, market_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        closes, volumes = market_data
        scorer = CompositeScorer()
        result = scorer.compute_composite(closes, volumes)
        scores = result["composite_score"].values
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_btc_risk_crash_signal(self) -> None:
        scorer = CompositeScorer()
        # BTC down 30% — should return 0 (full risk-off)
        btc = pd.Series(np.linspace(100, 65, 80))
        score = scorer.score_btc_risk(btc)
        assert score == 0.0

    def test_btc_risk_rally_signal(self) -> None:
        scorer = CompositeScorer()
        # BTC up 15%
        btc = pd.Series(np.linspace(100, 120, 80))
        score = scorer.score_btc_risk(btc)
        assert score > 0.5

    def test_regime_bear_boosts_defensives(self) -> None:
        scorer = CompositeScorer()
        tickers = list(RISK_RULES.tech_etfs) + list(RISK_RULES.defensive_tickers)
        scores = scorer._regime_adjustment(tickers, Regime.BEAR)
        for t in RISK_RULES.defensive_tickers:
            if t in tickers:
                assert scores[t] == 0.9
        for t in RISK_RULES.tech_etfs:
            if t in tickers:
                assert scores[t] == 0.1


class TestSectorRotationStrategy:
    def test_select_positions_returns_top_n(
        self, market_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        closes, volumes = market_data
        strategy = SectorRotationStrategy()
        scores, regime = strategy.analyze(closes, volumes)
        selected = strategy.select_positions(scores, regime, top_n=3)
        assert len(selected) == 3

    def test_bear_limits_tech(
        self, market_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        closes, volumes = market_data
        strategy = SectorRotationStrategy()
        scores, _ = strategy.analyze(closes, volumes)
        selected = strategy.select_positions(scores, Regime.BEAR, top_n=3)
        tech_count = sum(1 for t in selected if t in RISK_RULES.tech_etfs)
        assert tech_count <= 1  # max 1 tech in BEAR

    def test_generate_trades_from_empty(
        self, market_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        closes, volumes = market_data
        strategy = SectorRotationStrategy()
        scores, regime = strategy.analyze(closes, volumes)
        selected = strategy.select_positions(scores, regime)
        trades = strategy.generate_trades(
            selected_tickers=selected,
            current_positions={},
            cash=33_333.0,
            latest_prices=closes.iloc[-1],
        )
        buy_trades = [t for t in trades if t.side.value == "BUY"]
        assert len(buy_trades) == len(selected)

    def test_scores_to_db_rows(
        self, market_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        closes, volumes = market_data
        strategy = SectorRotationStrategy()
        scores, regime = strategy.analyze(closes, volumes)
        from datetime import date
        rows = strategy.scores_to_db_rows(scores, date(2026, 2, 5), regime)
        assert len(rows) == len(scores)
        assert rows[0].regime is not None
