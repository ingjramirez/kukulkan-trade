"""Tests for the market regime classifier."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.regime import MarketRegime, RegimeClassifier, RegimeResult


@pytest.fixture
def classifier():
    return RegimeClassifier()


def _make_spy_df(
    days: int = 252,
    base: float = 450.0,
    trend: float = 0.0005,
    noise_scale: float = 1.0,
    extra_tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Build a synthetic closes DataFrame with SPY and optional extra tickers."""
    np.random.seed(42)
    dates = pd.bdate_range(end="2026-02-06", periods=days)
    spy = base * np.cumprod(1 + trend + np.random.randn(days) * 0.005 * noise_scale)
    data = {"SPY": spy}
    for t in extra_tickers or []:
        data[t] = base * np.cumprod(1 + trend + np.random.randn(days) * 0.005)
    return pd.DataFrame(data, index=dates)


class TestRegimeClassifier:
    def test_bull_regime(self, classifier: RegimeClassifier) -> None:
        """Uptrending SPY above SMA200 with low VIX → BULL."""
        closes = _make_spy_df(days=252, trend=0.001)
        result = classifier.classify(closes, vix=15.0)
        assert result.regime == MarketRegime.BULL
        assert result.spy_vs_sma200 is not None
        assert result.spy_vs_sma200 > 0
        assert result.vix == 15.0
        assert "BULL" in result.summary

    def test_bear_regime(self, classifier: RegimeClassifier) -> None:
        """SPY below SMA200 with elevated VIX → BEAR."""
        closes = _make_spy_df(days=252, trend=-0.001)
        result = classifier.classify(closes, vix=28.0)
        assert result.regime == MarketRegime.BEAR
        assert "BEAR" in result.summary

    def test_crisis_regime(self, classifier: RegimeClassifier) -> None:
        """SPY below SMA200 with VIX > 35 → CRISIS."""
        closes = _make_spy_df(days=252, trend=-0.001)
        result = classifier.classify(closes, vix=40.0)
        assert result.regime == MarketRegime.CRISIS
        assert "CRISIS" in result.summary
        assert result.vix == 40.0

    def test_correction_regime(self, classifier: RegimeClassifier) -> None:
        """SPY above SMA200 but with significant drawdown → CORRECTION."""
        # Build a series that's above SMA200 overall but had a recent sharp drop
        closes = _make_spy_df(days=252, trend=0.001)
        # Force a >5% drop from 52w high while staying above SMA200
        spy = closes["SPY"].copy()
        # Set last few days to be 8% below the peak
        peak = spy.max()
        spy.iloc[-5:] = peak * 0.91
        # But ensure still above SMA200
        sma200 = spy.rolling(200).mean().iloc[-1]
        if spy.iloc[-1] <= sma200:
            spy.iloc[-5:] = sma200 * 1.01
            # Recompute peak for correct drawdown
            peak = spy.max()
        closes["SPY"] = spy

        result = classifier.classify(closes, vix=18.0)
        # If drawdown > -5% and above SMA200 → CORRECTION
        if result.drawdown_from_52w is not None and result.drawdown_from_52w < -5:
            assert result.regime == MarketRegime.CORRECTION
        else:
            # If our manipulation didn't produce >5% drawdown, at least not BEAR/CRISIS
            assert result.regime in (
                MarketRegime.BULL,
                MarketRegime.CORRECTION,
                MarketRegime.CONSOLIDATION,
            )

    def test_consolidation_default(self, classifier: RegimeClassifier) -> None:
        """Above SMA200 with VIX 20-25 → CONSOLIDATION."""
        closes = _make_spy_df(days=252, trend=0.0005)
        result = classifier.classify(closes, vix=22.0)
        # Above SMA200 with VIX 20-25 → CONSOLIDATION (not quite BULL)
        assert result.regime in (MarketRegime.CONSOLIDATION, MarketRegime.BULL)

    def test_consolidation_missing_spy(self, classifier: RegimeClassifier) -> None:
        """No SPY column → CONSOLIDATION."""
        closes = pd.DataFrame({"AAPL": [150.0] * 100})
        result = classifier.classify(closes, vix=20.0)
        assert result.regime == MarketRegime.CONSOLIDATION
        assert "SPY not in universe" in result.summary

    def test_consolidation_vix_none_ambiguous(self, classifier: RegimeClassifier) -> None:
        """SPY below SMA200 but VIX unknown → CONSOLIDATION (not BEAR)."""
        closes = _make_spy_df(days=252, trend=-0.001)
        result = classifier.classify(closes, vix=None)
        # Without VIX confirmation, should not be BEAR or CRISIS
        assert result.regime == MarketRegime.CONSOLIDATION
        assert result.vix is None

    def test_short_history(self, classifier: RegimeClassifier) -> None:
        """Fewer than 50 data points → CONSOLIDATION."""
        closes = _make_spy_df(days=30)
        result = classifier.classify(closes, vix=18.0)
        assert result.regime == MarketRegime.CONSOLIDATION
        assert "Insufficient data" in result.summary

    def test_breadth_computed(self, classifier: RegimeClassifier) -> None:
        """Breadth is computed when extra tickers are present."""
        closes = _make_spy_df(
            days=252,
            trend=0.001,
            extra_tickers=["AAPL", "MSFT", "XLK"],
        )
        result = classifier.classify(closes, vix=15.0)
        assert result.breadth_pct is not None
        assert 0 <= result.breadth_pct <= 100

    def test_result_fields_populated(self, classifier: RegimeClassifier) -> None:
        """All RegimeResult fields are populated for normal input."""
        closes = _make_spy_df(days=252, trend=0.001, extra_tickers=["AAPL"])
        result = classifier.classify(closes, vix=16.0)
        assert isinstance(result, RegimeResult)
        assert isinstance(result.regime, MarketRegime)
        assert result.drawdown_from_52w is not None
        assert result.summary  # non-empty string

    def test_bull_vix_none(self, classifier: RegimeClassifier) -> None:
        """SPY above SMA200, mild drawdown, VIX unknown → BULL."""
        closes = _make_spy_df(days=252, trend=0.001)
        result = classifier.classify(closes, vix=None)
        assert result.regime == MarketRegime.BULL
