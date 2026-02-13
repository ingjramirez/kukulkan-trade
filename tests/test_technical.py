"""Tests for technical analysis indicators."""

import numpy as np
import pandas as pd

from src.analysis.technical import (
    compute_all_indicators,
    compute_bollinger_bands,
    compute_macd,
    compute_rsi,
    compute_sma,
)


def _make_price_series(n: int = 200) -> pd.Series:
    """Create a synthetic price series for testing."""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.normal(0.1, 1.5, n))
    dates = pd.bdate_range(end="2026-02-05", periods=n)
    return pd.Series(prices, index=dates, name="Close")


class TestRSI:
    def test_rsi_range(self) -> None:
        closes = _make_price_series()
        rsi = compute_rsi(closes)
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_length(self) -> None:
        closes = _make_price_series()
        rsi = compute_rsi(closes, window=14)
        assert len(rsi) == len(closes)

    def test_rsi_custom_window(self) -> None:
        closes = _make_price_series()
        rsi_7 = compute_rsi(closes, window=7)
        rsi_21 = compute_rsi(closes, window=21)
        # Shorter window RSI should be more volatile (higher std)
        assert rsi_7.dropna().std() > rsi_21.dropna().std()


class TestMACD:
    def test_macd_columns(self) -> None:
        closes = _make_price_series()
        result = compute_macd(closes)
        assert list(result.columns) == ["macd", "signal", "histogram"]

    def test_histogram_equals_macd_minus_signal(self) -> None:
        closes = _make_price_series()
        result = compute_macd(closes)
        valid = result.dropna()
        diff = (valid["macd"] - valid["signal"] - valid["histogram"]).abs()
        assert (diff < 1e-10).all()


class TestSMA:
    def test_sma_smoothness(self) -> None:
        closes = _make_price_series()
        sma = compute_sma(closes, window=20)
        valid_sma = sma.dropna()
        valid_close = closes.loc[valid_sma.index]
        # SMA should be smoother (lower std) than raw prices
        assert valid_sma.std() < valid_close.std()

    def test_sma_length(self) -> None:
        closes = _make_price_series(50)
        sma = compute_sma(closes, window=20)
        assert len(sma) == 50


class TestBollingerBands:
    def test_band_ordering(self) -> None:
        closes = _make_price_series()
        bb = compute_bollinger_bands(closes)
        valid = bb.dropna()
        assert (valid["upper"] >= valid["middle"]).all()
        assert (valid["middle"] >= valid["lower"]).all()

    def test_columns(self) -> None:
        closes = _make_price_series()
        bb = compute_bollinger_bands(closes)
        assert list(bb.columns) == ["upper", "middle", "lower"]


class TestComputeAllIndicators:
    def test_all_columns_present(self) -> None:
        closes = _make_price_series(250)
        result = compute_all_indicators(closes)
        expected_cols = [
            "close",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "sma_20",
            "sma_50",
            "sma_200",
            "bb_upper",
            "bb_middle",
            "bb_lower",
        ]
        assert list(result.columns) == expected_cols

    def test_same_index_as_input(self) -> None:
        closes = _make_price_series(250)
        result = compute_all_indicators(closes)
        assert len(result) == len(closes)
