"""Tests for the complexity detector — signal evaluation and model routing decisions.

Each signal tested individually plus accumulation, capping, and custom threshold.
"""

import numpy as np
import pandas as pd
import pytest

from src.agent.complexity_detector import ComplexityDetector, ComplexityResult

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_closes(tickers: list[str], days: int = 10) -> pd.DataFrame:
    """Generate flat close prices (no big movers by default)."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end="2026-02-05", periods=days)
    data = {}
    for t in tickers:
        base = rng.uniform(100, 300)
        # Small daily moves (~0.1%)
        returns = rng.normal(0.001, 0.005, days)
        data[t] = base * np.cumprod(1 + returns)
    return pd.DataFrame(data, index=dates)


def _base_kwargs() -> dict:
    """Default kwargs with no signals triggered."""
    return {
        "closes": _make_closes(["XLK", "XLF", "GLD"]),
        "positions": [],
        "total_value": 33_333.0,
        "peak_value": 33_333.0,
        "regime_today": "BULL",
        "regime_yesterday": "BULL",
        "vix": 15.0,
        "indicators": {},
    }


# ── Frozen dataclass ────────────────────────────────────────────────────────


class TestComplexityResult:
    def test_is_frozen(self) -> None:
        r = ComplexityResult(score=50, should_escalate=True, signals=["test"])
        with pytest.raises(AttributeError):
            r.score = 99  # type: ignore[misc]

    def test_defaults(self) -> None:
        r = ComplexityResult(score=0, should_escalate=False)
        assert r.signals == []


# ── Individual signals ──────────────────────────────────────────────────────


class TestDrawdownSignal:
    def test_no_drawdown(self) -> None:
        detector = ComplexityDetector()
        kwargs = _base_kwargs()
        result = detector.evaluate(**kwargs)
        assert result.score == 0

    def test_small_drawdown_below_threshold(self) -> None:
        detector = ComplexityDetector()
        kwargs = _base_kwargs()
        kwargs["total_value"] = 32_000.0  # ~4% below peak, under 5%
        kwargs["peak_value"] = 33_333.0
        result = detector.evaluate(**kwargs)
        assert not any("Drawdown" in s for s in result.signals)

    def test_large_drawdown(self) -> None:
        detector = ComplexityDetector()
        kwargs = _base_kwargs()
        kwargs["total_value"] = 30_000.0
        kwargs["peak_value"] = 33_333.0  # ~10% drawdown
        result = detector.evaluate(**kwargs)
        assert result.score >= 20
        assert any("Drawdown" in s for s in result.signals)


class TestRegimeChangeSignal:
    def test_no_change(self) -> None:
        detector = ComplexityDetector()
        kwargs = _base_kwargs()
        result = detector.evaluate(**kwargs)
        assert not any("Regime" in s for s in result.signals)

    def test_regime_changed(self) -> None:
        detector = ComplexityDetector()
        kwargs = _base_kwargs()
        kwargs["regime_today"] = "BEAR"
        kwargs["regime_yesterday"] = "BULL"
        result = detector.evaluate(**kwargs)
        assert result.score >= 20
        assert any("Regime changed" in s for s in result.signals)
        assert any("BULL" in s and "BEAR" in s for s in result.signals)

    def test_none_regimes_ignored(self) -> None:
        detector = ComplexityDetector()
        kwargs = _base_kwargs()
        kwargs["regime_today"] = None
        kwargs["regime_yesterday"] = None
        result = detector.evaluate(**kwargs)
        assert not any("Regime" in s for s in result.signals)


class TestVixSignal:
    def test_low_vix(self) -> None:
        detector = ComplexityDetector()
        kwargs = _base_kwargs()
        kwargs["vix"] = 15.0
        result = detector.evaluate(**kwargs)
        assert not any("VIX" in s for s in result.signals)

    def test_moderate_vix_25(self) -> None:
        detector = ComplexityDetector()
        kwargs = _base_kwargs()
        kwargs["vix"] = 28.0
        result = detector.evaluate(**kwargs)
        assert result.score >= 15
        assert any("VIX elevated at 28.0" in s for s in result.signals)

    def test_high_vix_30(self) -> None:
        detector = ComplexityDetector()
        kwargs = _base_kwargs()
        kwargs["vix"] = 35.0
        result = detector.evaluate(**kwargs)
        assert result.score >= 20
        assert any("VIX elevated at 35.0" in s for s in result.signals)

    def test_none_vix(self) -> None:
        detector = ComplexityDetector()
        kwargs = _base_kwargs()
        kwargs["vix"] = None
        result = detector.evaluate(**kwargs)
        assert not any("VIX" in s for s in result.signals)


class TestVolatilitySignal:
    def test_no_big_movers(self) -> None:
        detector = ComplexityDetector()
        kwargs = _base_kwargs()
        result = detector.evaluate(**kwargs)
        assert not any("moved >5%" in s for s in result.signals)

    def test_many_big_movers(self) -> None:
        detector = ComplexityDetector()
        tickers = ["A", "B", "C", "D", "E"]
        dates = pd.bdate_range(end="2026-02-05", periods=5)
        data = {t: [100.0, 100.0, 100.0, 100.0, 100.0] for t in tickers}
        # Make 4 tickers move > 5% on the last day
        for t in ["A", "B", "C", "D"]:
            data[t][-1] = 106.0
        closes = pd.DataFrame(data, index=dates)

        kwargs = _base_kwargs()
        kwargs["closes"] = closes
        result = detector.evaluate(**kwargs)
        assert result.score >= 15
        assert any("moved >5%" in s for s in result.signals)


class TestPositionCountSignal:
    def test_few_positions(self) -> None:
        detector = ComplexityDetector()
        kwargs = _base_kwargs()
        kwargs["positions"] = [{"ticker": "XLK"}, {"ticker": "GLD"}]
        result = detector.evaluate(**kwargs)
        assert not any("Holding" in s for s in result.signals)

    def test_many_positions(self) -> None:
        detector = ComplexityDetector()
        kwargs = _base_kwargs()
        kwargs["positions"] = [{"ticker": f"T{i}"} for i in range(8)]
        result = detector.evaluate(**kwargs)
        assert result.score >= 10
        assert any("Holding 8" in s for s in result.signals)


class TestConflictingSignals:
    def test_no_conflicts(self) -> None:
        detector = ComplexityDetector()
        kwargs = _base_kwargs()
        kwargs["indicators"] = {"XLK": {"macd": 1.5, "rsi_14": 55.0}}
        result = detector.evaluate(**kwargs)
        assert not any("Conflicting" in s for s in result.signals)

    def test_macd_positive_rsi_overbought(self) -> None:
        detector = ComplexityDetector()
        kwargs = _base_kwargs()
        kwargs["indicators"] = {"NVDA": {"macd": 2.0, "rsi_14": 75.0}}
        result = detector.evaluate(**kwargs)
        assert result.score >= 15
        assert any("Conflicting" in s and "NVDA" in s for s in result.signals)

    def test_macd_negative_not_conflicting(self) -> None:
        detector = ComplexityDetector()
        kwargs = _base_kwargs()
        kwargs["indicators"] = {"XLK": {"macd": -1.0, "rsi_14": 75.0}}
        result = detector.evaluate(**kwargs)
        assert not any("Conflicting" in s for s in result.signals)

    def test_missing_values_no_conflict(self) -> None:
        detector = ComplexityDetector()
        kwargs = _base_kwargs()
        kwargs["indicators"] = {"XLK": {"macd": None, "rsi_14": None}}
        result = detector.evaluate(**kwargs)
        assert not any("Conflicting" in s for s in result.signals)


# ── Accumulation and capping ────────────────────────────────────────────────


class TestAccumulationAndCapping:
    def test_multiple_signals_accumulate(self) -> None:
        detector = ComplexityDetector()
        kwargs = _base_kwargs()
        # Drawdown (20) + regime change (20) = 40
        kwargs["total_value"] = 30_000.0
        kwargs["peak_value"] = 33_333.0
        kwargs["regime_today"] = "BEAR"
        kwargs["regime_yesterday"] = "BULL"
        result = detector.evaluate(**kwargs)
        assert result.score >= 40
        assert len(result.signals) >= 2

    def test_score_capped_at_100(self) -> None:
        detector = ComplexityDetector()
        # Trigger all signals for maximum score
        tickers = ["A", "B", "C", "D", "E"]
        dates = pd.bdate_range(end="2026-02-05", periods=5)
        data = {t: [100.0, 100.0, 100.0, 100.0, 106.0] for t in tickers}
        closes = pd.DataFrame(data, index=dates)

        result = detector.evaluate(
            closes=closes,
            positions=[{"ticker": f"T{i}"} for i in range(8)],
            total_value=25_000.0,
            peak_value=33_333.0,
            regime_today="BEAR",
            regime_yesterday="BULL",
            vix=35.0,
            indicators={"NVDA": {"macd": 2.0, "rsi_14": 75.0}},
        )
        assert result.score == 100


# ── Threshold and escalation ────────────────────────────────────────────────


class TestThreshold:
    def test_default_threshold(self) -> None:
        detector = ComplexityDetector()
        assert detector._threshold == 50

    def test_custom_threshold(self) -> None:
        detector = ComplexityDetector(threshold=30)
        kwargs = _base_kwargs()
        # Regime change alone = 20 points, below 30
        kwargs["regime_today"] = "BEAR"
        kwargs["regime_yesterday"] = "BULL"
        result = detector.evaluate(**kwargs)
        assert result.score >= 20
        assert not result.should_escalate

    def test_escalation_at_threshold(self) -> None:
        detector = ComplexityDetector(threshold=20)
        kwargs = _base_kwargs()
        # Regime change = 20 points, exactly at threshold
        kwargs["regime_today"] = "BEAR"
        kwargs["regime_yesterday"] = "BULL"
        result = detector.evaluate(**kwargs)
        assert result.should_escalate

    def test_no_escalation_below_threshold(self) -> None:
        detector = ComplexityDetector(threshold=50)
        kwargs = _base_kwargs()
        result = detector.evaluate(**kwargs)
        assert result.score == 0
        assert not result.should_escalate
