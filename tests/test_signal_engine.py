"""Tests for the local Ticker Signal Engine."""

from datetime import datetime

import numpy as np
import pandas as pd

from src.analysis.signal_engine import (
    SignalEngine,
    TickerSignal,
    _detect_alerts,
    _raw_bollinger_pctb,
    _raw_rsi,
    _raw_volume_ratio,
    db_rows_to_signals,
    format_signals_for_agent,
    signals_to_db_rows,
)
from src.storage.models import TickerSignalRow

# ── Test data helpers ─────────────────────────────────────────────────────────


def _make_closes(tickers: list[str], days: int = 100) -> pd.DataFrame:
    """Generate synthetic close prices."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end="2026-02-05", periods=days)
    data = {}
    for t in tickers:
        base = rng.uniform(50, 300)
        returns = rng.normal(0.0005, 0.015, days)
        data[t] = base * np.cumprod(1 + returns)
    return pd.DataFrame(data, index=dates)


def _make_volumes(tickers: list[str], days: int = 100) -> pd.DataFrame:
    """Generate synthetic volume data."""
    rng = np.random.default_rng(99)
    dates = pd.bdate_range(end="2026-02-05", periods=days)
    data = {}
    for t in tickers:
        base_vol = rng.uniform(500_000, 5_000_000)
        data[t] = rng.normal(base_vol, base_vol * 0.2, days).clip(min=1000)
    return pd.DataFrame(data, index=dates)


TICKERS = ["XLK", "XLF", "XLE", "GLD", "QQQ"]


# ── Signal computation tests ─────────────────────────────────────────────────


class TestSignalComputation:
    def test_momentum_20d(self) -> None:
        closes = _make_closes(TICKERS, days=30)
        engine = SignalEngine()
        result = engine._compute_momentum_20d(closes, TICKERS)
        assert len(result) == 5
        assert all(isinstance(v, (float, np.floating)) for v in result.values)

    def test_momentum_63d(self) -> None:
        closes = _make_closes(TICKERS, days=100)
        engine = SignalEngine()
        result = engine._compute_momentum_63d(closes, TICKERS)
        assert len(result) == 5

    def test_momentum_63d_insufficient_data(self) -> None:
        closes = _make_closes(TICKERS, days=30)
        engine = SignalEngine()
        result = engine._compute_momentum_63d(closes, TICKERS)
        assert all(v == 0 for v in result.values)

    def test_rsi_score_extremes_higher(self) -> None:
        """Tickers at RSI 20 or 80 should score higher than RSI 50."""
        engine = SignalEngine()
        closes = _make_closes(TICKERS, days=100)
        result = engine._compute_rsi_score(closes, TICKERS)
        # All scores should be between 0 and 1
        assert all(0 <= v <= 1 for v in result.values)

    def test_macd_score(self) -> None:
        closes = _make_closes(TICKERS, days=100)
        engine = SignalEngine()
        result = engine._compute_macd_score(closes, TICKERS)
        assert len(result) == 5

    def test_sma_trend_range(self) -> None:
        closes = _make_closes(TICKERS, days=250)
        engine = SignalEngine()
        result = engine._compute_sma_trend(closes, TICKERS)
        assert all(0 <= v <= 3 for v in result.values)

    def test_bollinger_score(self) -> None:
        closes = _make_closes(TICKERS, days=100)
        engine = SignalEngine()
        result = engine._compute_bollinger_score(closes, TICKERS)
        assert len(result) == 5

    def test_volume_score(self) -> None:
        closes = _make_closes(TICKERS, days=100)
        volumes = _make_volumes(TICKERS, days=100)
        engine = SignalEngine()
        result = engine._compute_volume_score(closes, volumes, TICKERS)
        assert all(v > 0 for v in result.values)

    def test_volume_score_missing_ticker(self) -> None:
        closes = _make_closes(TICKERS, days=100)
        volumes = _make_volumes(["XLK", "XLF"], days=100)  # Missing XLE, GLD, QQQ
        engine = SignalEngine()
        result = engine._compute_volume_score(closes, volumes, TICKERS)
        assert result["XLE"] == 1.0  # default when missing


# ── Full engine run tests ─────────────────────────────────────────────────────


class TestSignalEngineRun:
    async def test_composite_score_range(self) -> None:
        closes = _make_closes(TICKERS, days=100)
        volumes = _make_volumes(TICKERS, days=100)
        engine = SignalEngine()
        results = await engine.run("default", closes, volumes)
        assert len(results) == 5
        for s in results:
            assert 0 <= s.composite_score <= 100

    async def test_rank_ordering(self) -> None:
        closes = _make_closes(TICKERS, days=100)
        volumes = _make_volumes(TICKERS, days=100)
        engine = SignalEngine()
        results = await engine.run("default", closes, volumes)
        ranks = [s.rank for s in results]
        assert ranks == sorted(ranks)

    async def test_all_tickers_ranked(self) -> None:
        closes = _make_closes(TICKERS, days=100)
        volumes = _make_volumes(TICKERS, days=100)
        engine = SignalEngine()
        results = await engine.run("default", closes, volumes)
        tickers = {s.ticker for s in results}
        assert tickers == set(TICKERS)

    async def test_rank_velocity_first_run_is_zero(self) -> None:
        closes = _make_closes(TICKERS, days=100)
        volumes = _make_volumes(TICKERS, days=100)
        engine = SignalEngine()
        results = await engine.run("default", closes, volumes)
        assert all(s.rank_velocity == 0 for s in results)
        assert all(s.prev_rank is None for s in results)

    async def test_rank_velocity_on_second_run(self) -> None:
        closes = _make_closes(TICKERS, days=100)
        volumes = _make_volumes(TICKERS, days=100)
        engine = SignalEngine()
        await engine.run("default", closes, volumes)
        results = await engine.run("default", closes, volumes)
        # Same data → same ranks → velocity should be 0
        assert all(s.rank_velocity == 0 for s in results)
        assert all(s.prev_rank is not None for s in results)

    async def test_insufficient_data_returns_empty(self) -> None:
        closes = _make_closes(TICKERS, days=10)
        volumes = _make_volumes(TICKERS, days=10)
        engine = SignalEngine()
        results = await engine.run("default", closes, volumes)
        assert results == []

    async def test_empty_df_returns_empty(self) -> None:
        engine = SignalEngine()
        results = await engine.run("default", pd.DataFrame(), pd.DataFrame())
        assert results == []

    async def test_tenant_isolation(self) -> None:
        """Different tenants maintain separate state."""
        closes = _make_closes(TICKERS, days=100)
        volumes = _make_volumes(TICKERS, days=100)
        engine = SignalEngine()
        await engine.run("tenant_a", closes, volumes)
        results_b = await engine.run("tenant_b", closes, volumes)
        # tenant_b has no prev_ranks yet
        assert all(s.prev_rank is None for s in results_b)

    async def test_scored_at_is_set(self) -> None:
        closes = _make_closes(TICKERS, days=100)
        volumes = _make_volumes(TICKERS, days=100)
        engine = SignalEngine()
        results = await engine.run("default", closes, volumes)
        for s in results:
            assert isinstance(s.scored_at, datetime)


# ── Alert detection tests ─────────────────────────────────────────────────────


class TestAlertDetection:
    def test_rank_jump_up(self) -> None:
        alerts = _detect_alerts(
            rank=5, prev_rank=20, hours_elapsed=1.0,
            indicators={}, prev_indicators={},
        )
        assert any("rank_jump_up" in a for a in alerts)

    def test_rank_jump_down(self) -> None:
        alerts = _detect_alerts(
            rank=25, prev_rank=10, hours_elapsed=1.0,
            indicators={}, prev_indicators={},
        )
        assert any("rank_jump_down" in a for a in alerts)

    def test_no_rank_jump_small_change(self) -> None:
        alerts = _detect_alerts(
            rank=10, prev_rank=15, hours_elapsed=1.0,
            indicators={}, prev_indicators={},
        )
        assert not any("rank_jump" in a for a in alerts)

    def test_rsi_oversold_cross(self) -> None:
        alerts = _detect_alerts(
            rank=1, prev_rank=1, hours_elapsed=1.0,
            indicators={"rsi": 28.0},
            prev_indicators={"rsi": 32.0},
        )
        assert "rsi_oversold_cross" in alerts

    def test_rsi_overbought_cross(self) -> None:
        alerts = _detect_alerts(
            rank=1, prev_rank=1, hours_elapsed=1.0,
            indicators={"rsi": 72.0},
            prev_indicators={"rsi": 68.0},
        )
        assert "rsi_overbought_cross" in alerts

    def test_rsi_no_cross_when_already_oversold(self) -> None:
        """No alert if RSI was already below 30 (not a transition)."""
        alerts = _detect_alerts(
            rank=1, prev_rank=1, hours_elapsed=1.0,
            indicators={"rsi": 25.0},
            prev_indicators={"rsi": 28.0},
        )
        assert "rsi_oversold_cross" not in alerts

    def test_golden_cross(self) -> None:
        alerts = _detect_alerts(
            rank=1, prev_rank=1, hours_elapsed=1.0,
            indicators={"sma20": 105.0, "sma50": 100.0},
            prev_indicators={"sma20": 99.0, "sma50": 100.0},
        )
        assert "golden_cross" in alerts

    def test_death_cross(self) -> None:
        alerts = _detect_alerts(
            rank=1, prev_rank=1, hours_elapsed=1.0,
            indicators={"sma20": 95.0, "sma50": 100.0},
            prev_indicators={"sma20": 101.0, "sma50": 100.0},
        )
        assert "death_cross" in alerts

    def test_volume_spike(self) -> None:
        alerts = _detect_alerts(
            rank=1, prev_rank=1, hours_elapsed=1.0,
            indicators={"volume_ratio": 2.5},
            prev_indicators={},
        )
        assert any("volume_spike" in a for a in alerts)

    def test_bollinger_upper_breakout(self) -> None:
        alerts = _detect_alerts(
            rank=1, prev_rank=1, hours_elapsed=1.0,
            indicators={"bollinger_pct_b": 1.05},
            prev_indicators={},
        )
        assert "bollinger_upper_breakout" in alerts

    def test_bollinger_lower_breakout(self) -> None:
        alerts = _detect_alerts(
            rank=1, prev_rank=1, hours_elapsed=1.0,
            indicators={"bollinger_pct_b": -0.05},
            prev_indicators={},
        )
        assert "bollinger_lower_breakout" in alerts

    def test_no_alerts_normal_conditions(self) -> None:
        alerts = _detect_alerts(
            rank=10, prev_rank=12, hours_elapsed=1.0,
            indicators={"rsi": 50.0, "sma20": 100, "sma50": 95, "volume_ratio": 1.1, "bollinger_pct_b": 0.5},
            prev_indicators={"rsi": 52.0, "sma20": 99, "sma50": 95},
        )
        assert alerts == []


# ── Raw indicator helper tests ────────────────────────────────────────────────


class TestRawIndicators:
    def test_raw_rsi_normal(self) -> None:
        closes = _make_closes(["XLK"], days=100)
        rsi = _raw_rsi(closes, "XLK")
        assert 0 <= rsi <= 100

    def test_raw_rsi_insufficient_data(self) -> None:
        closes = _make_closes(["XLK"], days=5)
        assert _raw_rsi(closes, "XLK") == 50.0

    def test_raw_bollinger_pctb_normal(self) -> None:
        closes = _make_closes(["XLK"], days=100)
        pctb = _raw_bollinger_pctb(closes, "XLK")
        assert isinstance(pctb, float)

    def test_raw_volume_ratio_normal(self) -> None:
        closes = _make_closes(TICKERS, days=100)
        volumes = _make_volumes(TICKERS, days=100)
        ratio = _raw_volume_ratio(closes, volumes, "XLK")
        assert ratio > 0

    def test_raw_volume_ratio_missing_ticker(self) -> None:
        closes = _make_closes(TICKERS, days=100)
        volumes = _make_volumes(["XLF"], days=100)
        assert _raw_volume_ratio(closes, volumes, "XLK") == 1.0


# ── Format for agent tests ────────────────────────────────────────────────────


class TestFormatForAgent:
    def test_format_non_held_section(self) -> None:
        signals = [
            TickerSignal(
                ticker="XLK", composite_score=85, rank=1, prev_rank=None,
                rank_velocity=0, momentum_20d=0.08, momentum_63d=0.15,
                rsi=55, macd_histogram=0.5, sma_trend_score=3,
                bollinger_pct_b=0.7, volume_ratio=1.2, alerts=[],
            ),
            TickerSignal(
                ticker="XLF", composite_score=60, rank=2, prev_rank=None,
                rank_velocity=0, momentum_20d=0.03, momentum_63d=0.05,
                rsi=45, macd_histogram=0.1, sma_trend_score=2,
                bollinger_pct_b=0.5, volume_ratio=1.0, alerts=[],
            ),
        ]
        result = format_signals_for_agent(signals, held_tickers={"XLF"})
        assert "XLK" in result
        assert "Top non-held" in result

    def test_format_alerts_section(self) -> None:
        signals = [
            TickerSignal(
                ticker="XLE", composite_score=70, rank=1, prev_rank=10,
                rank_velocity=9, momentum_20d=0.05, momentum_63d=0.1,
                rsi=28, macd_histogram=0.2, sma_trend_score=2,
                bollinger_pct_b=0.3, volume_ratio=2.5,
                alerts=["rsi_oversold_cross", "volume_spike_2.5x"],
            ),
        ]
        result = format_signals_for_agent(signals, held_tickers=set())
        assert "Alerts triggered" in result
        assert "rsi_oversold_cross" in result

    def test_format_movers_section(self) -> None:
        signals = [
            TickerSignal(
                ticker="NVDA", composite_score=90, rank=3, prev_rank=28,
                rank_velocity=25, momentum_20d=0.1, momentum_63d=0.2,
                rsi=62, macd_histogram=1.0, sma_trend_score=3,
                bollinger_pct_b=0.8, volume_ratio=3.2, alerts=[],
            ),
        ]
        result = format_signals_for_agent(signals, held_tickers=set())
        assert "Biggest movers" in result
        assert "NVDA" in result

    def test_format_held_marker(self) -> None:
        signals = [
            TickerSignal(
                ticker="XLK", composite_score=80, rank=1, prev_rank=5,
                rank_velocity=4, momentum_20d=0.06, momentum_63d=0.12,
                rsi=55, macd_histogram=0.3, sma_trend_score=3,
                bollinger_pct_b=0.6, volume_ratio=1.5, alerts=[],
            ),
        ]
        result = format_signals_for_agent(signals, held_tickers={"XLK"})
        assert "[HELD]" in result

    def test_format_empty_signals(self) -> None:
        result = format_signals_for_agent([], held_tickers=set())
        assert result == ""


# ── DB conversion tests ───────────────────────────────────────────────────────


class TestSignalsToDbRows:
    def test_conversion(self) -> None:
        signals = [
            TickerSignal(
                ticker="XLK", composite_score=85.3, rank=1, prev_rank=2,
                rank_velocity=1.0, momentum_20d=0.08, momentum_63d=0.15,
                rsi=55, macd_histogram=0.5, sma_trend_score=3,
                bollinger_pct_b=0.7, volume_ratio=1.2,
                alerts=["golden_cross"],
            ),
        ]
        rows = signals_to_db_rows("default", signals)
        assert len(rows) == 1
        assert rows[0].ticker == "XLK"
        assert rows[0].tenant_id == "default"
        assert rows[0].composite_score == 85.3
        assert rows[0].alerts == '["golden_cross"]'

    def test_empty_alerts_serialized(self) -> None:
        signals = [
            TickerSignal(
                ticker="XLF", composite_score=50, rank=5, prev_rank=None,
                rank_velocity=0, momentum_20d=0, momentum_63d=0,
                rsi=50, macd_histogram=0, sma_trend_score=1,
                bollinger_pct_b=0.5, volume_ratio=1.0, alerts=[],
            ),
        ]
        rows = signals_to_db_rows("t1", signals)
        assert rows[0].alerts == "[]"


# ── DB row → TickerSignal conversion tests ────────────────────────────────────


class TestDbRowsToSignals:
    def test_round_trip(self) -> None:
        """signals → db rows → signals preserves data."""
        original = [
            TickerSignal(
                ticker="XLK", composite_score=85.3, rank=1, prev_rank=2,
                rank_velocity=1.5, momentum_20d=0.08, momentum_63d=0.15,
                rsi=55, macd_histogram=0.5, sma_trend_score=3,
                bollinger_pct_b=0.7, volume_ratio=1.2,
                alerts=["golden_cross"],
            ),
        ]
        rows = signals_to_db_rows("default", original)
        restored = db_rows_to_signals(rows)
        assert len(restored) == 1
        assert restored[0].ticker == "XLK"
        assert restored[0].composite_score == 85.3
        assert restored[0].rank == 1
        assert restored[0].prev_rank == 2
        assert restored[0].alerts == ["golden_cross"]

    def test_null_fields_get_defaults(self) -> None:
        """NULL indicator fields default to safe values."""
        row = TickerSignalRow(
            tenant_id="t1", ticker="BTC-USD", composite_score=40, rank=10,
            prev_rank=None, rank_velocity=0.0,
            momentum_20d=None, momentum_63d=None, rsi=None,
            macd_histogram=None, sma_trend_score=None,
            bollinger_pct_b=None, volume_ratio=None,
            alerts=None, scored_at=None,
        )
        restored = db_rows_to_signals([row])
        assert restored[0].momentum_20d == 0.0
        assert restored[0].rsi == 50.0
        assert restored[0].bollinger_pct_b == 0.5
        assert restored[0].volume_ratio == 1.0
        assert restored[0].alerts == []

    def test_empty_list(self) -> None:
        assert db_rows_to_signals([]) == []
