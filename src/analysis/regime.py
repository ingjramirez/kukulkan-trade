"""Market regime classifier using SPY trend, VIX, and breadth signals.

Classifies the current market into one of five regimes:
BULL, CONSOLIDATION, CORRECTION, BEAR, CRISIS.

Used by the orchestrator to give the AI agent situational context.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class MarketRegime(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    CORRECTION = "CORRECTION"
    CRISIS = "CRISIS"
    CONSOLIDATION = "CONSOLIDATION"


@dataclass(frozen=True)
class RegimeResult:
    """Output of regime classification."""

    regime: MarketRegime
    spy_vs_sma200: float | None  # fractional, e.g. 0.032 = 3.2% above
    drawdown_from_52w: float | None  # negative pct, e.g. -7.5
    vix: float | None
    breadth_pct: float | None  # % of tickers above their SMA50
    summary: str  # 1-line human-readable


class RegimeClassifier:
    """Classifies market regime from price data and VIX."""

    def classify(
        self,
        closes: pd.DataFrame,
        vix: float | None = None,
    ) -> RegimeResult:
        """Classify market regime based on SPY trend, drawdown, VIX, and breadth.

        Classification priority:
        1. Insufficient data -> CONSOLIDATION
        2. CRISIS: vix > 35 AND spy < sma200
        3. BEAR: spy < sma200 AND vix > 25
        4. CORRECTION: spy > sma200 but drawdown < -5%
        5. BULL: spy > sma200 AND drawdown > -5% AND (vix < 20 or unknown)
        6. CONSOLIDATION: everything else

        Args:
            closes: DataFrame with tickers as columns. Must contain 'SPY'.
            vix: Current VIX value. None if unavailable.

        Returns:
            RegimeResult with classification and supporting metrics.
        """
        # Check for SPY data
        if "SPY" not in closes.columns:
            return RegimeResult(
                regime=MarketRegime.CONSOLIDATION,
                spy_vs_sma200=None,
                drawdown_from_52w=None,
                vix=vix,
                breadth_pct=None,
                summary="Insufficient data — SPY not in universe",
            )

        spy = closes["SPY"].dropna()
        if len(spy) < 50:
            return RegimeResult(
                regime=MarketRegime.CONSOLIDATION,
                spy_vs_sma200=None,
                drawdown_from_52w=None,
                vix=vix,
                breadth_pct=None,
                summary="Insufficient data — fewer than 50 SPY data points",
            )

        # Compute metrics
        current_price = float(spy.iloc[-1])
        sma200 = float(spy.rolling(200).mean().iloc[-1]) if len(spy) >= 200 else None
        spy_vs_sma200 = (
            (current_price - sma200) / sma200 if sma200 and not np.isnan(sma200) else None
        )

        high_252 = float(spy.tail(252).max())
        drawdown = ((current_price - high_252) / high_252) * 100

        # Breadth: % of tickers above their SMA50
        breadth_pct = self._compute_breadth(closes)

        # Classification (priority order)
        regime, summary = self._classify_regime(
            spy_vs_sma200, drawdown, vix, breadth_pct, sma200,
        )

        return RegimeResult(
            regime=regime,
            spy_vs_sma200=round(spy_vs_sma200, 4) if spy_vs_sma200 is not None else None,
            drawdown_from_52w=round(drawdown, 2),
            vix=vix,
            breadth_pct=round(breadth_pct, 1) if breadth_pct is not None else None,
            summary=summary,
        )

    def _classify_regime(
        self,
        spy_vs_sma200: float | None,
        drawdown: float,
        vix: float | None,
        breadth_pct: float | None,
        sma200: float | None,
    ) -> tuple[MarketRegime, str]:
        """Apply classification rules in priority order."""
        # No SMA200 available (< 200 days of data)
        if spy_vs_sma200 is None or sma200 is None:
            return (
                MarketRegime.CONSOLIDATION,
                "Insufficient history for SMA200 — defaulting to CONSOLIDATION",
            )

        above_sma200 = spy_vs_sma200 > 0

        # CRISIS: VIX > 35 AND below SMA200
        if vix is not None and vix > 35 and not above_sma200:
            return (
                MarketRegime.CRISIS,
                f"CRISIS: VIX {vix:.0f}, SPY {spy_vs_sma200:+.1%} vs SMA200. "
                f"Drawdown {drawdown:+.1f}%.",
            )

        # BEAR: below SMA200 AND VIX > 25
        if not above_sma200 and vix is not None and vix > 25:
            return (
                MarketRegime.BEAR,
                f"BEAR: SPY {spy_vs_sma200:+.1%} below SMA200, VIX {vix:.0f}. "
                f"Drawdown {drawdown:+.1f}%.",
            )

        # Below SMA200 but VIX unknown → ambiguous → CONSOLIDATION
        if not above_sma200 and vix is None:
            return (
                MarketRegime.CONSOLIDATION,
                f"SPY below SMA200 ({spy_vs_sma200:+.1%}) but VIX unavailable "
                f"— defaulting to CONSOLIDATION.",
            )

        # Below SMA200 but VIX <= 25 → mild bear → CONSOLIDATION
        if not above_sma200:
            return (
                MarketRegime.CONSOLIDATION,
                f"SPY {spy_vs_sma200:+.1%} below SMA200, VIX {vix:.0f} "
                f"(moderate) — CONSOLIDATION.",
            )

        # CORRECTION: above SMA200 but drawdown < -5%
        if drawdown < -5:
            return (
                MarketRegime.CORRECTION,
                f"CORRECTION: SPY above SMA200 ({spy_vs_sma200:+.1%}) but "
                f"drawdown {drawdown:+.1f}% from 52w high.",
            )

        # BULL: above SMA200 AND drawdown > -5% AND (VIX < 20 or unknown)
        if vix is None or vix < 20:
            breadth_str = f", breadth {breadth_pct:.0f}%" if breadth_pct is not None else ""
            return (
                MarketRegime.BULL,
                f"BULL: SPY {spy_vs_sma200:+.1%} above SMA200, "
                f"drawdown {drawdown:+.1f}%"
                f"{', VIX ' + f'{vix:.0f}' if vix is not None else ''}"
                f"{breadth_str}.",
            )

        # Everything else → CONSOLIDATION (e.g. above SMA200, VIX 20-25, mild drawdown)
        return (
            MarketRegime.CONSOLIDATION,
            f"SPY {spy_vs_sma200:+.1%} above SMA200, VIX {vix:.0f}, "
            f"drawdown {drawdown:+.1f}% — CONSOLIDATION.",
        )

    def _compute_breadth(self, closes: pd.DataFrame) -> float | None:
        """Compute market breadth as % of tickers above their SMA50.

        Args:
            closes: Full DataFrame of close prices.

        Returns:
            Percentage (0-100) or None if insufficient data.
        """
        # Exclude SPY itself from breadth calculation
        tickers = [c for c in closes.columns if c != "SPY"]
        if not tickers:
            return None

        above_count = 0
        valid_count = 0
        for t in tickers:
            series = closes[t].dropna()
            if len(series) < 50:
                continue
            sma50 = series.rolling(50).mean().iloc[-1]
            if np.isnan(sma50):
                continue
            valid_count += 1
            if float(series.iloc[-1]) > sma50:
                above_count += 1

        if valid_count == 0:
            return None
        return (above_count / valid_count) * 100
