"""Complexity detector for Portfolio B model routing.

Evaluates market conditions to determine if the AI agent should escalate
from Sonnet to Opus for deeper reasoning on complex days.
"""

from dataclasses import dataclass, field

import pandas as pd
import structlog

from config.strategies import PORTFOLIO_B

log = structlog.get_logger()


@dataclass(frozen=True)
class ComplexityResult:
    """Result of complexity evaluation."""

    score: int
    should_escalate: bool
    signals: list[str] = field(default_factory=list)


class ComplexityDetector:
    """Evaluates 6 market complexity signals to decide model routing."""

    def __init__(self, threshold: int | None = None) -> None:
        self._threshold = threshold if threshold is not None else PORTFOLIO_B.escalation_threshold

    def evaluate(
        self,
        closes: pd.DataFrame,
        positions: list[dict],
        total_value: float,
        peak_value: float,
        regime_today: str | None,
        regime_yesterday: str | None,
        vix: float | None,
        indicators: dict[str, dict],
    ) -> ComplexityResult:
        """Evaluate market complexity across 6 signals.

        Args:
            closes: DataFrame of close prices (tickers as columns).
            positions: List of position dicts with 'ticker' key.
            total_value: Current portfolio value.
            peak_value: Historical peak portfolio value.
            regime_today: Current regime string (e.g. "BULL").
            regime_yesterday: Previous regime string.
            vix: Current VIX value.
            indicators: Dict of ticker -> dict with rsi_14, macd keys (held tickers only).

        Returns:
            ComplexityResult with score, should_escalate flag, and human-readable signals.
        """
        score = 0
        signals: list[str] = []

        # 1. Drawdown: Portfolio B > 5% below peak
        if peak_value > 0 and total_value < peak_value:
            drawdown_pct = ((peak_value - total_value) / peak_value) * 100
            if drawdown_pct > 5.0:
                score += 20
                signals.append(f"Drawdown {drawdown_pct:.1f}% from peak")

        # 2. Regime change: today != yesterday
        if (
            regime_today is not None
            and regime_yesterday is not None
            and regime_today != regime_yesterday
        ):
            score += 20
            signals.append(f"Regime changed: {regime_yesterday} → {regime_today}")

        # 3. VIX spike
        if vix is not None:
            if vix > 30:
                score += 20
                signals.append(f"VIX elevated at {vix:.1f}")
            elif vix > 25:
                score += 15
                signals.append(f"VIX elevated at {vix:.1f}")

        # 4. High volatility: >= 3 tickers moved > 5% today
        if len(closes) >= 2:
            today_prices = closes.iloc[-1]
            yesterday_prices = closes.iloc[-2]
            pct_changes = ((today_prices - yesterday_prices) / yesterday_prices).abs()
            big_movers = (pct_changes > 0.05).sum()
            if big_movers >= 3:
                score += 15
                signals.append(f"{big_movers} tickers moved >5% today")

        # 5. Large position count: > 7 positions held
        if len(positions) > 7:
            score += 10
            signals.append(f"Holding {len(positions)} positions")

        # 6. Conflicting indicators: held ticker with MACD > 0 AND RSI > 70
        conflicting = []
        for ticker, ind in indicators.items():
            macd = ind.get("macd")
            rsi = ind.get("rsi_14")
            if macd is not None and rsi is not None:
                if macd > 0 and rsi > 70:
                    conflicting.append(ticker)
        if conflicting:
            score += 15
            signals.append(f"Conflicting signals on {', '.join(conflicting)}")

        # Cap at 100
        score = min(score, 100)

        should_escalate = score >= self._threshold

        log.info(
            "complexity_evaluated",
            score=score,
            threshold=self._threshold,
            should_escalate=should_escalate,
            signals=signals,
        )

        return ComplexityResult(
            score=score,
            should_escalate=should_escalate,
            signals=signals,
        )
