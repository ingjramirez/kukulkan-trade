"""Overnight gap risk analyzer.

Evaluates portfolio exposure to overnight gaps from earnings,
volatile sectors, and concentrated positions. Runs before market close
to warn about positions that may gap significantly overnight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.storage.database import Database

log = structlog.get_logger()


@dataclass
class PositionGapRisk:
    ticker: str
    weight_pct: float
    gap_risk_score: float
    reasons: list[str] = field(default_factory=list)
    recommendation: str | None = None


@dataclass
class GapRiskAssessment:
    aggregate_risk_score: float
    rating: str  # LOW | MODERATE | HIGH | EXTREME
    earnings_tonight: list[str] = field(default_factory=list)
    positions: list[PositionGapRisk] = field(default_factory=list)


class GapRiskAnalyzer:
    """Analyzes overnight gap risk for a portfolio."""

    EARNINGS_TONIGHT_MULT = 3.0
    VOLATILE_SECTOR_MULT = 1.5
    CONCENTRATION_MULT = 1.2  # Position > 15% of portfolio
    INVERSE_ETF_MULT = 0.5

    VOLATILE_SECTORS = {"Technology", "Biotechnology", "Cryptocurrency", "Semiconductors"}

    RATING_THRESHOLDS = [
        ("LOW", 0, 5),
        ("MODERATE", 5, 15),
        ("HIGH", 15, 30),
        ("EXTREME", 30, float("inf")),
    ]

    async def analyze(
        self,
        db: Database,
        tenant_id: str,
        earnings_tickers: set[str] | None = None,
    ) -> GapRiskAssessment:
        """Analyze overnight gap risk for all held positions.

        Args:
            db: Database instance.
            tenant_id: Tenant UUID.
            earnings_tickers: Set of tickers with earnings tonight.
                If None, fetches from earnings calendar.

        Returns:
            GapRiskAssessment with per-position risk scores and aggregate rating.
        """
        positions = await db.get_positions("B", tenant_id=tenant_id)
        portfolio = await db.get_portfolio("B", tenant_id=tenant_id)
        portfolio_value = portfolio.total_value if portfolio else 0.0

        # Fetch tonight's earnings if not provided
        if earnings_tickers is None:
            earnings_tickers = await self._get_tonight_earnings(db, [p.ticker for p in positions])

        position_risks: list[PositionGapRisk] = []
        total_risk = 0.0

        for pos in positions:
            mv = pos.market_value
            market_value = (mv or 0.0) if mv else pos.shares * (pos.current_price or pos.avg_price)
            weight = (market_value / portfolio_value * 100) if portfolio_value > 0 else 0
            risk_score = weight
            reasons: list[str] = []

            if pos.ticker in earnings_tickers:
                risk_score *= self.EARNINGS_TONIGHT_MULT
                reasons.append("Earnings tonight")

            sector = self._get_sector(pos.ticker)
            if sector in self.VOLATILE_SECTORS:
                risk_score *= self.VOLATILE_SECTOR_MULT
                reasons.append(f"Volatile sector ({sector})")

            if weight > 15:
                risk_score *= self.CONCENTRATION_MULT
                reasons.append(f"Large position ({weight:.1f}%)")

            try:
                from config.universe import InstrumentType, classify_instrument

                if classify_instrument(pos.ticker) == InstrumentType.INVERSE_ETF:
                    risk_score *= self.INVERSE_ETF_MULT
                    reasons.append("Inverse ETF (hedge)")
            except Exception:
                pass

            recommendation = None
            if risk_score > 20:
                recommendation = "Consider reducing before close"
            elif risk_score > 10 and "Earnings tonight" in reasons:
                recommendation = "Earnings risk — review position size"

            position_risks.append(
                PositionGapRisk(
                    ticker=pos.ticker,
                    weight_pct=round(weight, 1),
                    gap_risk_score=round(risk_score, 1),
                    reasons=reasons,
                    recommendation=recommendation,
                )
            )
            total_risk += risk_score

        rating = "LOW"
        for r, low, high in self.RATING_THRESHOLDS:
            if low <= total_risk < high:
                rating = r
                break

        position_risks.sort(key=lambda p: p.gap_risk_score, reverse=True)

        held_earnings = list(earnings_tickers & {p.ticker for p in positions})

        return GapRiskAssessment(
            aggregate_risk_score=round(total_risk, 1),
            rating=rating,
            earnings_tonight=held_earnings,
            positions=position_risks,
        )

    async def _get_tonight_earnings(self, db: Database, tickers: list[str]) -> set[str]:
        """Fetch tickers with earnings today/tonight."""
        try:
            from src.data.earnings_calendar import EarningsCalendar

            cal = EarningsCalendar()
            upcoming = await cal.get_upcoming(db, tickers, days_ahead=1)
            return {row.ticker for row in upcoming}
        except Exception as e:
            log.warning("earnings_fetch_for_gap_risk_failed", error=str(e))
            return set()

    def _get_sector(self, ticker: str) -> str:
        """Get sector for a ticker from the universe map."""
        try:
            from config.universe import SECTOR_MAP

            return SECTOR_MAP.get(ticker, "")
        except Exception:
            return ""
