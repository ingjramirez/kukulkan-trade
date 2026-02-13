"""Evaluate quality of agent decisions using forward returns.

Analyzes whether proposed trades moved favorably 1/3/5 days after
the decision, regardless of whether the trade was executed.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import date, timedelta

import structlog
import yfinance as yf

from src.storage.database import Database

log = structlog.get_logger()


@dataclass(frozen=True)
class DecisionQuality:
    """Forward-return quality metrics for a single decision."""

    date: date
    ticker: str
    side: str
    fwd_1d: float | None
    fwd_3d: float | None
    fwd_5d: float | None
    favorable_1d: bool
    favorable_3d: bool
    favorable_5d: bool


@dataclass(frozen=True)
class DecisionQualitySummary:
    """Aggregate decision quality metrics."""

    total_decisions: int
    favorable_1d_pct: float
    favorable_3d_pct: float
    favorable_5d_pct: float


class DecisionQualityTracker:
    """Tracks decision quality using forward returns."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def analyze_recent(
        self,
        days: int = 30,
        tenant_id: str = "default",
    ) -> list[DecisionQuality]:
        """Analyze forward returns for recent Portfolio B decisions.

        Parses proposed_trades from agent_decisions, fetches forward
        prices via yfinance batch, and computes 1d/3d/5d returns.

        Args:
            days: Lookback window for decisions.
            tenant_id: Tenant UUID.

        Returns:
            List of DecisionQuality, one per proposed trade.
        """
        decisions = await self._db.get_agent_decisions(limit=50, tenant_id=tenant_id)
        cutoff = date.today() - timedelta(days=days)
        recent = [d for d in decisions if d.date >= cutoff]

        if not recent:
            return []

        # Parse all proposed trades
        proposals: list[tuple[date, str, str]] = []  # (date, ticker, side)
        for d in recent:
            if not d.proposed_trades:
                continue
            try:
                trades_list = json.loads(d.proposed_trades)
            except (json.JSONDecodeError, TypeError):
                continue
            for t in trades_list:
                if isinstance(t, dict):
                    ticker = t.get("ticker", "")
                    side = t.get("side", "BUY")
                    if ticker:
                        proposals.append((d.date, ticker, side))

        if not proposals:
            return []

        # Collect unique tickers
        tickers = list({p[1] for p in proposals})

        # Fetch historical data covering the full period + 10 days forward
        start_date = min(p[0] for p in proposals) - timedelta(days=2)
        end_date = date.today() + timedelta(days=1)

        prices = await self._fetch_prices(tickers, start_date, end_date)
        if not prices:
            return []

        qualities: list[DecisionQuality] = []
        for decision_date, ticker, side in proposals:
            fwd_1d = self._get_forward_return(prices, ticker, decision_date, 1)
            fwd_3d = self._get_forward_return(prices, ticker, decision_date, 3)
            fwd_5d = self._get_forward_return(prices, ticker, decision_date, 5)

            # Favorable = move in the direction of the trade
            is_buy = side.upper() == "BUY"
            favorable_1d = (fwd_1d > 0) if (is_buy and fwd_1d is not None) else (fwd_1d is not None and fwd_1d < 0)
            favorable_3d = (fwd_3d > 0) if (is_buy and fwd_3d is not None) else (fwd_3d is not None and fwd_3d < 0)
            favorable_5d = (fwd_5d > 0) if (is_buy and fwd_5d is not None) else (fwd_5d is not None and fwd_5d < 0)

            qualities.append(
                DecisionQuality(
                    date=decision_date,
                    ticker=ticker,
                    side=side,
                    fwd_1d=round(fwd_1d, 2) if fwd_1d is not None else None,
                    fwd_3d=round(fwd_3d, 2) if fwd_3d is not None else None,
                    fwd_5d=round(fwd_5d, 2) if fwd_5d is not None else None,
                    favorable_1d=favorable_1d,
                    favorable_3d=favorable_3d,
                    favorable_5d=favorable_5d,
                )
            )

        return qualities

    @staticmethod
    def summarize(qualities: list[DecisionQuality]) -> DecisionQualitySummary:
        """Compute aggregate decision quality.

        Args:
            qualities: List of DecisionQuality from analyze_recent().

        Returns:
            DecisionQualitySummary with favorable percentages.
        """
        if not qualities:
            return DecisionQualitySummary(
                total_decisions=0,
                favorable_1d_pct=0.0,
                favorable_3d_pct=0.0,
                favorable_5d_pct=0.0,
            )

        has_1d = [q for q in qualities if q.fwd_1d is not None]
        has_3d = [q for q in qualities if q.fwd_3d is not None]
        has_5d = [q for q in qualities if q.fwd_5d is not None]

        fav_1d = sum(1 for q in has_1d if q.favorable_1d) / len(has_1d) * 100 if has_1d else 0.0
        fav_3d = sum(1 for q in has_3d if q.favorable_3d) / len(has_3d) * 100 if has_3d else 0.0
        fav_5d = sum(1 for q in has_5d if q.favorable_5d) / len(has_5d) * 100 if has_5d else 0.0

        return DecisionQualitySummary(
            total_decisions=len(qualities),
            favorable_1d_pct=round(fav_1d, 1),
            favorable_3d_pct=round(fav_3d, 1),
            favorable_5d_pct=round(fav_5d, 1),
        )

    @staticmethod
    def format_for_prompt(summary: DecisionQualitySummary) -> str:
        """Format decision quality for prompt injection.

        Args:
            summary: DecisionQualitySummary.

        Returns:
            Short formatted text.
        """
        if summary.total_decisions == 0:
            return "No decisions to evaluate yet."

        return (
            f"Decision accuracy ({summary.total_decisions} trades): "
            f"1d={summary.favorable_1d_pct:.0f}% favorable, "
            f"3d={summary.favorable_3d_pct:.0f}%, "
            f"5d={summary.favorable_5d_pct:.0f}%"
        )

    @staticmethod
    async def _fetch_prices(
        tickers: list[str],
        start: date,
        end: date,
    ) -> dict[str, dict[date, float]]:
        """Fetch close prices for tickers between dates.

        Returns:
            Dict of ticker → {date → close_price}.
        """
        if not tickers:
            return {}
        try:
            data = await asyncio.to_thread(
                yf.download,
                tickers,
                start=start.isoformat(),
                end=end.isoformat(),
                progress=False,
            )
            if data.empty:
                return {}

            result: dict[str, dict[date, float]] = {}
            for t in tickers:
                try:
                    if len(tickers) == 1:
                        series = data["Close"]
                    else:
                        if t not in data["Close"].columns:
                            continue
                        series = data["Close"][t]
                    prices_for_ticker: dict[date, float] = {}
                    for idx, val in series.dropna().items():
                        prices_for_ticker[idx.date()] = float(val)
                    result[t] = prices_for_ticker
                except (KeyError, AttributeError):
                    continue
            return result
        except Exception as e:
            log.warning("decision_quality_price_fetch_failed", error=str(e))
            return {}

    @staticmethod
    def _get_forward_return(
        prices: dict[str, dict[date, float]],
        ticker: str,
        decision_date: date,
        days_forward: int,
    ) -> float | None:
        """Get forward return N business days after decision.

        Returns:
            Percentage return, or None if data unavailable.
        """
        ticker_prices = prices.get(ticker)
        if not ticker_prices:
            return None

        # Find decision day price (on or closest before)
        base_price = None
        for offset in range(3):
            d = decision_date - timedelta(days=offset)
            if d in ticker_prices:
                base_price = ticker_prices[d]
                break

        if base_price is None or base_price <= 0:
            return None

        # Find forward price (on or closest after target)
        target_date = decision_date + timedelta(days=days_forward)
        forward_price = None
        for offset in range(5):
            d = target_date + timedelta(days=offset)
            if d in ticker_prices:
                forward_price = ticker_prices[d]
                break

        if forward_price is None:
            return None

        return ((forward_price - base_price) / base_price) * 100
