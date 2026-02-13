"""Track trade outcomes for Portfolio B feedback loop.

Joins executed trades with agent decisions to compute P&L,
alpha vs sector ETF, and alpha vs SPY for each position.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import date, timedelta

import structlog
import yfinance as yf

from config.universe import SECTOR_ETF_MAP, SECTOR_MAP
from src.storage.database import Database

log = structlog.get_logger()


@dataclass(frozen=True)
class TradeOutcome:
    """Single trade outcome with benchmark comparisons."""

    ticker: str
    side: str
    entry_price: float
    current_price: float
    exit_price: float | None
    pnl_pct: float
    hold_days: int
    sector: str
    sector_etf_pct: float | None
    spy_pct: float | None
    alpha_vs_sector: float | None
    alpha_vs_spy: float | None
    conviction: str
    reasoning: str
    regime_at_entry: str | None = None
    session_at_entry: str | None = None
    verdict: str | None = None


class OutcomeTracker:
    """Computes trade outcomes by joining trades and agent decisions."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def get_recent_outcomes(
        self,
        days: int = 30,
        tenant_id: str = "default",
    ) -> list[TradeOutcome]:
        """Get outcomes for Portfolio B trades in the last N days.

        Joins trades (BUY side) with agent_decisions to extract conviction
        and reasoning. Fetches current prices and benchmarks via yfinance.

        Args:
            days: Lookback window.
            tenant_id: Tenant UUID.

        Returns:
            List of TradeOutcome, oldest first.
        """
        since = date.today() - timedelta(days=days)
        trades = await self._db.get_trades("B", since=since, tenant_id=tenant_id)

        # Only track BUY entries (sells are exits, tracked via position close)
        buy_trades = [t for t in trades if t.side == "BUY"]
        if not buy_trades:
            return []

        # Get current positions to distinguish open vs closed
        positions = await self._db.get_positions("B", tenant_id=tenant_id)
        position_map = {p.ticker: p for p in positions}

        # Get agent decisions for conviction extraction
        decisions = await self._db.get_agent_decisions(limit=50, tenant_id=tenant_id)
        decision_map = self._build_decision_map(decisions)

        # Collect all tickers we need prices for
        tickers_needed: set[str] = set()
        for t in buy_trades:
            tickers_needed.add(t.ticker)
            sector = SECTOR_MAP.get(t.ticker, "")
            etf = SECTOR_ETF_MAP.get(sector)
            if etf:
                tickers_needed.add(etf)
        tickers_needed.add("SPY")

        # Fetch current prices in one batch
        current_prices = await self._fetch_current_prices(list(tickers_needed))

        # Also fetch prices at entry dates for benchmarks
        entry_dates = {t.executed_at.date() for t in buy_trades}
        entry_prices_by_date = await self._fetch_prices_at_dates(
            list(tickers_needed),
            list(entry_dates),
        )

        outcomes: list[TradeOutcome] = []
        for trade in buy_trades:
            ticker = trade.ticker
            entry_price = trade.price
            trade_date = trade.executed_at.date()

            # Current or exit price
            pos = position_map.get(ticker)
            if pos and pos.shares > 0:
                # Still held — use current price
                current = current_prices.get(ticker, entry_price)
                exit_price = None
            else:
                # Closed — find the sell trade price
                sell_trades = [
                    t for t in trades if t.ticker == ticker and t.side == "SELL" and t.executed_at > trade.executed_at
                ]
                if sell_trades:
                    exit_price = sell_trades[0].price
                    current = exit_price
                else:
                    current = current_prices.get(ticker, entry_price)
                    exit_price = None

            pnl_pct = ((current - entry_price) / entry_price) * 100 if entry_price > 0 else 0.0
            hold_days = (date.today() - trade_date).days

            # Sector benchmark
            sector = SECTOR_MAP.get(ticker, "Unknown")
            sector_etf = SECTOR_ETF_MAP.get(sector)
            sector_etf_pct = None
            alpha_vs_sector = None
            if sector_etf:
                etf_entry = entry_prices_by_date.get((sector_etf, trade_date))
                etf_current = current_prices.get(sector_etf)
                if etf_entry and etf_current and etf_entry > 0:
                    sector_etf_pct = ((etf_current - etf_entry) / etf_entry) * 100
                    alpha_vs_sector = pnl_pct - sector_etf_pct

            # SPY benchmark
            spy_entry = entry_prices_by_date.get(("SPY", trade_date))
            spy_current = current_prices.get("SPY")
            spy_pct = None
            alpha_vs_spy = None
            if spy_entry and spy_current and spy_entry > 0:
                spy_pct = ((spy_current - spy_entry) / spy_entry) * 100
                alpha_vs_spy = pnl_pct - spy_pct

            # Extract conviction from decision
            conviction, reasoning, regime_at_entry, session_at_entry = self._extract_conviction(
                decision_map,
                trade_date,
                ticker,
            )

            rounded_alpha_sector = round(alpha_vs_sector, 2) if alpha_vs_sector is not None else None
            verdict = self._compute_verdict(rounded_alpha_sector)

            outcomes.append(
                TradeOutcome(
                    ticker=ticker,
                    side=trade.side,
                    entry_price=entry_price,
                    current_price=current,
                    exit_price=exit_price,
                    pnl_pct=round(pnl_pct, 2),
                    hold_days=hold_days,
                    sector=sector,
                    sector_etf_pct=round(sector_etf_pct, 2) if sector_etf_pct is not None else None,
                    spy_pct=round(spy_pct, 2) if spy_pct is not None else None,
                    alpha_vs_sector=rounded_alpha_sector,
                    alpha_vs_spy=round(alpha_vs_spy, 2) if alpha_vs_spy is not None else None,
                    conviction=conviction,
                    reasoning=reasoning,
                    regime_at_entry=regime_at_entry,
                    session_at_entry=session_at_entry,
                    verdict=verdict,
                )
            )

        # Sort by trade date (oldest first)
        return outcomes

    async def get_open_position_outcomes(
        self,
        tenant_id: str = "default",
    ) -> list[TradeOutcome]:
        """Get outcomes for currently open Portfolio B positions only.

        Args:
            tenant_id: Tenant UUID.

        Returns:
            List of TradeOutcome for open positions.
        """
        outcomes = await self.get_recent_outcomes(days=90, tenant_id=tenant_id)
        return [o for o in outcomes if o.exit_price is None]

    @staticmethod
    def _build_decision_map(
        decisions: list,
    ) -> dict[date, dict]:
        """Build a map of date → {trades: ticker_map, regime, session_label} from decisions.

        Backward compatible: old format (no "trades" key) is auto-detected.
        """
        result: dict[date, dict] = {}
        for d in decisions:
            if not d.proposed_trades:
                continue
            try:
                trades_list = json.loads(d.proposed_trades)
            except (json.JSONDecodeError, TypeError):
                continue
            ticker_map: dict[str, dict] = {}
            for t in trades_list:
                if isinstance(t, dict):
                    ticker_map[t.get("ticker", "")] = t
            result[d.date] = {
                "trades": ticker_map,
                "regime": getattr(d, "regime", None),
                "session_label": getattr(d, "session_label", None),
            }
        return result

    @staticmethod
    def _extract_conviction(
        decision_map: dict[date, dict],
        trade_date: date,
        ticker: str,
    ) -> tuple[str, str, str | None, str | None]:
        """Extract conviction, reasoning, regime, and session for a specific trade.

        Returns:
            Tuple of (conviction, reasoning, regime, session_label).
        """
        day_data = decision_map.get(trade_date, {})
        # Backward compat: if no "trades" key, treat as old ticker_map format
        if "trades" in day_data:
            ticker_map = day_data["trades"]
            regime = day_data.get("regime")
            session_label = day_data.get("session_label")
        else:
            ticker_map = day_data
            regime = None
            session_label = None

        trade_info = ticker_map.get(ticker, {})
        conviction = "medium"
        reasoning = ""
        if isinstance(trade_info, dict):
            reason = trade_info.get("reason", "")
            reasoning = reason
            for level in ("high", "medium", "low"):
                if level in reason.lower():
                    conviction = level
                    break
        return conviction, reasoning, regime, session_label

    @staticmethod
    def _compute_verdict(alpha_vs_sector: float | None) -> str | None:
        """Compute verdict based on alpha vs sector ETF.

        Returns:
            "OUTPERFORMED", "UNDERPERFORMED", "MATCHED", or None.
        """
        if alpha_vs_sector is None:
            return None
        if alpha_vs_sector > 0.5:
            return "OUTPERFORMED"
        if alpha_vs_sector < -0.5:
            return "UNDERPERFORMED"
        return "MATCHED"

    @staticmethod
    async def _fetch_current_prices(tickers: list[str]) -> dict[str, float]:
        """Fetch current prices for a list of tickers via yfinance.

        Returns:
            Dict of ticker → current price.
        """
        if not tickers:
            return {}
        try:
            data = await asyncio.to_thread(
                yf.download,
                tickers,
                period="2d",
                progress=False,
            )
            if data.empty:
                return {}
            prices: dict[str, float] = {}
            if len(tickers) == 1:
                # Single ticker: columns are just OHLCV
                last = data["Close"].iloc[-1]
                if last and not (isinstance(last, float) and last != last):
                    prices[tickers[0]] = float(last)
            else:
                for t in tickers:
                    if t in data["Close"].columns:
                        val = data["Close"][t].dropna()
                        if not val.empty:
                            prices[t] = float(val.iloc[-1])
            return prices
        except Exception as e:
            log.warning("outcome_price_fetch_failed", error=str(e))
            return {}

    @staticmethod
    async def _fetch_prices_at_dates(
        tickers: list[str],
        dates: list[date],
    ) -> dict[tuple[str, date], float]:
        """Fetch historical prices at specific dates.

        Returns:
            Dict of (ticker, date) → close price.
        """
        if not tickers or not dates:
            return {}
        try:
            min_date = min(dates) - timedelta(days=5)
            max_date = max(dates) + timedelta(days=1)
            data = await asyncio.to_thread(
                yf.download,
                tickers,
                start=min_date.isoformat(),
                end=max_date.isoformat(),
                progress=False,
            )
            if data.empty:
                return {}

            result: dict[tuple[str, date], float] = {}
            for d in dates:
                for t in tickers:
                    try:
                        if len(tickers) == 1:
                            series = data["Close"]
                        else:
                            if t not in data["Close"].columns:
                                continue
                            series = data["Close"][t]
                        # Find closest date on or before
                        mask = series.index.date <= d
                        if mask.any():
                            val = series[mask].iloc[-1]
                            if val and not (isinstance(val, float) and val != val):
                                result[(t, d)] = float(val)
                    except (KeyError, IndexError):
                        continue
            return result
        except Exception as e:
            log.warning("outcome_historical_price_fetch_failed", error=str(e))
            return {}
