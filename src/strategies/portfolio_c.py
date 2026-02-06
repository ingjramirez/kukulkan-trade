"""Portfolio C: AI Full Autonomy strategy.

Claude (Sonnet 4.5) has full discretion over the entire ticker universe.
Receives market data, news context, and portfolio state to make trade decisions.
"""

import json
from datetime import date

import pandas as pd
import structlog

from config.strategies import PORTFOLIO_C
from config.universe import PORTFOLIO_C_UNIVERSE
from src.agent.claude_agent import ClaudeAgent
from src.analysis.technical import compute_all_indicators
from src.storage.database import Database
from src.storage.models import (
    AgentDecisionRow,
    OrderSide,
    PortfolioName,
    TradeSchema,
)

log = structlog.get_logger()


class AIAutonomyStrategy:
    """Portfolio C strategy: Claude decides everything."""

    def __init__(self, agent: ClaudeAgent | None = None) -> None:
        self._agent = agent or ClaudeAgent()

    def prepare_context(
        self,
        closes: pd.DataFrame,
        volumes: pd.DataFrame,
        positions: list[dict],
        cash: float,
        total_value: float,
        recent_trades: list[dict],
        regime: str | None = None,
        yield_curve: float | None = None,
        vix: float | None = None,
    ) -> dict:
        """Prepare all data needed for the Claude agent call.

        Args:
            closes: Price DataFrame for the universe.
            volumes: Volume DataFrame.
            positions: Current positions as list of dicts.
            cash: Available cash.
            total_value: Total portfolio value.
            recent_trades: Recent trades history.
            regime: Current regime string.
            yield_curve: 10Y-2Y spread.
            vix: Current VIX.

        Returns:
            Dict of kwargs ready to pass to agent.analyze().
        """
        tickers = [t for t in PORTFOLIO_C_UNIVERSE if t in closes.columns]

        # Build price dict (last 5 days)
        prices: dict[str, list[float]] = {}
        for t in tickers:
            if t in closes.columns:
                vals = closes[t].dropna().tail(5).tolist()
                if len(vals) >= 5:
                    prices[t] = vals

        # Build indicators dict (latest values)
        indicators: dict[str, dict] = {}
        for t in tickers:
            if t in closes.columns and len(closes[t].dropna()) >= 50:
                try:
                    ind = compute_all_indicators(closes[t].dropna())
                    latest = ind.iloc[-1]
                    indicators[t] = {
                        "rsi_14": float(latest["rsi_14"]) if pd.notna(latest["rsi_14"]) else None,
                        "macd": float(latest["macd"]) if pd.notna(latest["macd"]) else None,
                        "sma_20": float(latest["sma_20"]) if pd.notna(latest["sma_20"]) else None,
                        "sma_50": float(latest["sma_50"]) if pd.notna(latest["sma_50"]) else None,
                    }
                except Exception:
                    pass

        return {
            "analysis_date": date.today(),
            "cash": cash,
            "total_value": total_value,
            "positions": positions,
            "prices": prices,
            "tickers": list(prices.keys()),
            "indicators": indicators,
            "recent_trades": recent_trades,
            "regime": regime,
            "yield_curve": yield_curve,
            "vix": vix,
        }

    def agent_response_to_trades(
        self,
        response: dict,
        total_value: float,
        current_positions: dict[str, float],
        latest_prices: pd.Series,
    ) -> list[TradeSchema]:
        """Convert Claude's trade proposals into validated TradeSchema objects.

        Applies risk limits:
        - Max 30% weight per position
        - Only valid tickers from the universe
        - Converts weight-based sizing to share counts

        Args:
            response: Parsed response from agent.analyze().
            total_value: Current total portfolio value.
            current_positions: Dict of ticker -> shares held.
            latest_prices: Most recent prices.

        Returns:
            List of validated TradeSchema objects.
        """
        raw_trades = response.get("trades", [])
        if not raw_trades:
            log.info("agent_no_trades_proposed")
            return []

        trades: list[TradeSchema] = []
        valid_tickers = set(PORTFOLIO_C_UNIVERSE)

        for t in raw_trades:
            ticker = t.get("ticker", "")
            side = t.get("side", "").upper()
            weight = t.get("weight", 0)
            reason = t.get("reason", "")

            # Validate
            if ticker not in valid_tickers:
                log.warning("agent_invalid_ticker", ticker=ticker)
                continue
            if side not in ("BUY", "SELL"):
                log.warning("agent_invalid_side", side=side)
                continue

            # Enforce max weight
            weight = min(weight, PORTFOLIO_C.max_single_position_pct)

            price = latest_prices.get(ticker)
            if price is None or pd.isna(price) or price <= 0:
                log.warning("agent_no_price", ticker=ticker)
                continue

            target_value = total_value * weight
            current_shares = current_positions.get(ticker, 0)

            if side == "BUY":
                target_shares = int(target_value / price)
                shares_to_buy = target_shares - current_shares
                if shares_to_buy > 0:
                    trades.append(TradeSchema(
                        portfolio=PortfolioName.C,
                        ticker=ticker,
                        side=OrderSide.BUY,
                        shares=float(shares_to_buy),
                        price=float(price),
                        reason=f"AI: {reason}",
                    ))
            elif side == "SELL":
                if weight == 0:
                    # Full exit
                    if current_shares > 0:
                        trades.append(TradeSchema(
                            portfolio=PortfolioName.C,
                            ticker=ticker,
                            side=OrderSide.SELL,
                            shares=float(current_shares),
                            price=float(price),
                            reason=f"AI exit: {reason}",
                        ))
                else:
                    # Trim to target weight
                    target_shares = int(target_value / price)
                    shares_to_sell = current_shares - target_shares
                    if shares_to_sell > 0:
                        trades.append(TradeSchema(
                            portfolio=PortfolioName.C,
                            ticker=ticker,
                            side=OrderSide.SELL,
                            shares=float(shares_to_sell),
                            price=float(price),
                            reason=f"AI trim: {reason}",
                        ))

        # Enforce max positions: if buys would exceed 10, drop the smallest
        current_count = len([s for s in current_positions.values() if s > 0])
        new_buys = [t for t in trades if t.side == OrderSide.BUY and t.ticker not in current_positions]
        sells = [t for t in trades if t.side == OrderSide.SELL]
        exits = [s for s in sells if current_positions.get(s.ticker, 0) == s.shares]

        projected_count = current_count + len(new_buys) - len(exits)
        if projected_count > PORTFOLIO_C.max_positions:
            excess = projected_count - PORTFOLIO_C.max_positions
            # Drop smallest new buys
            new_buys_sorted = sorted(new_buys, key=lambda t: t.total)
            for drop in new_buys_sorted[:excess]:
                trades.remove(drop)
                log.info("agent_trade_dropped_max_positions", ticker=drop.ticker)

        log.info("agent_trades_validated", count=len(trades))
        return trades

    async def save_decision(
        self,
        db: Database,
        analysis_date: date,
        response: dict,
        trades: list[TradeSchema],
    ) -> None:
        """Persist the agent's decision to the database.

        Args:
            db: Database instance.
            analysis_date: Date of the decision.
            response: Full agent response dict.
            trades: Validated trades list.
        """
        trades_json = json.dumps([
            {"ticker": t.ticker, "side": t.side.value, "shares": t.shares, "price": t.price}
            for t in trades
        ])

        async with db.session() as s:
            s.add(AgentDecisionRow(
                date=analysis_date,
                prompt_summary=f"Portfolio C analysis for {analysis_date}",
                response_summary=response.get("regime_assessment", ""),
                proposed_trades=trades_json,
                reasoning=response.get("reasoning", ""),
                model_used=response.get("_model", ""),
                tokens_used=response.get("_tokens_used", 0),
            ))
            await s.commit()

        log.info(
            "agent_decision_saved",
            date=str(analysis_date),
            trades=len(trades),
            tokens=response.get("_tokens_used", 0),
        )
