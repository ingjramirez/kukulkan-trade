"""Portfolio B: AI Full Autonomy strategy.

Claude (Sonnet 4.5) has full discretion over the entire ticker universe.
Receives market data, news context, and portfolio state to make trade decisions.
"""

import json
from datetime import date

import pandas as pd
import structlog

from config.strategies import PORTFOLIO_B
from config.universe import PORTFOLIO_B_UNIVERSE
from src.agent.claude_agent import ClaudeAgent
from src.analysis.technical import compute_all_indicators, compute_rsi
from src.storage.database import Database
from src.storage.models import (
    AgentDecisionRow,
    OrderSide,
    PortfolioName,
    TradeSchema,
)

log = structlog.get_logger()


def filter_interesting_tickers(
    closes: pd.DataFrame,
    current_positions: list[str],
    top_movers: int = 15,
    universe: list[str] | None = None,
) -> list[str]:
    """Select the most actionable tickers to send to Claude.

    Includes:
    - All tickers currently held in Portfolio B
    - Top N movers by absolute 1-day % change
    - Any ticker with RSI < 30 (oversold) or RSI > 70 (overbought)

    Args:
        closes: DataFrame of close prices.
        current_positions: Tickers currently held.
        top_movers: Number of top movers to include.
        universe: Custom ticker universe (defaults to PORTFOLIO_B_UNIVERSE).

    Returns:
        Deduplicated list of ~15-25 interesting tickers.
    """
    base_universe = universe if universe is not None else PORTFOLIO_B_UNIVERSE
    tickers = [t for t in base_universe if t in closes.columns]
    if len(closes) < 2:
        return tickers

    interesting: set[str] = set()

    # Always include current holdings
    interesting.update(current_positions)

    # Top movers by absolute 1-day % change
    today = closes[tickers].iloc[-1]
    yesterday = closes[tickers].iloc[-2]
    pct_change = ((today - yesterday) / yesterday).abs().dropna()
    top = pct_change.sort_values(ascending=False).head(top_movers).index.tolist()
    interesting.update(top)

    # RSI extremes
    for t in tickers:
        series = closes[t].dropna()
        if len(series) < 20:
            continue
        try:
            rsi = compute_rsi(series)
            if not rsi.empty and pd.notna(rsi.iloc[-1]):
                val = rsi.iloc[-1]
                if val < 30 or val > 70:
                    interesting.add(t)
        except (ValueError, KeyError, IndexError) as e:
            log.debug("rsi_filter_failed", ticker=t, error=str(e))
            continue

    # Filter to valid tickers only
    result = [t for t in tickers if t in interesting]
    log.info("interesting_tickers_filtered", total=len(result), held=len(current_positions))
    return result


def build_universe_opportunities(
    closes: pd.DataFrame,
    current_positions: list[str],
    universe: list[str] | None = None,
    top_n: int = 10,
) -> dict:
    """Surface non-held tickers with strong technicals for the agent.

    Breaks tunnel vision by showing what's happening outside the current
    portfolio and the filter_interesting_tickers funnel.

    Args:
        closes: Full price DataFrame for the universe.
        current_positions: Tickers currently held in Portfolio B.
        universe: Custom ticker universe (defaults to PORTFOLIO_B_UNIVERSE).
        top_n: Max tickers to return per category.

    Returns:
        Dict with top_momentum, oversold, and sector_gaps lists.
    """
    from config.universe import SECTOR_MAP

    base_universe = universe if universe is not None else PORTFOLIO_B_UNIVERSE
    held = set(current_positions)
    non_held = [t for t in base_universe if t in closes.columns and t not in held]

    result: dict = {"top_momentum": [], "oversold": [], "sector_gaps": []}

    if len(closes) < 22 or not non_held:
        return result

    # Top momentum: 20-day return for non-held tickers
    try:
        price_now = closes[non_held].iloc[-1]
        price_20d = closes[non_held].iloc[-21]
        returns_20d = ((price_now - price_20d) / price_20d).dropna().sort_values(ascending=False)
        for ticker in returns_20d.head(top_n).index:
            result["top_momentum"].append(
                {"ticker": ticker, "return_20d_pct": round(float(returns_20d[ticker]) * 100, 1)}
            )
    except (IndexError, KeyError):
        pass

    # Oversold: RSI < 35 among non-held
    for t in non_held:
        series = closes[t].dropna()
        if len(series) < 20:
            continue
        try:
            rsi = compute_rsi(series)
            if not rsi.empty and pd.notna(rsi.iloc[-1]):
                val = float(rsi.iloc[-1])
                if val < 35:
                    result["oversold"].append({"ticker": t, "rsi": round(val, 1)})
        except (ValueError, KeyError, IndexError):
            continue
    result["oversold"] = sorted(result["oversold"], key=lambda x: x["rsi"])[:top_n]

    # Sector gaps: sectors with no holdings
    held_sectors = {SECTOR_MAP.get(t, "Unknown") for t in held}
    all_sectors = {SECTOR_MAP.get(t, "Unknown") for t in base_universe} - {"Unknown", "Inverse"}
    missing_sectors = all_sectors - held_sectors
    if missing_sectors:
        result["sector_gaps"] = sorted(missing_sectors)[:5]

    return result


class AIAutonomyStrategy:
    """Portfolio B strategy: Claude decides everything."""

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
        news_context: str = "",
        system_prompt: str | None = None,
        universe: list[str] | None = None,
    ) -> dict:
        """Prepare all data needed for the Claude agent call.

        Uses compact format: filters to interesting tickers and builds
        CSV-style price/indicator summaries for ~45% token savings.

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
            news_context: News headlines string.
            universe: Custom ticker universe for filtering.

        Returns:
            Dict of kwargs ready to pass to agent.analyze().
        """
        # Filter to interesting tickers for compact format
        held_tickers = [p["ticker"] for p in positions if p.get("ticker")]
        interesting = filter_interesting_tickers(
            closes,
            held_tickers,
            universe=universe,
        )

        # Build price dict (last 5 days) — kept for backward compat
        prices: dict[str, list[float]] = {}
        for t in interesting:
            if t in closes.columns:
                vals = closes[t].dropna().tail(5).tolist()
                if len(vals) >= 5:
                    prices[t] = vals

        # Build indicators dict (latest values) — kept for backward compat
        indicators: dict[str, dict] = {}
        for t in interesting:
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
                except (ValueError, KeyError, IndexError) as e:
                    log.debug("technical_context_failed", ticker=t, error=str(e))

        ctx = {
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
            "news_context": news_context,
            "interesting_tickers": interesting,
            "closes_df": closes,
        }
        if system_prompt is not None:
            ctx["system_prompt"] = system_prompt
        return ctx

    def agent_response_to_trades(
        self,
        response: dict,
        total_value: float,
        current_positions: dict[str, float],
        latest_prices: pd.Series,
        extra_tickers: list[str] | None = None,
        universe: list[str] | None = None,
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
            extra_tickers: Additional tickers to accept (e.g. discovered tickers).
            universe: Custom ticker universe (defaults to PORTFOLIO_B_UNIVERSE).

        Returns:
            List of validated TradeSchema objects.
        """
        raw_trades = response.get("trades", [])
        if not raw_trades:
            log.info("agent_no_trades_proposed")
            return []

        trades: list[TradeSchema] = []
        base_universe = universe if universe is not None else PORTFOLIO_B_UNIVERSE
        valid_tickers = set(base_universe)
        if extra_tickers:
            valid_tickers.update(extra_tickers)

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

            # Enforce max weight, then apply conviction scaling
            weight = min(weight, PORTFOLIO_B.max_single_position_pct)

            conviction_multipliers = {"high": 1.0, "medium": 0.7, "low": 0.4}
            conviction = t.get("conviction", "high").lower()
            weight *= conviction_multipliers.get(conviction, 1.0)

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
                    trades.append(
                        TradeSchema(
                            portfolio=PortfolioName.B,
                            ticker=ticker,
                            side=OrderSide.BUY,
                            shares=float(shares_to_buy),
                            price=float(price),
                            reason=f"AI: {reason}",
                        )
                    )
            elif side == "SELL":
                if weight == 0:
                    # Full exit
                    if current_shares > 0:
                        trades.append(
                            TradeSchema(
                                portfolio=PortfolioName.B,
                                ticker=ticker,
                                side=OrderSide.SELL,
                                shares=float(current_shares),
                                price=float(price),
                                reason=f"AI exit: {reason}",
                            )
                        )
                else:
                    # Trim to target weight
                    target_shares = int(target_value / price)
                    shares_to_sell = current_shares - target_shares
                    if shares_to_sell > 0:
                        trades.append(
                            TradeSchema(
                                portfolio=PortfolioName.B,
                                ticker=ticker,
                                side=OrderSide.SELL,
                                shares=float(shares_to_sell),
                                price=float(price),
                                reason=f"AI trim: {reason}",
                            )
                        )

        # Enforce max positions: if buys would exceed 10, drop the smallest
        current_count = len([s for s in current_positions.values() if s > 0])
        new_buys = [t for t in trades if t.side == OrderSide.BUY and t.ticker not in current_positions]
        sells = [t for t in trades if t.side == OrderSide.SELL]
        exits = [s for s in sells if current_positions.get(s.ticker, 0) == s.shares]

        projected_count = current_count + len(new_buys) - len(exits)
        if projected_count > PORTFOLIO_B.max_positions:
            excess = projected_count - PORTFOLIO_B.max_positions
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
        tenant_id: str = "default",
        regime: str | None = None,
        session_label: str | None = None,
    ) -> None:
        """Persist the agent's decision to the database.

        Args:
            db: Database instance.
            analysis_date: Date of the decision.
            response: Full agent response dict.
            trades: Validated trades list.
            tenant_id: Tenant UUID.
            regime: Market regime at time of decision.
            session_label: Session label (Morning/Midday/Closing).
        """
        trades_json = json.dumps(
            [{"ticker": t.ticker, "side": t.side.value, "shares": t.shares, "price": t.price} for t in trades]
        )

        async with db.session() as s:
            s.add(
                AgentDecisionRow(
                    tenant_id=tenant_id,
                    date=analysis_date,
                    prompt_summary=f"Portfolio B analysis for {analysis_date}",
                    response_summary=response.get("regime_assessment", ""),
                    proposed_trades=trades_json,
                    reasoning=response.get("reasoning", ""),
                    model_used=response.get("_model", ""),
                    tokens_used=response.get("_tokens_used", 0),
                    regime=regime,
                    session_label=session_label,
                )
            )
            await s.commit()

        log.info(
            "agent_decision_saved",
            date=str(analysis_date),
            trades=len(trades),
            tokens=response.get("_tokens_used", 0),
        )
