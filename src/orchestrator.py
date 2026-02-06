"""Daily pipeline orchestrator.

Connects all components: data fetching → strategy execution → trade generation
→ paper trading → snapshots. Single entry point: run_daily().
"""

from datetime import date

import pandas as pd
import structlog

from config.universe import FULL_UNIVERSE, PORTFOLIO_A_UNIVERSE, PORTFOLIO_B_UNIVERSE
from src.agent.claude_agent import ClaudeAgent
from src.analysis.technical import compute_all_indicators
from src.data.market_data import MarketDataFetcher
from src.data.macro_data import MacroDataFetcher
from src.execution.paper_trader import PaperTrader
from src.notifications.telegram_bot import TelegramNotifier
from src.storage.database import Database
from src.strategies.portfolio_a import MomentumStrategy
from src.strategies.portfolio_b import SectorRotationStrategy
from src.strategies.portfolio_c import AIAutonomyStrategy

log = structlog.get_logger()


class Orchestrator:
    """Runs the complete daily trading pipeline."""

    def __init__(self, db: Database, notifier: TelegramNotifier | None = None) -> None:
        self._db = db
        self._market_data = MarketDataFetcher(db)
        self._macro_data = MacroDataFetcher(db)
        self._paper_trader = PaperTrader(db)
        self._strategy_a = MomentumStrategy()
        self._strategy_b = SectorRotationStrategy()
        self._strategy_c = AIAutonomyStrategy()
        self._notifier = notifier or TelegramNotifier()

    async def run_daily(self, today: date | None = None) -> dict:
        """Execute the full daily pipeline.

        Steps:
        1. Initialize portfolios (if first run)
        2. Fetch market data (OHLCV for full universe)
        3. Fetch macro data (yield curve, VIX)
        4. Run Portfolio A (momentum)
        5. Run Portfolio B (composite scoring + regime)
        6. Run Portfolio C (AI agent)
        7. Execute all trades via paper trader
        8. Take end-of-day snapshots

        Args:
            today: Override date for testing. Defaults to date.today().

        Returns:
            Summary dict with results from each step.
        """
        today = today or date.today()
        summary: dict = {"date": today.isoformat(), "trades": {}, "errors": []}

        log.info("daily_pipeline_start", date=str(today))

        # Step 1: Initialize portfolios
        await self._paper_trader.initialize_portfolios()

        # Step 2: Fetch market data
        log.info("step_2_fetching_market_data")
        try:
            data = await self._market_data.fetch_universe(period="1y")
        except Exception as e:
            log.error("market_data_fetch_failed", error=str(e))
            summary["errors"].append(f"Market data fetch failed: {e}")
            return summary

        if not data:
            log.error("no_market_data")
            summary["errors"].append("No market data returned")
            return summary

        # Build closes and volumes DataFrames
        closes = pd.DataFrame({t: df["Close"] for t, df in data.items()})
        volumes = pd.DataFrame({t: df["Volume"] for t, df in data.items()})
        closes = closes.sort_index()
        volumes = volumes.sort_index()

        summary["tickers_fetched"] = len(data)
        log.info("market_data_ready", tickers=len(data), rows=len(closes))

        # Step 3: Fetch macro data
        log.info("step_3_fetching_macro_data")
        yield_curve = None
        vix = None
        try:
            yield_curve = self._macro_data.get_latest_yield_curve()
            vix = self._macro_data.get_latest_vix()
            summary["macro"] = {"yield_curve": yield_curve, "vix": vix}
        except Exception as e:
            log.warning("macro_data_fetch_failed", error=str(e))
            summary["errors"].append(f"Macro data fetch failed: {e}")

        # Step 4: Portfolio A — Momentum
        log.info("step_4_portfolio_a")
        try:
            trades_a = await self._run_portfolio_a(closes, today)
            summary["trades"]["A"] = len(trades_a)
        except Exception as e:
            log.error("portfolio_a_failed", error=str(e))
            summary["errors"].append(f"Portfolio A failed: {e}")
            trades_a = []

        # Step 5: Portfolio B — Sector Rotation
        log.info("step_5_portfolio_b")
        regime = None
        try:
            trades_b, regime = await self._run_portfolio_b(
                closes, volumes, yield_curve, vix, today
            )
            summary["trades"]["B"] = len(trades_b)
            summary["regime"] = regime.value if regime else None
        except Exception as e:
            log.error("portfolio_b_failed", error=str(e))
            summary["errors"].append(f"Portfolio B failed: {e}")
            trades_b = []

        # Step 6: Portfolio C — AI Agent
        log.info("step_6_portfolio_c")
        try:
            trades_c = await self._run_portfolio_c(
                closes, volumes, regime, yield_curve, vix, today
            )
            summary["trades"]["C"] = len(trades_c)
        except Exception as e:
            log.error("portfolio_c_failed", error=str(e))
            summary["errors"].append(f"Portfolio C failed: {e}")
            trades_c = []

        # Step 7: Execute trades
        log.info("step_7_executing_trades")
        all_trades = trades_a + trades_b + trades_c
        if all_trades:
            executed = await self._paper_trader.execute_trades(all_trades)
            summary["trades_executed"] = len(executed)
        else:
            summary["trades_executed"] = 0

        # Step 8: Take snapshots
        log.info("step_8_taking_snapshots")
        latest_prices = {t: float(closes[t].iloc[-1]) for t in closes.columns if not pd.isna(closes[t].iloc[-1])}
        for portfolio_name in ("A", "B", "C"):
            try:
                await self._paper_trader.take_snapshot(portfolio_name, today, latest_prices)
            except Exception as e:
                log.error("snapshot_failed", portfolio=portfolio_name, error=str(e))

        # Step 9: Send Telegram notifications
        log.info("step_9_sending_notifications")
        await self._send_notifications(
            today=today,
            regime=regime,
            all_trades=all_trades,
            summary=summary,
        )

        log.info(
            "daily_pipeline_complete",
            date=str(today),
            trades_a=len(trades_a),
            trades_b=len(trades_b),
            trades_c=len(trades_c),
            executed=summary["trades_executed"],
            errors=len(summary["errors"]),
        )

        return summary

    async def _run_portfolio_a(self, closes: pd.DataFrame, today: date):
        """Run Portfolio A momentum strategy and return trades."""
        portfolio = await self._db.get_portfolio("A")
        positions = await self._db.get_positions("A")
        position_map = {p.ticker: p.shares for p in positions}
        cash = portfolio.cash if portfolio else 33_333.0

        trades = self._strategy_a.generate_trades(closes, position_map, cash)

        # Save momentum rankings
        ranking_rows = self._strategy_a.get_ranking_rows(closes, today)
        if ranking_rows:
            await self._db.save_momentum_rankings(ranking_rows)

        log.info("portfolio_a_complete", trades=len(trades), target=self._strategy_a.get_target_ticker(closes))
        return trades

    async def _run_portfolio_b(self, closes, volumes, yield_curve, vix, today):
        """Run Portfolio B sector rotation strategy and return trades + regime."""
        portfolio = await self._db.get_portfolio("B")
        positions = await self._db.get_positions("B")
        position_map = {p.ticker: p.shares for p in positions}
        cash = portfolio.cash if portfolio else 33_333.0

        # Get SPY and BTC data for regime detection
        spy_closes = closes["XLK"] if "XLK" not in closes.columns else None
        # Use a broad market proxy — if SPY isn't in universe, approximate with XLK
        for proxy in ["SPY", "QQQ", "XLK"]:
            if proxy in closes.columns:
                spy_closes = closes[proxy]
                break

        btc_closes = closes.get("IBIT")

        scores, regime = self._strategy_b.analyze(
            closes=closes,
            volumes=volumes,
            spy_closes=spy_closes,
            btc_closes=btc_closes,
            yield_curve=yield_curve,
            vix=vix,
        )

        selected = self._strategy_b.select_positions(scores, regime)
        trades = self._strategy_b.generate_trades(
            selected_tickers=selected,
            current_positions=position_map,
            cash=cash,
            latest_prices=closes.iloc[-1],
        )

        # Save composite scores
        if not scores.empty:
            rows = self._strategy_b.scores_to_db_rows(scores, today, regime)
            async with self._db.session() as s:
                s.add_all(rows)
                await s.commit()

        log.info("portfolio_b_complete", trades=len(trades), regime=regime.value, selected=selected)
        return trades, regime

    async def _run_portfolio_c(self, closes, volumes, regime, yield_curve, vix, today):
        """Run Portfolio C AI strategy and return trades."""
        portfolio = await self._db.get_portfolio("C")
        positions = await self._db.get_positions("C")
        position_map = {p.ticker: p.shares for p in positions}
        cash = portfolio.cash if portfolio else 33_333.0
        total_value = portfolio.total_value if portfolio else 33_333.0

        # Build positions list for agent context
        positions_for_agent = [
            {
                "ticker": p.ticker,
                "shares": p.shares,
                "avg_price": p.avg_price,
                "market_value": p.shares * float(closes[p.ticker].iloc[-1])
                if p.ticker in closes.columns else 0,
            }
            for p in positions
        ]

        # Get recent trades for context
        recent_trades_raw = await self._db.get_trades("C")
        recent_trades = [
            {
                "ticker": t.ticker,
                "side": t.side,
                "shares": t.shares,
                "price": t.price,
                "reason": t.reason or "",
            }
            for t in recent_trades_raw[:5]
        ]

        # Prepare context and call agent
        context = self._strategy_c.prepare_context(
            closes=closes,
            volumes=volumes,
            positions=positions_for_agent,
            cash=cash,
            total_value=total_value,
            recent_trades=recent_trades,
            regime=regime.value if regime else None,
            yield_curve=yield_curve,
            vix=vix,
        )

        response = self._strategy_c._agent.analyze(**context)

        # Convert to trades
        trades = self._strategy_c.agent_response_to_trades(
            response=response,
            total_value=total_value,
            current_positions=position_map,
            latest_prices=closes.iloc[-1],
        )

        # Save decision
        await self._strategy_c.save_decision(self._db, today, response, trades)

        log.info(
            "portfolio_c_complete",
            trades=len(trades),
            reasoning=response.get("reasoning", "")[:100],
            tokens=response.get("_tokens_used", 0),
        )
        return trades

    async def _send_notifications(
        self,
        today: date,
        regime,
        all_trades: list,
        summary: dict,
    ) -> None:
        """Send daily brief and trade confirmation via Telegram."""
        try:
            # Build portfolio summaries from snapshots
            portfolio_summaries = {}
            for name in ("A", "B", "C"):
                portfolio = await self._db.get_portfolio(name)
                snapshots = await self._db.get_snapshots(name)
                today_snap = next(
                    (s for s in snapshots if s.date == today), None
                )
                portfolio_summaries[name] = {
                    "total_value": today_snap.total_value if today_snap else (portfolio.total_value if portfolio else 33_333.0),
                    "cash": portfolio.cash if portfolio else 33_333.0,
                    "daily_return_pct": today_snap.daily_return_pct if today_snap else None,
                }

            # Add strategy-specific fields
            portfolio_summaries["A"]["top_ticker"] = "—"
            portfolio_summaries["B"]["selected"] = []
            portfolio_summaries["C"]["reasoning"] = ""

            # Get AI commentary (if Portfolio C ran)
            commentary = ""

            from src.storage.models import TradeSchema, PortfolioName, OrderSide
            trade_schemas = []
            for t in all_trades:
                if isinstance(t, TradeSchema):
                    trade_schemas.append(t)

            await self._notifier.send_daily_brief(
                brief_date=today,
                regime=regime.value if regime else None,
                portfolio_a=portfolio_summaries["A"],
                portfolio_b=portfolio_summaries["B"],
                portfolio_c=portfolio_summaries["C"],
                proposed_trades=trade_schemas,
                commentary=commentary,
            )

            if trade_schemas:
                await self._notifier.send_trade_confirmation(trade_schemas)

            log.info("notifications_sent")
        except Exception as e:
            log.error("notification_failed", error=str(e))
