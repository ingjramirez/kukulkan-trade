"""Daily pipeline orchestrator.

Connects all components: data fetching → strategy execution → trade generation
→ execution → snapshots. Single entry point: run_daily().
"""

import uuid
from datetime import date

import pandas as pd
import structlog

from config.strategies import PORTFOLIO_B
from config.universe import FULL_UNIVERSE, PORTFOLIO_A_UNIVERSE, PORTFOLIO_B_UNIVERSE, get_dynamic_universe
from src.agent.claude_agent import ClaudeAgent
from src.agent.complexity_detector import ComplexityDetector
from src.agent.ticker_discovery import TickerDiscovery
from src.analysis.technical import compute_all_indicators
from src.data.market_data import MarketDataFetcher
from src.data.macro_data import MacroDataFetcher
from src.data.news_fetcher import NewsFetcher
from src.execution.paper_trader import PaperTrader
from src.notifications.telegram_bot import TelegramNotifier
from src.storage.database import Database
from src.strategies.portfolio_a import MomentumStrategy
from src.strategies.portfolio_b import AIAutonomyStrategy

log = structlog.get_logger()


class Orchestrator:
    """Runs the complete daily trading pipeline."""

    def __init__(
        self,
        db: Database,
        notifier: TelegramNotifier | None = None,
        executor=None,
    ) -> None:
        self._db = db
        # Always use yfinance for data (fast, no pacing limits).
        self._market_data = MarketDataFetcher(db)
        self._macro_data = MacroDataFetcher(db)

        # Executor: provided externally, or default to PaperTrader
        self._executor = executor or PaperTrader(db)

        self._strategy_a = MomentumStrategy()
        self._strategy_b = AIAutonomyStrategy()
        self._notifier = notifier or TelegramNotifier()
        self._news_fetcher = NewsFetcher()
        self._complexity_detector = ComplexityDetector()
        self._ticker_discovery = TickerDiscovery(db)

    async def run_daily(self, today: date | None = None) -> dict:
        """Execute the full daily pipeline.

        Steps:
        1. Initialize portfolios (if first run)
        2. Fetch market data (OHLCV for full universe)
        3. Fetch macro data (yield curve, VIX)
        4. Run Portfolio A (momentum)
        5. Fetch news for AI context
        6. Run Portfolio B (AI agent)
        7. Execute all trades
        8. Take end-of-day snapshots
        9. Send Telegram notifications

        Args:
            today: Override date for testing. Defaults to date.today().

        Returns:
            Summary dict with results from each step.
        """
        today = today or date.today()
        summary: dict = {"date": today.isoformat(), "trades": {}, "errors": []}

        log.info("daily_pipeline_start", date=str(today))

        # Step 1: Initialize portfolios
        await self._executor.initialize_portfolios()

        # Step 1.5: Expire old dynamic tickers + build universe
        await self._ticker_discovery.expire_old(today)
        dynamic_universe = await get_dynamic_universe(self._db)

        # Step 2: Fetch market data
        log.info("step_2_fetching_market_data")
        try:
            data = await self._market_data.fetch_universe(
                tickers=dynamic_universe, period="1y"
            )
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

        # Step 5: Fetch news for AI context
        news_context = ""
        log.info("step_5_fetching_news")
        try:
            tickers_for_news = list(closes.columns)[:20]  # Top 20 tickers
            articles = self._news_fetcher.fetch_news(tickers_for_news, max_per_ticker=3)
            if articles:
                rows = self._news_fetcher.store_articles(articles)
                async with self._db.session() as s:
                    s.add_all(rows)
                    await s.commit()
                summary["news_articles"] = len(articles)
            news_context = self._news_fetcher.get_news_context(tickers_for_news)
        except Exception as e:
            log.warning("news_fetch_failed", error=str(e))
            summary["errors"].append(f"News fetch failed: {e}")

        # Step 6: Portfolio B — AI Agent
        log.info("step_6_portfolio_b")
        try:
            trades_b = await self._run_portfolio_b(
                closes, volumes, yield_curve, vix, today,
                news_context=news_context,
            )
            summary["trades"]["B"] = len(trades_b)
        except Exception as e:
            log.error("portfolio_b_failed", error=str(e))
            summary["errors"].append(f"Portfolio B failed: {e}")
            trades_b = []

        # Step 7: Execute trades
        log.info("step_7_executing_trades")
        all_trades = trades_a + trades_b
        if all_trades:
            executed = await self._executor.execute_trades(all_trades)
            summary["trades_executed"] = len(executed)
        else:
            summary["trades_executed"] = 0

        # Step 8: Take snapshots
        log.info("step_8_taking_snapshots")
        latest_prices = {t: float(closes[t].iloc[-1]) for t in closes.columns if not pd.isna(closes[t].iloc[-1])}
        for portfolio_name in ("A", "B"):
            try:
                await self._executor.take_snapshot(portfolio_name, today, latest_prices)
            except Exception as e:
                log.error("snapshot_failed", portfolio=portfolio_name, error=str(e))

        # Step 9: Send Telegram notifications
        log.info("step_9_sending_notifications")
        await self._send_notifications(
            today=today,
            all_trades=all_trades,
            summary=summary,
        )

        log.info(
            "daily_pipeline_complete",
            date=str(today),
            trades_a=len(trades_a),
            trades_b=len(trades_b),
            executed=summary["trades_executed"],
            errors=len(summary["errors"]),
        )

        return summary

    async def _run_portfolio_a(self, closes: pd.DataFrame, today: date):
        """Run Portfolio A momentum strategy and return trades."""
        portfolio = await self._db.get_portfolio("A")
        positions = await self._db.get_positions("A")
        position_map = {p.ticker: p.shares for p in positions}
        cash = portfolio.cash if portfolio else 33_000.0

        trades = self._strategy_a.generate_trades(closes, position_map, cash)

        # Save momentum rankings
        ranking_rows = self._strategy_a.get_ranking_rows(closes, today)
        if ranking_rows:
            await self._db.save_momentum_rankings(ranking_rows)

        log.info("portfolio_a_complete", trades=len(trades), target=self._strategy_a.get_target_ticker(closes))
        return trades

    async def _run_portfolio_b(self, closes, volumes, yield_curve, vix, today, news_context: str = ""):
        """Run Portfolio B AI strategy with complexity-based model routing."""
        portfolio = await self._db.get_portfolio("B")
        positions = await self._db.get_positions("B")
        position_map = {p.ticker: p.shares for p in positions}
        cash = portfolio.cash if portfolio else 66_000.0
        total_value = portfolio.total_value if portfolio else 66_000.0

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
        recent_trades_raw = await self._db.get_trades("B")
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

        # ── Complexity detection & model routing ─────────────────────────
        snapshots = await self._db.get_snapshots("B")
        peak_value = max(
            (s.total_value for s in snapshots if s.total_value),
            default=total_value,
        )

        # Build indicators for held tickers (for conflict detection)
        held_indicators: dict[str, dict] = {}
        for p in positions:
            if p.ticker in closes.columns and len(closes[p.ticker].dropna()) >= 50:
                try:
                    ind = compute_all_indicators(closes[p.ticker].dropna())
                    latest = ind.iloc[-1]
                    held_indicators[p.ticker] = {
                        "rsi_14": float(latest["rsi_14"]) if pd.notna(latest["rsi_14"]) else None,
                        "macd": float(latest["macd"]) if pd.notna(latest["macd"]) else None,
                    }
                except Exception:
                    pass

        complexity = self._complexity_detector.evaluate(
            closes=closes,
            positions=positions_for_agent,
            total_value=total_value,
            peak_value=peak_value,
            regime_today=None,
            regime_yesterday=None,
            vix=vix,
            indicators=held_indicators,
        )

        model_override: str | None = None
        if complexity.should_escalate and self._notifier_available():
            choice = await self._request_model_approval(complexity)
            if choice == "opus":
                model_override = PORTFOLIO_B.escalation_model
            elif choice == "skip":
                log.info("portfolio_b_skipped_by_user")
                return []
            # "sonnet" or timeout → model_override stays None

        # ── Prepare context and call agent ───────────────────────────────
        context = self._strategy_b.prepare_context(
            closes=closes,
            volumes=volumes,
            positions=positions_for_agent,
            cash=cash,
            total_value=total_value,
            recent_trades=recent_trades,
            regime=None,
            yield_curve=yield_curve,
            vix=vix,
            news_context=news_context,
        )
        context["model_override"] = model_override

        response = self._strategy_b._agent.analyze(**context)

        # Convert to trades (include dynamic tickers as valid)
        dynamic_tickers = await self._ticker_discovery.get_active_tickers()
        trades = self._strategy_b.agent_response_to_trades(
            response=response,
            total_value=total_value,
            current_positions=position_map,
            latest_prices=closes.iloc[-1],
            extra_tickers=dynamic_tickers,
        )

        # Save decision
        await self._strategy_b.save_decision(self._db, today, response, trades)

        # Process suggested tickers from agent response
        await self._process_suggested_tickers(response, today)

        log.info(
            "portfolio_b_complete",
            trades=len(trades),
            reasoning=response.get("reasoning", "")[:100],
            tokens=response.get("_tokens_used", 0),
            model_override=model_override,
            complexity_score=complexity.score,
        )
        return trades

    def _notifier_available(self) -> bool:
        """Check if Telegram notifier is configured."""
        return bool(self._notifier._token and self._notifier._chat_id)

    async def _request_model_approval(self, complexity) -> str:
        """Send approval request via Telegram and wait for response.

        Returns:
            "opus", "sonnet", or "skip".
        """
        request_id = uuid.uuid4().hex[:8]
        msg_id = await self._notifier.send_approval_request(complexity, request_id)
        if msg_id is None:
            return "sonnet"
        return await self._notifier.wait_for_approval(
            request_id, PORTFOLIO_B.approval_timeout_seconds
        )

    async def _process_suggested_tickers(self, response: dict, today: date) -> None:
        """Validate agent-suggested tickers and send for Telegram approval.

        Args:
            response: Agent response dict potentially containing suggested_tickers.
            today: Current date.
        """
        suggestions = response.get("suggested_tickers", [])
        if not suggestions:
            return

        for suggestion in suggestions:
            ticker = suggestion.get("ticker", "").upper().strip()
            rationale = suggestion.get("rationale", "")
            if not ticker:
                continue

            # Propose (validates via yfinance internally)
            row = await self._ticker_discovery.propose_ticker(
                ticker=ticker, rationale=rationale, source="agent", today=today,
            )
            if row is None:
                continue

            # Send Telegram approval if available
            if self._notifier_available():
                choice = await self._request_ticker_approval(row)
                if choice == "approve":
                    await self._db.update_discovered_ticker_status(ticker, "approved")
                    log.info("ticker_approved", ticker=ticker)
                else:
                    await self._db.update_discovered_ticker_status(ticker, "rejected")
                    log.info("ticker_rejected", ticker=ticker)
            else:
                # No Telegram — auto-reject (requires human approval)
                await self._db.update_discovered_ticker_status(ticker, "rejected")
                log.info("ticker_auto_rejected_no_telegram", ticker=ticker)

    async def _request_ticker_approval(self, row) -> str:
        """Send ticker approval request via Telegram and wait for response.

        Returns:
            "approve" or "reject".
        """
        request_id = uuid.uuid4().hex[:8]
        msg_id = await self._notifier.send_ticker_proposal(row, request_id)
        if msg_id is None:
            return "reject"
        return await self._notifier.wait_for_ticker_approval(
            request_id, PORTFOLIO_B.approval_timeout_seconds
        )

    async def _send_notifications(
        self,
        today: date,
        all_trades: list,
        summary: dict,
    ) -> None:
        """Send daily brief and trade confirmation via Telegram."""
        try:
            # Build portfolio summaries from snapshots
            portfolio_summaries = {}
            for name in ("A", "B"):
                portfolio = await self._db.get_portfolio(name)
                snapshots = await self._db.get_snapshots(name)
                today_snap = next(
                    (s for s in snapshots if s.date == today), None
                )
                default_value = 33_000.0 if name == "A" else 66_000.0
                portfolio_summaries[name] = {
                    "total_value": today_snap.total_value if today_snap else (portfolio.total_value if portfolio else default_value),
                    "cash": portfolio.cash if portfolio else default_value,
                    "daily_return_pct": today_snap.daily_return_pct if today_snap else None,
                }

            # Add strategy-specific fields
            portfolio_summaries["A"]["top_ticker"] = "—"
            portfolio_summaries["B"]["reasoning"] = ""

            from src.storage.models import TradeSchema
            trade_schemas = []
            for t in all_trades:
                if isinstance(t, TradeSchema):
                    trade_schemas.append(t)

            await self._notifier.send_daily_brief(
                brief_date=today,
                regime=None,
                portfolio_a=portfolio_summaries["A"],
                portfolio_b=portfolio_summaries["B"],
                proposed_trades=trade_schemas,
                commentary="",
            )

            if trade_schemas:
                await self._notifier.send_trade_confirmation(trade_schemas)

            log.info("notifications_sent")
        except Exception as e:
            log.error("notification_failed", error=str(e))
