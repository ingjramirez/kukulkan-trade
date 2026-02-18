"""Daily pipeline orchestrator.

Connects all components: data fetching → strategy execution → trade generation
→ execution → snapshots. Single entry point: run_daily().

Multi-tenant support: run_all_tenants() iterates active tenants,
creating per-tenant Alpaca clients and Telegram bots.
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd
import structlog

from config.risk_rules import TRAIL_PCT
from config.settings import settings
from config.strategies import PORTFOLIO_B
from config.universe import (
    get_dynamic_universe,
)
from src.agent.claude_agent import (
    _build_decision_review,
    _build_track_record,
    build_system_prompt,
)
from src.agent.complexity_detector import ComplexityDetector
from src.agent.memory import AgentMemoryManager
from src.agent.ticker_discovery import TickerDiscovery
from src.analysis.performance import PerformanceTracker
from src.analysis.regime import RegimeClassifier, RegimeResult
from src.analysis.risk_manager import RiskManager
from src.analysis.technical import compute_all_indicators
from src.data.macro_data import MacroDataFetcher
from src.data.market_data import MarketDataFetcher
from src.data.news_aggregator import NewsAggregator
from src.data.news_compactor import NewsCompactor
from src.data.news_fetcher import NewsFetcher
from src.execution.paper_trader import PaperTrader
from src.notifications.telegram_bot import TelegramNotifier
from src.storage.database import Database
from src.storage.models import OrderSide, PortfolioName, TenantRow, TradeSchema
from src.strategies.portfolio_a import MomentumStrategy
from src.strategies.portfolio_b import AIAutonomyStrategy
from src.utils.allocations import (
    DEFAULT_ALLOCATIONS,
    DEPOSIT_THRESHOLD,
    RECONCILE_THRESHOLD,
    TenantAllocations,
    resolve_from_tenant,
)
from src.utils.market_calendar import is_market_open, trading_days_between
from src.utils.tenant_universe import get_tenant_universe

log = structlog.get_logger()

# Delay between tenant sessions is configured via settings.inter_tenant_delay


@dataclass
class MarketContext:
    """Data assembled during market data fetch + regime classification."""

    closes: pd.DataFrame
    volumes: pd.DataFrame
    dynamic_universe: list[str]
    trailing_stop_sells: list = field(default_factory=list)
    trailing_stop_alerts: list[dict] = field(default_factory=list)
    yield_curve: float | None = None
    vix: float | None = None
    regime_result: Any = None
    allocations: TenantAllocations = field(default_factory=lambda: DEFAULT_ALLOCATIONS)
    halted_portfolios: set[str] = field(default_factory=set)


@dataclass
class NewsContext:
    """News and earnings data assembled for the AI agent prompt."""

    news_context: str = ""
    earnings_context: str = ""


@dataclass
class PortfolioBContext:
    """Context about Portfolio B positions, trades, and complexity."""

    positions_for_agent: list[dict] = field(default_factory=list)
    recent_trades: list[dict] = field(default_factory=list)
    complexity: Any = None
    memory_text: str | None = None
    perf_text: str | None = None
    model_override: str | None = None
    position_map: dict[str, float] = field(default_factory=dict)
    cash: float = 0.0
    total_value: float = 0.0


@dataclass
class DynamicContext:
    """Dynamic context blocks for the Portfolio B system prompt."""

    trailing_context: str | None = None
    watchlist_context: str | None = None
    inverse_etf_context: str | None = None


def _active_portfolio_names(
    run_a: bool,
    run_b: bool,
    tenant_id: str,
) -> list[str]:
    """Return list of enabled portfolio names.

    Falls back to both if tenant_id is 'default' (legacy behavior).
    """
    if tenant_id == "default":
        return ["A", "B"]
    active: list[str] = []
    if run_a:
        active.append("A")
    if run_b:
        active.append("B")
    return active


def _get_trail_pct(strategy_mode: str, trade: TradeSchema, multiplier: float = 1.0) -> float:
    """Look up trailing stop percentage from strategy mode and trade conviction.

    Parses conviction from the trade reason (e.g. "high conviction" or
    "conviction: medium"). Falls back to "medium".

    Args:
        strategy_mode: conservative/standard/aggressive.
        trade: The trade to look up trail pct for.
        multiplier: Tenant trailing_stop_multiplier (0.5-2.0, default 1.0).
    """
    conviction = "medium"
    reason_lower = (trade.reason or "").lower()
    for level in ("high", "low"):
        if level in reason_lower:
            conviction = level
            break
    strategy_pcts = TRAIL_PCT.get(strategy_mode, TRAIL_PCT["conservative"])
    base_pct = strategy_pcts.get(conviction, strategy_pcts["medium"])
    return round(base_pct * multiplier, 4)


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
        self._news_aggregator = NewsAggregator()
        self._news_compactor = NewsCompactor()
        self._complexity_detector = ComplexityDetector()
        self._ticker_discovery = TickerDiscovery(db)
        self._risk_manager = RiskManager()
        self._performance_tracker = PerformanceTracker()
        self._memory_manager = AgentMemoryManager()
        self._regime_classifier = RegimeClassifier()

    async def run_all_tenants(
        self,
        today: date | None = None,
        session: str = "",
    ) -> list[dict]:
        """Run the daily pipeline for all active tenants.

        Iterates tenants sequentially with a delay between each to avoid
        broker rate limits. Each tenant's failure is isolated.

        Args:
            today: Override date for testing.
            session: Run label (e.g. "Morning", "Midday", "Closing").

        Returns:
            List of per-tenant summary dicts.
        """
        tenants = await self._db.get_active_tenants()
        if not tenants:
            # No tenants configured — fall back to default behavior
            log.info("no_tenants_configured_running_default")
            summary = await self.run_daily(today=today, session=session)
            return [summary]

        results: list[dict] = []
        for i, tenant in enumerate(tenants):
            # Skip tenants without complete credentials
            if not self.tenant_fully_configured(tenant):
                log.info(
                    "tenant_skipped_incomplete_config",
                    tenant_id=tenant.id,
                    tenant_name=tenant.name,
                )
                results.append(
                    {
                        "tenant_id": tenant.id,
                        "tenant_name": tenant.name,
                        "skipped": "incomplete_credentials",
                    }
                )
                continue

            log.info(
                "tenant_session_start",
                tenant_id=tenant.id,
                tenant_name=tenant.name,
                strategy=tenant.strategy_mode,
            )
            try:
                summary = await self.run_tenant_session(
                    tenant=tenant,
                    today=today,
                    session=session,
                )
                results.append(summary)
            except Exception as e:
                log.error(
                    "tenant_session_failed",
                    tenant_id=tenant.id,
                    tenant_name=tenant.name,
                    error=str(e),
                )
                results.append(
                    {
                        "tenant_id": tenant.id,
                        "tenant_name": tenant.name,
                        "error": str(e),
                    }
                )
                try:
                    from src.events.event_bus import Event, EventType, event_bus

                    event_bus.publish(
                        Event(
                            type=EventType.SYSTEM_ERROR,
                            tenant_id=tenant.id,
                            data={
                                "message": str(e)[:200],
                                "step": session,
                            },
                        )
                    )
                except Exception as exc:
                    log.debug("event_publish_failed", error=str(exc))
                # Try to notify tenant of the failure
                try:
                    from src.notifications.telegram_factory import TelegramFactory

                    notifier = TelegramFactory.get_notifier(tenant)
                    await notifier.send_error(f"Pipeline failed for {tenant.name} ({session}): {e}")
                except (ConnectionError, TimeoutError, OSError) as notify_err:
                    log.warning("pipeline_error_notification_failed", tenant_id=tenant.id, error=str(notify_err))

            # Delay between tenants (except after the last one)
            if i < len(tenants) - 1:
                await asyncio.sleep(settings.inter_tenant_delay)

        log.info(
            "all_tenants_complete",
            tenants=len(tenants),
            successes=sum(1 for r in results if "error" not in r),
        )
        return results

    async def run_tenant_session(
        self,
        tenant: TenantRow,
        today: date | None = None,
        session: str = "",
    ) -> dict:
        """Run the daily pipeline for a single tenant.

        Creates tenant-specific executor and notifier, then runs the
        standard pipeline with the tenant's configuration.

        Args:
            tenant: Active TenantRow.
            today: Override date for testing.
            session: Run label.

        Returns:
            Summary dict with tenant info.
        """
        from src.execution.client_factory import AlpacaClientFactory
        from src.notifications.telegram_factory import TelegramFactory

        # Create tenant-specific components
        notifier = TelegramFactory.get_notifier(tenant)
        client = AlpacaClientFactory.get_trading_client(tenant)

        from src.execution.alpaca_executor import AlpacaExecutor

        executor = AlpacaExecutor(self._db, client)

        # Capture initial equity on first run
        if tenant.initial_equity is None:
            equity = await self._capture_alpaca_equity(executor)
            if equity is not None:
                tenant.initial_equity = equity
                tenant.portfolio_a_cash = equity * tenant.portfolio_a_pct / 100
                tenant.portfolio_b_cash = equity * tenant.portfolio_b_pct / 100
                await self._db.update_tenant(
                    tenant.id,
                    {
                        "initial_equity": equity,
                        "portfolio_a_cash": tenant.portfolio_a_cash,
                        "portfolio_b_cash": tenant.portfolio_b_cash,
                    },
                )
                log.info(
                    "initial_equity_captured",
                    tenant_id=tenant.id,
                    equity=equity,
                    a_cash=tenant.portfolio_a_cash,
                    b_cash=tenant.portfolio_b_cash,
                )

        # Resolve allocations from tenant config
        alloc = resolve_from_tenant(tenant)

        # Resolve tenant-specific ticker universe (include discovered tickers)
        discovered = await self._ticker_discovery.get_active_tickers(
            tenant_id=tenant.id,
        )
        tenant_b_universe = get_tenant_universe(
            tenant,
            "B",
            discovered_tickers=discovered,
        )

        # Build tenant-scoped orchestrator state
        saved_executor = self._executor
        saved_notifier = self._notifier

        self._executor = executor
        self._notifier = notifier

        try:
            summary = await self.run_daily(
                today=today,
                session=session,
                tenant_id=tenant.id,
                strategy_mode=tenant.strategy_mode,
                run_portfolio_a=tenant.run_portfolio_a,
                run_portfolio_b=tenant.run_portfolio_b,
                allocations=alloc,
                portfolio_b_universe=tenant_b_universe,
            )
            summary["tenant_id"] = tenant.id
            summary["tenant_name"] = tenant.name
            return summary
        finally:
            # Restore original executor/notifier
            self._executor = saved_executor
            self._notifier = saved_notifier

    async def run_daily(
        self,
        today: date | None = None,
        session: str = "",
        tenant_id: str = "default",
        strategy_mode: str | None = None,
        run_portfolio_a: bool = True,
        run_portfolio_b: bool = True,
        allocations: TenantAllocations | None = None,
        portfolio_b_universe: list[str] | None = None,
    ) -> dict:
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
            session: Label for this run (e.g. "Morning", "Midday", "Closing").
            tenant_id: Tenant UUID for data isolation (default="default").
            strategy_mode: Override strategy mode (None = use settings).
            run_portfolio_a: Whether to run Portfolio A for this tenant.
            run_portfolio_b: Whether to run Portfolio B for this tenant.

        Returns:
            Summary dict with results from each step.
        """
        today = today or date.today()
        active_strategy = strategy_mode or settings.agent.strategy_mode
        alloc = allocations or DEFAULT_ALLOCATIONS
        summary: dict = {"date": today.isoformat(), "trades": {}, "errors": []}

        # Guard: skip if market is closed (holidays, weekends)
        if not is_market_open(today):
            log.info("pipeline_skipped_market_closed", date=str(today))
            if self._notifier_available():
                try:
                    await self._notifier.send_message(
                        f"Market closed today ({today.strftime('%A, %b %d')}). Pipeline skipped."
                    )
                except (ConnectionError, TimeoutError, OSError) as e:
                    log.warning("market_closed_notification_failed", error=str(e))
            summary["skipped"] = "market_closed"
            return summary

        log.info(
            "daily_pipeline_start",
            date=str(today),
            strategy_mode=active_strategy,
            tenant_id=tenant_id,
        )

        # Step 1: Initialize portfolios
        await self._executor.initialize_portfolios(allocations=alloc, tenant_id=tenant_id)

        # Steps 1.1–3.5: Market data, macro, regime, circuit breakers
        mkt = await self._fetch_market_context(
            today,
            tenant_id,
            alloc,
            run_portfolio_a,
            run_portfolio_b,
            summary,
        )
        if mkt is None:
            return summary
        closes = mkt.closes
        volumes = mkt.volumes
        alloc = mkt.allocations
        halted_portfolios = mkt.halted_portfolios

        # Step 4: Portfolio A — Momentum
        log.info("step_4_portfolio_a")
        trades_a: list = []
        if not run_portfolio_a:
            log.info("portfolio_a_skipped_not_configured", tenant_id=tenant_id)
            summary["trades"]["A"] = 0
            summary["a_reason"] = "Portfolio A not configured"
        elif "A" in halted_portfolios:
            log.info("portfolio_a_skipped_circuit_breaker")
            summary["trades"]["A"] = 0
            summary["a_reason"] = "Portfolio A halted (circuit breaker)"
        else:
            try:
                trades_a, a_reason = await self._run_portfolio_a(
                    closes,
                    today,
                    tenant_id=tenant_id,
                    allocations=alloc,
                )
                summary["trades"]["A"] = len(trades_a)
                summary["a_reason"] = a_reason
            except Exception as e:
                log.error("portfolio_a_failed", error=str(e))
                summary["errors"].append(f"Portfolio A failed: {e}")

        # Step 5: News, earnings, watchlist cleanup
        news_ctx = await self._build_news_context(
            closes,
            mkt.dynamic_universe,
            today,
            session,
            tenant_id,
            summary,
        )

        # Step 6: Portfolio B — AI Agent
        log.info("step_6_portfolio_b")
        trades_b: list = []
        b_tool_summary = None
        if not run_portfolio_b:
            log.info("portfolio_b_skipped_not_configured", tenant_id=tenant_id)
            summary["trades"]["B"] = 0
        elif "B" in halted_portfolios:
            log.info("portfolio_b_skipped_circuit_breaker")
            summary["trades"]["B"] = 0
        else:
            try:
                trades_b, b_reasoning, b_tool_summary = await self._run_portfolio_b(
                    closes,
                    volumes,
                    mkt.yield_curve,
                    mkt.vix,
                    today,
                    news_context=news_ctx.news_context,
                    session=session,
                    regime_result=mkt.regime_result,
                    tenant_id=tenant_id,
                    strategy_mode=active_strategy,
                    allocations=alloc,
                    portfolio_b_universe=portfolio_b_universe,
                    earnings_context=news_ctx.earnings_context or None,
                )
                summary["trades"]["B"] = len(trades_b)
                summary["b_reasoning"] = b_reasoning
                summary["b_tool_summary"] = b_tool_summary
            except Exception as e:
                log.error("portfolio_b_failed", error=str(e))
                summary["errors"].append(f"Portfolio B failed: {e}")

        # Steps 6.5–6.6: Risk filter + inverse hold times + approvals
        summary["_halted_portfolios"] = halted_portfolios
        all_trades = await self._filter_and_approve_trades(
            trades_a,
            trades_b,
            b_tool_summary,
            closes,
            today,
            session,
            tenant_id,
            mkt.regime_result,
            run_portfolio_b,
            alloc,
            mkt.trailing_stop_sells,
            summary,
        )
        summary.pop("_halted_portfolios", None)

        # Steps 7–8.5: Execute, trailing stops, snapshots, reconcile
        executed = await self._execute_and_record(
            all_trades,
            [],
            b_tool_summary,
            closes,
            today,
            tenant_id,
            active_strategy,
            alloc,
            run_portfolio_a,
            run_portfolio_b,
            summary,
        )

        # Step 9: Send Telegram notifications
        log.info("step_9_sending_notifications")
        await self._send_notifications(
            today=today,
            proposed_trades=all_trades,
            executed_trades=executed,
            summary=summary,
            session=session,
            regime_result=mkt.regime_result,
            tenant_id=tenant_id,
            strategy_mode=active_strategy,
            allocations=alloc,
            run_portfolio_a=run_portfolio_a,
            run_portfolio_b=run_portfolio_b,
            trailing_stop_alerts=mkt.trailing_stop_alerts,
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

    # ── Extracted helpers for run_daily() ─────────────────────────────────────

    async def _fetch_market_context(
        self,
        today: date,
        tenant_id: str,
        alloc: TenantAllocations,
        run_portfolio_a: bool,
        run_portfolio_b: bool,
        summary: dict,
    ) -> MarketContext | None:
        """Fetch market data, macro indicators, classify regime, check circuit breakers.

        Returns MarketContext on success, or None if market data is unavailable
        (caller should abort the pipeline).
        """
        # Step 1.1: Sync positions with broker (if supported)
        if hasattr(self._executor, "sync_positions"):
            try:
                await self._executor.sync_positions()
            except Exception as e:
                log.warning("position_sync_failed", error=str(e))

        # Step 1.2: Detect deposits
        try:
            alloc = await self._detect_deposits(alloc, tenant_id)
        except Exception as e:
            log.warning("deposit_detection_failed", error=str(e))

        # Step 1.5: Expire old dynamic tickers + build universe
        await self._ticker_discovery.expire_old(today, tenant_id=tenant_id)
        dynamic_universe = await get_dynamic_universe(self._db)

        # Step 2: Fetch market data
        log.info("step_2_fetching_market_data")
        try:
            data = await self._market_data.fetch_universe(tickers=dynamic_universe, period="1y")
        except Exception as e:
            log.error("market_data_fetch_failed", error=str(e))
            summary["errors"].append(f"Market data fetch failed: {e}")
            return None

        if not data:
            log.error("no_market_data")
            summary["errors"].append("No market data returned")
            return None

        closes = pd.DataFrame({t: df["Close"] for t, df in data.items()})
        volumes = pd.DataFrame({t: df["Volume"] for t, df in data.items()})
        closes = closes.sort_index()
        volumes = volumes.sort_index()
        summary["tickers_fetched"] = len(data)
        log.info("market_data_ready", tickers=len(data), rows=len(closes))

        # Step 2.1: Check trailing stops
        trailing_stop_sells: list[TradeSchema] = []
        trailing_stop_alerts: list[dict] = []
        try:
            trailing_stop_sells, trailing_stop_alerts = await self._check_trailing_stops(
                tenant_id,
                closes,
                run_portfolio_a,
                run_portfolio_b,
            )
            if trailing_stop_sells:
                summary["trailing_stops_triggered"] = len(trailing_stop_sells)
        except Exception as e:
            log.warning("trailing_stop_check_failed", error=str(e))

        # Step 2.3: Handle pending portfolio rebalance
        if tenant_id != "default":
            try:
                new_alloc = await self._handle_rebalance(
                    tenant_id,
                    closes,
                    run_portfolio_a,
                    run_portfolio_b,
                )
                if new_alloc is not None:
                    alloc = new_alloc
                    summary["rebalanced"] = True
            except Exception as e:
                log.error("rebalance_failed", error=str(e), tenant_id=tenant_id)
                summary["errors"].append(f"Rebalance failed: {e}")

        # Step 2.5: Missed day recovery
        try:
            recovered = await self.recovery_check(
                today,
                closes,
                tenant_id=tenant_id,
                allocations=alloc,
                run_portfolio_a=run_portfolio_a,
                run_portfolio_b=run_portfolio_b,
            )
            if recovered:
                summary["recovered_days"] = recovered
        except Exception as e:
            log.warning("recovery_check_failed", error=str(e))

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

        # Step 3.1: Classify market regime
        regime_result: RegimeResult | None = None
        try:
            regime_result = self._regime_classifier.classify(closes, vix=vix)
            summary["regime"] = regime_result.regime.value
            log.info("regime_classified", regime=regime_result.regime.value, summary=regime_result.summary)
        except Exception as e:
            log.warning("regime_classification_failed", error=str(e))

        # Step 3.5: Circuit breaker check
        halted_portfolios: set[str] = set()
        for pname in ("A", "B"):
            halted, reason = await self._risk_manager.check_circuit_breakers(
                pname,
                self._db,
                today,
                tenant_id=tenant_id,
            )
            if halted:
                halted_portfolios.add(pname)
                summary["errors"].append(f"Portfolio {pname} halted: {reason}")
                log.warning("portfolio_halted", portfolio=pname, reason=reason)
                try:
                    from src.events.event_bus import Event, EventType, event_bus

                    event_bus.publish(
                        Event(
                            type=EventType.CIRCUIT_BREAKER_TRIGGERED,
                            tenant_id=tenant_id,
                            data={"portfolio": pname, "reason": reason},
                        )
                    )
                except Exception as exc:
                    log.debug("event_publish_failed", error=str(exc))

        return MarketContext(
            closes=closes,
            volumes=volumes,
            dynamic_universe=dynamic_universe,
            trailing_stop_sells=trailing_stop_sells,
            trailing_stop_alerts=trailing_stop_alerts,
            yield_curve=yield_curve,
            vix=vix,
            regime_result=regime_result,
            allocations=alloc,
            halted_portfolios=halted_portfolios,
        )

    async def _build_news_context(
        self,
        closes: pd.DataFrame,
        dynamic_universe: list[str],
        today: date,
        session: str,
        tenant_id: str,
        summary: dict,
    ) -> NewsContext:
        """Fetch news from all sources, store in ChromaDB, compact for agent, fetch earnings."""
        news_context = ""
        log.info("step_5_fetching_news")
        try:
            tickers_for_news = list(closes.columns)[:20]
            raw_articles = self._news_aggregator.fetch_all(tickers_for_news)
            summary["news_articles"] = len(raw_articles)

            # Store in ChromaDB
            if raw_articles:
                store_dicts = [
                    {
                        "ticker": a.tickers[0] if a.tickers else "",
                        "title": a.headline,
                        "link": a.url,
                        "publisher": a.publisher,
                        "published": a.published_at,
                    }
                    for a in raw_articles
                ]
                try:
                    rows = self._news_fetcher.store_articles(store_dicts)
                    async with self._db.session() as s:
                        s.add_all(rows)
                        await s.commit()
                except Exception as e:
                    log.warning("news_store_failed", error=str(e))

            # Compact for agent prompt
            positions_b = await self._db.get_positions("B", tenant_id=tenant_id)
            held_tickers = [p.ticker for p in positions_b]
            if len(closes) >= 2:
                pct = ((closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2]).abs().dropna()
                movers = pct.sort_values(ascending=False).head(5).index.tolist()
            else:
                movers = []

            news_context = self._news_compactor.compact(
                articles=raw_articles,
                held_tickers=held_tickers,
                top_movers=movers,
                universe_tickers=set(dynamic_universe),
            )

            # Step 5.1: Append historical context from ChromaDB
            try:
                today_headlines = [a.headline for a in raw_articles] if raw_articles else []
                historical = self._news_fetcher.get_historical_context(
                    held_tickers=held_tickers,
                    today_headlines=today_headlines,
                )
                if historical:
                    news_context = news_context + historical
            except Exception as e:
                log.warning("historical_context_failed", error=str(e))

        except Exception as e:
            log.warning("news_fetch_failed", error=str(e))
            summary["errors"].append(f"News fetch failed: {e}")

        # Step 5.5: Earnings calendar (Morning session only)
        earnings_context = ""
        if session == "Morning":
            try:
                from src.data.earnings_calendar import EarningsCalendar

                earnings_cal = EarningsCalendar()
                await earnings_cal.refresh_earnings(self._db, list(closes.columns))
                positions_b = await self._db.get_positions("B", tenant_id=tenant_id)
                held_tickers = [p.ticker for p in positions_b]
                all_tickers = list(set(held_tickers + list(closes.columns)))
                upcoming = await earnings_cal.get_upcoming(self._db, all_tickers, days_ahead=14)
                if upcoming:
                    e_lines = ["Upcoming Earnings (next 14 days):"]
                    for row in upcoming:
                        days_until = (row.earnings_date - today).days
                        warning = ""
                        if days_until <= 3:
                            warning = " — IMMINENT"
                        elif days_until <= 5 and row.ticker in held_tickers:
                            warning = " — consider reducing position"
                        e_lines.append(
                            f"- {row.ticker}: {row.earnings_date.strftime('%b %d')} ({days_until}d away){warning}"
                        )
                    e_lines.append(
                        "\nNote: Earnings dates are approximate. "
                        "Consider reducing positions or tightening stops "
                        "3-5 days before earnings."
                    )
                    earnings_context = "\n".join(e_lines)
                    summary["earnings_alerts"] = len(upcoming)
            except Exception as e:
                log.warning("earnings_fetch_failed", error=str(e))

        # Step 5.6: Clean up expired watchlist (Morning only)
        if session == "Morning":
            try:
                expired_wl = await self._db.cleanup_expired_watchlist(tenant_id)
                if expired_wl:
                    log.info("watchlist_expired", count=expired_wl, tenant_id=tenant_id)
            except Exception as e:
                log.warning("watchlist_cleanup_failed", error=str(e))

        # Step 5.7: Morning queue — inject overnight sentinel actions into context
        if session == "Morning":
            queue_ctx = await self._process_morning_queue(tenant_id)
            if queue_ctx:
                news_context = queue_ctx + "\n\n" + news_context
                summary["overnight_queue_items"] = queue_ctx.count("[")

        # Step 5.8: Gap risk — inject into close and manual session context
        if session in ("Closing", "Manual"):
            gap_ctx = await self._build_gap_risk_context(tenant_id)
            if gap_ctx:
                news_context = gap_ctx + "\n\n" + news_context
                summary["gap_risk_injected"] = True

        return NewsContext(news_context=news_context, earnings_context=earnings_context)

    async def _filter_and_approve_trades(
        self,
        trades_a: list,
        trades_b: list,
        b_tool_summary: dict | None,
        closes: pd.DataFrame,
        today: date,
        session: str,
        tenant_id: str,
        regime_result: RegimeResult | None,
        run_portfolio_b: bool,
        alloc: TenantAllocations,
        trailing_stop_sells: list,
        summary: dict,
    ) -> list:
        """Resolve posture, apply risk filter, check inverse holds, handle approvals.

        Returns the final list of approved trades (including trailing stop sells).
        """
        log.info("step_6_5_risk_filter")

        # Resolve posture from Portfolio B tool summary
        posture_limits_b = None
        if b_tool_summary and isinstance(b_tool_summary, dict):
            declared = b_tool_summary.get("declared_posture")
            if declared:
                try:
                    from src.agent.posture import PostureLevel, PostureManager

                    posture_mgr = PostureManager()
                    posture_level = PostureLevel(declared)

                    # For aggressive, compute track record stats for gate check
                    tr_total, tr_wr, tr_alpha = 0, 0.0, None
                    if posture_level == PostureLevel.AGGRESSIVE:
                        try:
                            from src.analysis.outcome_tracker import OutcomeTracker
                            from src.analysis.track_record import TrackRecord

                            tracker = OutcomeTracker(self._db)
                            outcomes = await tracker.get_recent_outcomes(days=90, tenant_id=tenant_id)
                            if outcomes:
                                stats = TrackRecord().compute(outcomes, min_trades=1)
                                tr_total = stats.total_trades
                                tr_wr = stats.win_rate_pct
                                tr_alpha = stats.avg_alpha_vs_spy
                        except Exception as e:
                            log.warning("posture_track_record_fetch_failed", error=str(e))

                    posture_limits_b, effective_posture = posture_mgr.resolve_effective_limits(
                        posture_level,
                        total_trades=tr_total,
                        win_rate_pct=tr_wr,
                        avg_alpha_vs_spy=tr_alpha,
                    )

                    # Save posture history
                    try:
                        reason = b_tool_summary.get("posture_reason", "")
                        await self._db.save_posture(
                            tenant_id=tenant_id,
                            session_date=today,
                            session_label=session,
                            posture=declared,
                            effective_posture=effective_posture.value,
                            reason=reason,
                        )
                    except Exception as e:
                        log.warning("posture_save_failed", error=str(e))

                    log.info(
                        "posture_resolved",
                        declared=declared,
                        effective=effective_posture.value,
                        limits_single=posture_limits_b.max_single_position_pct,
                        limits_sector=posture_limits_b.max_sector_concentration,
                    )
                    try:
                        from src.events.event_bus import Event, EventType, event_bus

                        event_bus.publish(
                            Event(
                                type=EventType.POSTURE_CHANGED,
                                tenant_id=tenant_id,
                                data={
                                    "declared": declared,
                                    "effective": effective_posture.value,
                                },
                            )
                        )
                    except Exception as exc:
                        log.debug("event_publish_failed", error=str(exc))
                except (ValueError, KeyError) as e:
                    log.warning("posture_resolve_failed", declared=declared, error=str(e))

        # Determine effective posture string for risk checks
        effective_posture_str: str | None = None
        if b_tool_summary and isinstance(b_tool_summary, dict):
            declared = b_tool_summary.get("declared_posture")
            if declared:
                effective_posture_str = declared

        # Step 6.6: Check inverse hold times for Portfolio B
        inverse_hold_alerts: list[dict] = []
        halted_portfolios = summary.get("_halted_portfolios", set())
        if run_portfolio_b and "B" not in halted_portfolios:
            try:
                inverse_hold_alerts = await self._risk_manager.check_inverse_hold_times(
                    self._db, "B", tenant_id=tenant_id
                )
                if inverse_hold_alerts:
                    log.info("inverse_hold_alerts", count=len(inverse_hold_alerts))
                    summary["inverse_hold_alerts"] = len(inverse_hold_alerts)
            except Exception as e:
                log.warning("inverse_hold_time_check_failed", error=str(e))

        all_trades: list = []
        for pname, trades in [("A", trades_a), ("B", trades_b)]:
            if not trades:
                continue
            portfolio = await self._db.get_portfolio(pname, tenant_id=tenant_id)
            positions = await self._db.get_positions(pname, tenant_id=tenant_id)
            position_map = {p.ticker: p.shares for p in positions}
            pval = portfolio.total_value if portfolio else alloc.for_portfolio(pname)
            pcash = portfolio.cash if portfolio else pval

            latest_prices = {t: float(closes[t].iloc[-1]) for t in closes.columns if not pd.isna(closes[t].iloc[-1])}
            verdict = self._risk_manager.check_pre_trade(
                trades=trades,
                portfolio_name=pname,
                current_positions=position_map,
                latest_prices=latest_prices,
                portfolio_value=pval,
                cash=pcash,
                posture_limits=posture_limits_b if pname == "B" else None,
                regime=regime_result.regime.value if regime_result else None,
                current_posture=effective_posture_str if pname == "B" else None,
            )
            all_trades.extend(verdict.allowed)
            for blocked_trade, reason in verdict.blocked:
                log.warning("trade_blocked_by_risk", portfolio=pname, ticker=blocked_trade.ticker, reason=reason)
                try:
                    from src.events.event_bus import Event, EventType, event_bus

                    event_bus.publish(
                        Event(
                            type=EventType.TRADE_REJECTED,
                            tenant_id=tenant_id,
                            data={
                                "ticker": blocked_trade.ticker,
                                "side": blocked_trade.side.value,
                                "shares": blocked_trade.shares,
                                "reason": reason,
                                "portfolio": pname,
                            },
                        )
                    )
                except Exception as exc:
                    log.debug("event_publish_failed", error=str(exc))

            # Handle inverse trades requiring approval
            if verdict.requires_approval:
                for inv_trade in verdict.requires_approval:
                    approved = await self._request_inverse_trade_approval(inv_trade, regime_result)
                    if approved == "approve":
                        log.info("inverse_trade_approved", ticker=inv_trade.ticker)
                    else:
                        log.info("inverse_trade_rejected", ticker=inv_trade.ticker)
                        if inv_trade in verdict.allowed:
                            verdict.allowed.remove(inv_trade)
                            all_trades[:] = [t for t in all_trades if t is not inv_trade]
                        verdict.blocked.append((inv_trade, "Rejected via Telegram approval"))

            # Handle large trades requiring approval (non-inverse, >threshold% of portfolio)
            if verdict.requires_trade_approval:
                for lg_trade, approval_reason in verdict.requires_trade_approval:
                    trade_pct = (lg_trade.total / pval * 100) if pval > 0 else 0
                    choice = await self._request_large_trade_approval(lg_trade, trade_pct, approval_reason, tenant_id)
                    if choice == "approve":
                        log.info("large_trade_approved", ticker=lg_trade.ticker)
                    else:
                        log.info("large_trade_rejected", ticker=lg_trade.ticker)
                        if lg_trade in verdict.allowed:
                            verdict.allowed.remove(lg_trade)
                            all_trades[:] = [t for t in all_trades if t is not lg_trade]
                        verdict.blocked.append((lg_trade, f"Rejected via Telegram: {approval_reason}"))

        # Merge trailing stop sells (bypass risk filter — stops ARE the risk mechanism)
        all_trades.extend(trailing_stop_sells)
        return all_trades

    async def _execute_and_record(
        self,
        all_trades: list,
        executed_handler_trades: list,
        b_tool_summary: dict | None,
        closes: pd.DataFrame,
        today: date,
        tenant_id: str,
        active_strategy: str,
        alloc: TenantAllocations,
        run_portfolio_a: bool,
        run_portfolio_b: bool,
        summary: dict,
    ) -> list:
        """Execute trades, manage trailing stops, take snapshots, reconcile equity.

        Returns the list of executed trades.
        """
        executed: list = []
        if all_trades:
            executed = await self._executor.execute_trades(all_trades, tenant_id=tenant_id)
            summary["trades_executed"] = len(executed)
            # Publish SSE events for each executed trade
            try:
                from src.events.event_bus import Event, EventType, event_bus

                for trade in executed:
                    event_bus.publish(
                        Event(
                            type=EventType.TRADE_EXECUTED,
                            tenant_id=tenant_id,
                            data={
                                "ticker": trade.ticker,
                                "side": trade.side.value,
                                "shares": trade.shares,
                                "price": trade.price,
                                "portfolio": trade.portfolio.value,
                            },
                        )
                    )
            except Exception as exc:
                log.debug("event_publish_failed", error=str(exc))
        else:
            summary["trades_executed"] = 0

        # Step 7.1: Create trailing stops for new buys + auto-remove from watchlist
        agent_stop_requests: dict[str, float] = {}
        if b_tool_summary and isinstance(b_tool_summary, dict):
            for req in b_tool_summary.get("trailing_stop_requests", []):
                agent_stop_requests[req["ticker"]] = req["trail_pct"]

        # Get tenant trailing stop multiplier (scales TRAIL_PCT matrix)
        tenant = await self._db.get_tenant(tenant_id)
        trail_multiplier = tenant.trailing_stop_multiplier if tenant and tenant.trailing_stop_multiplier else 1.0

        for trade in executed:
            if trade.side.value == "BUY":
                trail_pct = agent_stop_requests.pop(
                    trade.ticker, _get_trail_pct(active_strategy, trade, trail_multiplier)
                )
                try:
                    await self._db.create_trailing_stop(
                        tenant_id=tenant_id,
                        portfolio=trade.portfolio.value,
                        ticker=trade.ticker,
                        entry_price=trade.price,
                        trail_pct=trail_pct,
                    )
                except Exception as e:
                    log.warning("trailing_stop_create_failed", ticker=trade.ticker, error=str(e))
                try:
                    await self._db.remove_watchlist_if_traded(tenant_id, trade.ticker)
                except Exception as e:
                    log.warning("watchlist_auto_remove_failed", error=str(e))

        # Step 7.1.1: Apply agent stop requests for existing positions (not newly bought)
        for ticker, trail_pct in agent_stop_requests.items():
            try:
                await self._db.update_trailing_stop_pct(tenant_id, "B", ticker, trail_pct)
                log.info("agent_trailing_stop_updated", ticker=ticker, trail_pct=trail_pct)
            except Exception as e:
                log.warning("agent_trailing_stop_update_failed", ticker=ticker, error=str(e))

        # Step 7.2: Deactivate trailing stops for sold positions
        for trade in executed:
            if trade.side.value == "SELL":
                try:
                    await self._db.deactivate_trailing_stops_for_ticker(
                        tenant_id,
                        trade.portfolio.value,
                        trade.ticker,
                    )
                except Exception as e:
                    log.warning("trailing_stop_deactivate_failed", error=str(e))

        # Step 8: Take snapshots (only for enabled portfolios)
        log.info("step_8_taking_snapshots")
        latest_prices: dict[str, float] = {}
        if hasattr(self._executor, "_client"):
            try:
                alpaca_positions = await asyncio.to_thread(self._executor._client.get_all_positions)
                for pos in alpaca_positions:
                    latest_prices[pos.symbol] = float(pos.current_price)
            except Exception as e:
                log.warning("alpaca_prices_fetch_failed", error=str(e))
        for t in closes.columns:
            if t not in latest_prices and not pd.isna(closes[t].iloc[-1]):
                latest_prices[t] = float(closes[t].iloc[-1])
        snapshot_portfolios = _active_portfolio_names(run_portfolio_a, run_portfolio_b, tenant_id)
        for portfolio_name in snapshot_portfolios:
            try:
                await self._executor.take_snapshot(
                    portfolio_name,
                    today,
                    latest_prices,
                    allocations=alloc,
                    tenant_id=tenant_id,
                )
            except Exception as e:
                log.error("snapshot_failed", portfolio=portfolio_name, error=str(e))

        # Publish data refresh events
        try:
            from src.events.event_bus import Event, EventType, event_bus

            if executed:
                event_bus.publish(
                    Event(
                        type=EventType.POSITIONS_UPDATED,
                        tenant_id=tenant_id,
                        data={"trades_executed": len(executed)},
                    )
                )
            for pname in snapshot_portfolios:
                event_bus.publish(
                    Event(
                        type=EventType.PORTFOLIO_SNAPSHOT,
                        tenant_id=tenant_id,
                        data={"portfolio": pname, "date": str(today)},
                    )
                )
        except Exception as exc:
            log.debug("event_publish_failed", error=str(exc))

        # Step 8.5: Reconcile equity against Alpaca
        try:
            drift = await self._reconcile_equity(tenant_id, run_portfolio_a, run_portfolio_b, alloc)
            if drift is not None:
                summary["equity_drift_corrected"] = round(drift, 2)
        except Exception as e:
            log.warning("equity_reconciliation_failed", error=str(e))

        return executed

    # ── Extracted helpers for _run_portfolio_b() ──────────────────────────────

    async def _build_portfolio_b_context(
        self,
        closes: pd.DataFrame,
        vix: float | None,
        regime_result: RegimeResult | None,
        tenant_id: str,
        alloc: TenantAllocations,
        session: str,
    ) -> PortfolioBContext:
        """Build position/trade/complexity context for Portfolio B."""
        portfolio = await self._db.get_portfolio("B", tenant_id=tenant_id)
        positions = await self._db.get_positions("B", tenant_id=tenant_id)
        position_map = {p.ticker: p.shares for p in positions}
        cash = portfolio.cash if portfolio else alloc.portfolio_b_cash
        total_value = portfolio.total_value if portfolio else alloc.portfolio_b_cash

        positions_for_agent = [
            {
                "ticker": p.ticker,
                "shares": p.shares,
                "avg_price": p.avg_price,
                "market_value": p.shares * float(closes[p.ticker].iloc[-1]) if p.ticker in closes.columns else 0,
            }
            for p in positions
        ]

        recent_trades_raw = await self._db.get_trades("B", tenant_id=tenant_id)
        recent_trades = [
            {"ticker": t.ticker, "side": t.side, "shares": t.shares, "price": t.price, "reason": t.reason or ""}
            for t in recent_trades_raw[:5]
        ]

        # Complexity detection & model routing
        snapshots = await self._db.get_snapshots("B", tenant_id=tenant_id)
        peak_value = max((s.total_value for s in snapshots if s.total_value), default=total_value)

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
                except (ValueError, KeyError, IndexError) as e:
                    log.debug("held_indicator_failed", ticker=p.ticker, error=str(e))

        regime_str = regime_result.regime.value if regime_result else None
        complexity = self._complexity_detector.evaluate(
            closes=closes,
            positions=positions_for_agent,
            total_value=total_value,
            peak_value=peak_value,
            regime_today=regime_str,
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
                return PortfolioBContext(model_override="__skip__")

        # Build memory context
        memory_text: str | None = None
        try:
            memories = await self._db.get_all_agent_memory_context(tenant_id=tenant_id)
            memory_text = self._memory_manager.build_memory_prompt(memories) or None
        except Exception as e:
            log.warning("memory_context_failed", error=str(e))

        # Compute performance stats
        perf_text: str | None = None
        try:
            spy_closes = closes["SPY"] if "SPY" in closes.columns else None
            stats = await self._performance_tracker.get_portfolio_stats(
                self._db,
                "B",
                alloc.portfolio_b_cash,
                spy_closes=spy_closes,
            )
            if stats.days_tracked > 0:
                perf_text = self._performance_tracker.format_for_prompt(stats)
        except Exception as e:
            log.warning("performance_stats_failed", error=str(e))

        # Correlation monitor (Morning session only)
        if session == "Morning" and positions_for_agent:
            try:
                held = [p["ticker"] for p in positions_for_agent]
                corr = self._risk_manager.compute_portfolio_correlation(closes, held)
                if corr["matrix_size"] >= 2:
                    corr_text = self._format_correlation(corr)
                    if perf_text:
                        perf_text += f"\n{corr_text}"
                    else:
                        perf_text = corr_text
            except Exception as e:
                log.warning("correlation_failed", error=str(e))

        return PortfolioBContext(
            positions_for_agent=positions_for_agent,
            recent_trades=recent_trades,
            complexity=complexity,
            memory_text=memory_text,
            perf_text=perf_text,
            model_override=model_override,
            position_map=position_map,
            cash=cash,
            total_value=total_value,
        )

    async def _process_morning_queue(self, tenant_id: str) -> str | None:
        """Process queued sentinel actions. Returns context for agent or None."""
        try:
            pending = await self._db.get_pending_sentinel_actions(tenant_id)
        except Exception as e:
            log.warning("morning_queue_fetch_failed", tenant_id=tenant_id, error=str(e))
            return None

        if not pending:
            return None

        queue_context = "OVERNIGHT QUEUE — Review these before trading:\n\n"
        for action in pending:
            queue_context += (
                f"  [{action['alert_level'].upper()}] {action['action_type'].upper()} "
                f"{action['ticker']} — {action['reason']}\n"
                f"  Detected: {action['created_at']} | Source: {action['source']}\n\n"
            )
        queue_context += (
            "For each item: execute the action, modify it, or cancel it. "
            "Use execute_trade or save_observation to document your decision."
        )
        log.info("morning_queue_loaded", tenant_id=tenant_id, items=len(pending))
        return queue_context

    async def _build_gap_risk_context(self, tenant_id: str) -> str | None:
        """Build gap risk context for the close session."""
        try:
            from src.analysis.gap_risk import GapRiskAnalyzer

            analyzer = GapRiskAnalyzer()
            assessment = await analyzer.analyze(self._db, tenant_id)
            if assessment.rating in ("HIGH", "EXTREME"):
                ctx = f"OVERNIGHT GAP RISK: {assessment.rating} (score {assessment.aggregate_risk_score})\n"
                if assessment.earnings_tonight:
                    ctx += f"Earnings tonight: {', '.join(assessment.earnings_tonight)}\n"
                for p in assessment.positions[:5]:
                    if p.recommendation:
                        ctx += f"  {p.ticker}: {p.recommendation}\n"
                ctx += "\nConsider reducing high-risk positions before close."
                return ctx
        except Exception as e:
            log.warning("gap_risk_context_failed", tenant_id=tenant_id, error=str(e))
        return None

    async def _build_outcome_feedback(
        self,
        tenant_id: str,
    ) -> tuple[str | None, str | None]:
        """Build decision review and track record text for the system prompt."""
        decision_review_text: str | None = None
        track_record_text: str | None = None
        try:
            from src.analysis.outcome_tracker import OutcomeTracker
            from src.analysis.track_record import TrackRecord

            outcome_tracker = OutcomeTracker(self._db)
            outcomes = await outcome_tracker.get_recent_outcomes(days=30, tenant_id=tenant_id)
            if outcomes:
                decision_review_text = _build_decision_review(outcomes[-5:])
                stats = TrackRecord().compute(outcomes)
                track_record_text = _build_track_record(stats)
        except Exception as e:
            log.warning("outcome_feedback_failed", error=str(e))
        return decision_review_text, track_record_text

    async def _build_dynamic_context(
        self,
        closes: pd.DataFrame,
        positions: list,
        today: date,
        tenant_id: str,
    ) -> DynamicContext:
        """Build trailing stops, watchlist, and inverse ETF context for Portfolio B."""
        trailing_context: str | None = None
        try:
            stops_b = await self._db.get_active_trailing_stops(tenant_id, "B")
            if stops_b:
                stop_lines = []
                for s in stops_b:
                    current = float(closes[s.ticker].iloc[-1]) if s.ticker in closes.columns else s.peak_price
                    pct_from_stop = ((current - s.stop_price) / current) * 100
                    stop_lines.append(
                        f"- {s.ticker}: entry ${s.entry_price:.2f}, "
                        f"peak ${s.peak_price:.2f}, "
                        f"stop ${s.stop_price:.2f} ({s.trail_pct * 100:.1f}% trail) "
                        f"— {pct_from_stop:.1f}% from trigger"
                    )
                trailing_context = "\n".join(stop_lines)
        except Exception as e:
            log.warning("trailing_context_build_failed", error=str(e))

        watchlist_context: str | None = None
        try:
            watchlist = await self._db.get_watchlist(tenant_id)
            if watchlist:
                wl_lines = []
                for w in watchlist:
                    days_left = (w.expires_at - today).days
                    target = f", target ${w.target_entry:.2f}" if w.target_entry else ""
                    wl_lines.append(
                        f'- {w.ticker}: "{w.reason}" ({w.conviction} conviction{target}, {days_left}d left)'
                    )
                watchlist_context = "\n".join(wl_lines)
        except Exception as e:
            log.warning("watchlist_context_build_failed", error=str(e))

        inverse_etf_context: str | None = None
        try:
            from config.universe import INVERSE_ETF_META

            inverse_lines: list[str] = []
            for p in positions:
                ticker = p.ticker if hasattr(p, "ticker") else p.get("ticker", "")
                if ticker in INVERSE_ETF_META:
                    meta = INVERSE_ETF_META[ticker]
                    avg_price = p.avg_price if hasattr(p, "avg_price") else p.get("avg_price", 0)
                    shares = p.shares if hasattr(p, "shares") else p.get("shares", 0)
                    price = float(closes[ticker].iloc[-1]) if ticker in closes.columns else avg_price
                    value = shares * price
                    pnl_pct = ((price - avg_price) / avg_price) * 100 if avg_price > 0 else 0
                    inverse_lines.append(
                        f"- {ticker} ({meta['description']}): {shares:.0f} shares, ${value:,.0f}, P&L {pnl_pct:+.1f}%"
                    )
            if inverse_lines:
                inverse_etf_context = "\n".join(inverse_lines)
        except Exception as e:
            log.warning("inverse_context_build_failed", error=str(e))

        return DynamicContext(
            trailing_context=trailing_context,
            watchlist_context=watchlist_context,
            inverse_etf_context=inverse_etf_context,
        )

    def _build_portfolio_b_prompt(
        self,
        perf_text: str | None,
        memory_text: str | None,
        active_strategy: str,
        session: str,
        regime_result: RegimeResult | None,
        alloc: TenantAllocations,
        portfolio_b_universe: list[str] | None,
        dynamic_ctx: DynamicContext,
        earnings_context: str | None,
        decision_review_text: str | None,
        track_record_text: str | None,
    ) -> str:
        """Assemble the dynamic system prompt for Portfolio B from all context blocks."""
        return build_system_prompt(
            performance_stats=perf_text,
            memory_context=memory_text,
            strategy_mode=active_strategy,
            session=session,
            regime_summary=regime_result.summary if regime_result else None,
            portfolio_allocation=alloc.portfolio_b_cash,
            universe_size=(len(portfolio_b_universe) if portfolio_b_universe else None),
            trailing_stops_context=dynamic_ctx.trailing_context,
            earnings_context=earnings_context,
            watchlist_context=dynamic_ctx.watchlist_context,
            decision_review=decision_review_text,
            track_record=track_record_text,
            inverse_etf_context=dynamic_ctx.inverse_etf_context,
        )

    async def _run_portfolio_a(
        self,
        closes: pd.DataFrame,
        today: date,
        tenant_id: str = "default",
        allocations: TenantAllocations | None = None,
    ) -> tuple[list, str]:
        """Run Portfolio A momentum strategy and return trades with reason."""
        alloc = allocations or DEFAULT_ALLOCATIONS
        portfolio = await self._db.get_portfolio("A", tenant_id=tenant_id)
        positions = await self._db.get_positions("A", tenant_id=tenant_id)
        position_map = {p.ticker: p.shares for p in positions}
        cash = portfolio.cash if portfolio else alloc.portfolio_a_cash

        total_value = portfolio.total_value if portfolio else alloc.portfolio_a_cash
        trades = self._strategy_a.generate_trades(
            closes,
            position_map,
            cash,
            portfolio_value=total_value,
        )

        # Save momentum rankings
        ranking_rows = self._strategy_a.get_ranking_rows(closes, today)
        if ranking_rows:
            await self._db.save_momentum_rankings(ranking_rows)

        target = self._strategy_a.get_target_ticker(closes)

        # Build reason for no-trade explanation
        if not trades:
            reason = f"Holding momentum target {target}" if target else "No momentum signal"
        else:
            reason = f"Rebalancing to {target}"

        log.info("portfolio_a_complete", trades=len(trades), target=target)
        return trades, reason

    async def _run_portfolio_b(
        self,
        closes,
        volumes,
        yield_curve,
        vix,
        today,
        news_context: str = "",
        session: str = "",
        regime_result: RegimeResult | None = None,
        tenant_id: str = "default",
        strategy_mode: str | None = None,
        allocations: TenantAllocations | None = None,
        portfolio_b_universe: list[str] | None = None,
        earnings_context: str | None = None,
    ):
        """Run Portfolio B AI strategy with complexity-based model routing."""
        alloc = allocations or DEFAULT_ALLOCATIONS
        active_strategy = strategy_mode or settings.agent.strategy_mode
        regime_str = regime_result.regime.value if regime_result else None

        # Build context (positions, trades, complexity, memory, perf)
        pb_ctx = await self._build_portfolio_b_context(
            closes,
            vix,
            regime_result,
            tenant_id,
            alloc,
            session,
        )
        if pb_ctx.model_override == "__skip__":
            return [], "", None

        positions_for_agent = pb_ctx.positions_for_agent
        recent_trades = pb_ctx.recent_trades
        complexity = pb_ctx.complexity
        model_override = pb_ctx.model_override
        position_map = pb_ctx.position_map
        cash = pb_ctx.cash
        total_value = pb_ctx.total_value

        # Build outcome feedback
        decision_review_text, track_record_text = await self._build_outcome_feedback(tenant_id)

        # Build dynamic context blocks (trailing stops, watchlist, inverse ETFs)
        positions = await self._db.get_positions("B", tenant_id=tenant_id)
        dynamic_ctx = await self._build_dynamic_context(closes, positions, today, tenant_id)

        # Assemble system prompt
        dynamic_prompt = self._build_portfolio_b_prompt(
            pb_ctx.perf_text,
            pb_ctx.memory_text,
            active_strategy,
            session,
            regime_result,
            alloc,
            portfolio_b_universe,
            dynamic_ctx,
            earnings_context,
            decision_review_text,
            track_record_text,
        )

        # ── Check agentic mode ────────────────────────────────────────────
        tenant = await self._db.get_tenant(tenant_id)
        use_persistent = getattr(tenant, "use_persistent_agent", False) if tenant else False
        use_agent_loop = getattr(tenant, "use_agent_loop", False) if tenant else False

        tool_summary: dict | None = None

        if use_persistent:
            # ── Persistent agent path ────────────────────────────────────
            return await self._run_portfolio_b_persistent(
                tenant_id=tenant_id,
                trigger_type=session.lower() if session else "morning",
                dynamic_prompt=dynamic_prompt,
                closes=closes,
                volumes=volumes,
                vix=vix,
                yield_curve=yield_curve,
                regime_str=regime_str,
                news_context=news_context,
                positions_for_agent=positions_for_agent,
                cash=cash,
                total_value=total_value,
                recent_trades=recent_trades,
                model_override=model_override,
                portfolio_b_universe=portfolio_b_universe,
                allocations=alloc,
                today=today,
                session=session,
                strategy_mode=active_strategy,
            )

        if use_agent_loop:
            # ── Agentic path: two-phase seed → investigate ───────────────
            log.info("portfolio_b_agentic_mode", tenant_id=tenant_id)
            from src.agent.agent_runner import AgentRunner
            from src.agent.tools.actions import ActionState, register_action_tools
            from src.agent.tools.market import register_market_tools
            from src.agent.tools.news import register_news_tools
            from src.agent.tools.portfolio import register_portfolio_tools

            # Shared context (called once for both phases)
            context = self._strategy_b.prepare_context(
                closes=closes,
                volumes=volumes,
                positions=positions_for_agent,
                cash=cash,
                total_value=total_value,
                recent_trades=recent_trades,
                regime=regime_str,
                yield_curve=yield_curve,
                vix=vix,
                news_context=news_context,
                system_prompt=dynamic_prompt,
                universe=portfolio_b_universe,
            )

            # ── Phase 1: Seed (single-shot, complexity-routed model) ────
            context["model_override"] = model_override
            seed_response = self._strategy_b._agent.analyze(**context)

            seed_trades = seed_response.get("trades", [])
            seed_reasoning = seed_response.get("reasoning", "")
            seed_regime = seed_response.get("regime_assessment", "")
            seed_risk = seed_response.get("risk_notes", "")

            log.info(
                "seed_phase_complete",
                trades=len(seed_trades),
                reasoning=seed_reasoning[:100],
            )

            # ── Phase 2: Investigate (agentic loop, always tool model) ──
            investigation_message = (
                f"## Initial Analysis (from senior analyst)\n"
                f"Regime: {seed_regime}\n"
                f"Reasoning: {seed_reasoning}\n"
                f"Proposed Trades: {json.dumps(seed_trades)}\n"
                f"Risk Notes: {seed_risk}\n\n"
                f"## Your Task\n"
                f"Use tools to verify or refine. Submit final trades via propose_trades."
            )

            investigation_prompt = (
                dynamic_prompt + "\n\nYou have tools to investigate before finalizing. "
                "Verify key assumptions. When satisfied, submit trades via propose_trades."
            )

            runner = AgentRunner(
                api_key=settings.anthropic_api_key,
                model=settings.agent.agent_tool_model,
                max_turns=settings.agent.agent_max_turns,
                max_cost_usd=settings.agent.agent_session_budget,
            )

            current_prices = {t: float(closes[t].iloc[-1]) for t in closes.columns if not pd.isna(closes[t].iloc[-1])}

            action_state = ActionState()
            register_portfolio_tools(runner.registry, self._db, tenant_id, current_prices, closes=closes)
            held_tickers = [p["ticker"] for p in positions_for_agent] if positions_for_agent else []
            register_market_tools(
                runner.registry,
                closes,
                vix=vix,
                yield_curve=yield_curve,
                regime=regime_str,
                db=self._db,
                held_tickers=held_tickers,
                tenant_id=tenant_id,
            )
            register_news_tools(
                runner.registry,
                news_context,
                news_fetcher=getattr(self, "_news_fetcher", None),
                db=self._db,
                tenant_id=tenant_id,
                current_prices=current_prices,
            )
            register_action_tools(
                runner.registry,
                action_state,
                db=self._db,
                tenant_id=tenant_id,
                ticker_discovery=self._ticker_discovery,
            )

            result = await runner.run(
                system_prompt=investigation_prompt,
                user_message=investigation_message,
            )

            # ── Merge: investigation overrides seed ─────────────────────
            response = result.response
            accumulated = action_state.get_accumulated_state()

            # Use investigation trades if any, otherwise fall back to seed
            if accumulated["trades"]:
                response["trades"] = accumulated["trades"]
            elif not response.get("trades"):
                response["trades"] = seed_trades

            if accumulated["watchlist_updates"] and not response.get("watchlist_updates"):
                response["watchlist_updates"] = accumulated["watchlist_updates"]
            if accumulated["memory_notes"] and not response.get("memory_notes"):
                response["memory_notes"] = accumulated["memory_notes"]

            # Carry forward seed fields if investigation didn't produce them
            if not response.get("regime_assessment"):
                response["regime_assessment"] = seed_regime
            if not response.get("reasoning"):
                response["reasoning"] = seed_reasoning
            if not response.get("risk_notes"):
                response["risk_notes"] = seed_risk

            # Preserve token info from seed for decision logging
            if "_tokens_used" not in response:
                response["_tokens_used"] = seed_response.get("_tokens_used", 0)
            if "_model" not in response:
                response["_model"] = seed_response.get("_model", "")

            tool_summary = {
                "tools_used": len(result.tool_calls),
                "turns": result.turns,
                "cost_usd": round(result.token_tracker.total_cost_usd, 4),
                "trailing_stop_requests": accumulated.get("trailing_stop_requests", []),
                "declared_posture": accumulated.get("declared_posture"),
            }

            # Save tool call logs (capture IDs for influenced_decision tracking)
            saved_log_ids: list[int] = []
            try:
                tool_log_entries = [
                    {
                        "turn": tc.turn,
                        "tool_name": tc.tool_name,
                        "tool_input": tc.tool_input,
                        "tool_output_preview": tc.tool_output_preview,
                        "success": tc.success,
                        "error": tc.error,
                    }
                    for tc in result.tool_calls
                ]
                saved_log_ids = await self._db.save_tool_call_logs(
                    tool_log_entries, today, session_label=session, tenant_id=tenant_id
                )
            except Exception as e:
                log.warning("tool_call_log_save_failed", error=str(e))

            log.info(
                "agent_loop_complete",
                turns=result.turns,
                tool_calls=len(result.tool_calls),
                cost_usd=tool_summary["cost_usd"],
            )
        else:
            # ── Single-shot path (existing) ──────────────────────────────
            context = self._strategy_b.prepare_context(
                closes=closes,
                volumes=volumes,
                positions=positions_for_agent,
                cash=cash,
                total_value=total_value,
                recent_trades=recent_trades,
                regime=regime_str,
                yield_curve=yield_curve,
                vix=vix,
                news_context=news_context,
                system_prompt=dynamic_prompt,
                universe=portfolio_b_universe,
            )
            context["model_override"] = model_override

            response = self._strategy_b._agent.analyze(**context)

        # Convert to trades (include dynamic tickers as valid)
        dynamic_tickers = await self._ticker_discovery.get_active_tickers(
            tenant_id=tenant_id,
        )
        trades = self._strategy_b.agent_response_to_trades(
            response=response,
            total_value=total_value,
            current_positions=position_map,
            latest_prices=closes.iloc[-1],
            extra_tickers=dynamic_tickers,
            universe=portfolio_b_universe,
        )

        # Save decision
        await self._strategy_b.save_decision(
            self._db,
            today,
            response,
            trades,
            tenant_id=tenant_id,
            regime=regime_str,
            session_label=session,
        )

        # Mark influential tool calls (agentic path only)
        if use_agent_loop and saved_log_ids:
            try:
                traded_tickers = {t.ticker for t in trades}
                always_influential = {
                    # Phase 2 tools
                    "execute_trade",
                    "set_trailing_stop",
                    "save_observation",
                    "get_portfolio_state",
                    "get_market_overview",
                    # Legacy aliases
                    "propose_trades",
                    "update_watchlist",
                    "save_memory_note",
                    "get_current_positions",
                    "get_portfolio_summary",
                    "get_market_context",
                }
                influenced_ids: list[int] = []
                for idx, tc in enumerate(result.tool_calls):
                    if idx >= len(saved_log_ids):
                        break
                    if tc.tool_name in always_influential:
                        influenced_ids.append(saved_log_ids[idx])
                    elif traded_tickers:
                        # Check if tool_input mentions any traded ticker
                        input_str = json.dumps(tc.tool_input) if isinstance(tc.tool_input, dict) else str(tc.tool_input)
                        if any(ticker in input_str for ticker in traded_tickers):
                            influenced_ids.append(saved_log_ids[idx])
                if influenced_ids:
                    await self._db.update_tool_call_influenced(influenced_ids)
                    log.info("tool_calls_marked_influential", count=len(influenced_ids))
            except Exception as e:
                log.warning("influenced_decision_marking_failed", error=str(e))

        # Save agent memory (short-term + notes)
        try:
            await self._memory_manager.save_short_term(
                self._db,
                today.isoformat(),
                response,
                tenant_id=tenant_id,
            )
            await self._memory_manager.save_agent_notes(
                self._db,
                response.get("memory_notes", []),
                tenant_id=tenant_id,
            )
        except Exception as e:
            log.warning("memory_save_failed", error=str(e))

        # Process suggested tickers from agent response (passive path)
        await self._process_suggested_tickers(response, today, tenant_id=tenant_id)

        # Process tool-based ticker discoveries (already validated + saved by tool)
        try:
            await self._process_tool_discoveries(accumulated, today, tenant_id=tenant_id)
        except Exception as e:
            log.warning("tool_discovery_processing_failed", error=str(e))

        # Process watchlist updates from agent response
        try:
            await self._process_watchlist_updates(
                response.get("watchlist_updates", []),
                tenant_id,
                today,
            )
        except Exception as e:
            log.warning("watchlist_update_failed", error=str(e))

        log.info(
            "portfolio_b_complete",
            trades=len(trades),
            reasoning=response.get("reasoning", "")[:100],
            tokens=response.get("_tokens_used", 0),
            model_override=model_override,
            complexity_score=complexity.score,
            agentic=use_agent_loop,
        )
        return trades, response.get("reasoning", ""), tool_summary

    async def _run_portfolio_b_persistent(
        self,
        tenant_id: str,
        trigger_type: str,
        dynamic_prompt: str,
        closes,
        volumes,
        vix: float | None,
        yield_curve: float | None,
        regime_str: str | None,
        news_context: str,
        positions_for_agent: list[dict],
        cash: float,
        total_value: float,
        recent_trades: list[dict],
        model_override: str | None,
        portfolio_b_universe: list[str] | None,
        allocations: TenantAllocations | None,
        today: date | None,
        session: str,
        strategy_mode: str | None,
    ):
        """Run Portfolio B through the persistent agent path.

        Wraps the existing AgentRunner with conversation persistence.
        Returns the same 3-tuple as _run_portfolio_b: (trades, reasoning, tool_summary).
        """
        from src.agent.agent_runner import AgentRunner
        from src.agent.persistent_agent import PersistentAgent
        from src.agent.tools.actions import ActionState, register_action_tools
        from src.agent.tools.market import register_market_tools
        from src.agent.tools.news import register_news_tools
        from src.agent.tools.portfolio import register_portfolio_tools

        log.info("portfolio_b_persistent_mode", tenant_id=tenant_id, trigger=trigger_type)
        try:
            from src.events.event_bus import Event, EventType, event_bus

            event_bus.publish(
                Event(
                    type=EventType.SESSION_STARTED,
                    tenant_id=tenant_id,
                    data={"trigger": trigger_type, "session": session},
                )
            )
        except Exception as exc:
            log.debug("event_publish_failed", error=str(exc))

        # Build runner (same as agentic path)
        runner = AgentRunner(
            api_key=settings.anthropic_api_key,
            model=settings.agent.agent_tool_model,
            max_turns=settings.agent.agent_max_turns,
            max_cost_usd=settings.agent.agent_session_budget,
        )

        # Register tools
        current_prices = {t: float(closes[t].iloc[-1]) for t in closes.columns if not pd.isna(closes[t].iloc[-1])}
        action_state = ActionState()
        register_portfolio_tools(runner.registry, self._db, tenant_id, current_prices, closes=closes)
        held_tickers = [p["ticker"] for p in positions_for_agent] if positions_for_agent else []
        register_market_tools(
            runner.registry,
            closes,
            vix=vix,
            yield_curve=yield_curve,
            regime=regime_str,
            db=self._db,
            held_tickers=held_tickers,
            tenant_id=tenant_id,
        )
        register_news_tools(
            runner.registry,
            news_context,
            news_fetcher=getattr(self, "_news_fetcher", None),
            db=self._db,
            tenant_id=tenant_id,
            current_prices=current_prices,
        )
        register_action_tools(
            runner.registry,
            action_state,
            db=self._db,
            tenant_id=tenant_id,
            ticker_discovery=self._ticker_discovery,
        )

        # Build market data for trigger message
        market_data = {
            "regime": regime_str or "unknown",
            "vix": vix,
            "spy_change_pct": (
                round(((float(closes["SPY"].iloc[-1]) / float(closes["SPY"].iloc[-2])) - 1) * 100, 2)
                if "SPY" in closes.columns and len(closes["SPY"].dropna()) >= 2
                else None
            ),
        }
        portfolio_data = {
            "total_value": total_value,
            "cash": cash,
            "positions_count": len(positions_for_agent),
        }

        # Build pinned context from DB (posture, playbook, calibration)
        pinned_context = ""
        try:
            from src.agent.context_manager import ContextManager
            from src.analysis.conviction_calibrator import ConvictionBucket, ConvictionCalibrator
            from src.analysis.playbook_generator import PlaybookCell, PlaybookGenerator

            ctx_mgr = ContextManager()

            # Current posture
            posture_row = await self._db.get_current_posture(tenant_id)
            current_posture = posture_row.effective_posture if posture_row else "balanced"

            # Track record
            tr_summary = ""
            try:
                from src.analysis.outcome_tracker import OutcomeTracker
                from src.analysis.track_record import TrackRecord

                tracker = OutcomeTracker(self._db)
                outcomes = await tracker.get_recent_outcomes(days=30, tenant_id=tenant_id)
                if outcomes:
                    stats = TrackRecord().compute(outcomes, min_trades=1)
                    tr_summary = TrackRecord.format_for_prompt(stats)
            except Exception as e:
                log.warning("pinned_context_track_record_failed", error=str(e))

            pinned_context = ctx_mgr.build_pinned_context(
                current_posture=current_posture.capitalize(),
                track_record_summary=tr_summary,
            )

            # Append playbook
            playbook_rows = await self._db.get_latest_playbook(tenant_id)
            if playbook_rows:
                cells = [
                    PlaybookCell(
                        regime=r.regime,
                        sector=r.sector,
                        total=r.total_trades,
                        wins=r.wins,
                        losses=r.losses,
                        win_rate_pct=r.win_rate_pct,
                        avg_pnl_pct=r.avg_pnl_pct,
                        recommendation=r.recommendation,
                    )
                    for r in playbook_rows
                ]
                playbook_text = PlaybookGenerator().format_for_prompt(cells)
                if playbook_text:
                    pinned_context += f"\n\n{playbook_text}"

            # Append calibration
            cal_rows = await self._db.get_latest_calibration(tenant_id)
            if cal_rows:
                buckets = [
                    ConvictionBucket(
                        conviction=r.conviction_level,
                        total=r.total_trades,
                        wins=r.wins,
                        losses=r.losses,
                        win_rate_pct=r.win_rate_pct,
                        avg_pnl_pct=r.avg_pnl_pct,
                        assessment=r.assessment,
                        suggested_multiplier=r.suggested_multiplier,
                    )
                    for r in cal_rows
                ]
                cal_text = ConvictionCalibrator().format_for_prompt(buckets)
                if cal_text:
                    pinned_context += f"\n\n{cal_text}"
        except Exception as e:
            log.warning("pinned_context_build_failed", error=str(e))

        # ── Tiered model + budget integration ─────────────────────────────
        extra_runner_kwargs: dict = {}
        budget_tracker = None
        session_profile_str: str | None = None

        tenant_row = await self._db.get_tenant(tenant_id)
        use_tiered = (
            (settings.agent.enable_tiered and getattr(tenant_row, "use_tiered_models", False)) if tenant_row else False
        )

        if use_tiered:
            from src.agent.budget_tracker import BudgetTracker
            from src.agent.haiku_scanner import HaikuScanner
            from src.agent.opus_validator import OpusValidator
            from src.agent.session_profiles import get_session_profile
            from src.agent.tiered_runner import TieredModelRunner

            budget_tracker = BudgetTracker(
                self._db,
                daily_limit=settings.agent.daily_budget,
                monthly_limit=settings.agent.monthly_budget,
            )
            budget_status = await budget_tracker.check_budget(tenant_id, today)

            if budget_status.daily_exhausted:
                log.warning("daily_budget_exhausted", tenant_id=tenant_id, spent=budget_status.daily_spent)
                try:
                    from src.events.event_bus import Event, EventType, event_bus

                    event_bus.publish(
                        Event(
                            type=EventType.SESSION_SKIPPED,
                            tenant_id=tenant_id,
                            data={"reason": "daily_budget_exhausted", "spent": budget_status.daily_spent},
                        )
                    )
                except Exception as exc:
                    log.debug("event_publish_failed", error=str(exc))
                return [], "Daily budget exhausted — session skipped", None

            session_profile = get_session_profile(trigger_type, budget_exhausted=budget_status.monthly_exhausted)
            session_profile_str = session_profile.value

            scanner = HaikuScanner(api_key=settings.anthropic_api_key)
            validator = OpusValidator(api_key=settings.anthropic_api_key)
            tiered_runner = TieredModelRunner(
                scanner=scanner,
                validator=validator,
                agent_runner=runner,
                token_tracker=runner._token_tracker,
            )
            extra_runner_kwargs["tiered_runner"] = tiered_runner
            extra_runner_kwargs["session_profile"] = session_profile
            extra_runner_kwargs["market_data"] = market_data
            extra_runner_kwargs["portfolio_summary"] = portfolio_data
            try:
                _posture_val = posture_row.effective_posture if posture_row else "balanced"
            except NameError:
                _posture_val = "balanced"
            extra_runner_kwargs["posture"] = _posture_val

        # Build system prompt (cached if enabled and tiered)
        if use_tiered and settings.agent.enable_cache:
            from src.agent.context_manager import ContextManager as CtxCached

            cached_prompt = CtxCached().build_cached_system_prompt(
                pinned_context=pinned_context,
                strategy_directive="",
            )
            extra_runner_kwargs["cached_system_prompt"] = cached_prompt

        # Run persistent session
        persistent = PersistentAgent(
            db=self._db,
            api_key=settings.anthropic_api_key,
            tenant_id=tenant_id,
        )
        result = await persistent.run_session(
            trigger_type=trigger_type,
            market_data=market_data,
            portfolio_summary=portfolio_data,
            runner_kwargs={
                "runner": runner,
                "system_prompt": dynamic_prompt,
                **extra_runner_kwargs,
            },
            pinned_context=pinned_context,
            strategy_directive="",
        )

        response = result.response

        # Merge action state (same as agentic path)
        accumulated = action_state.get_accumulated_state()
        if accumulated["trades"] and not response.get("trades"):
            response["trades"] = accumulated["trades"]
        if accumulated["watchlist_updates"] and not response.get("watchlist_updates"):
            response["watchlist_updates"] = accumulated["watchlist_updates"]
        if accumulated["memory_notes"] and not response.get("memory_notes"):
            response["memory_notes"] = accumulated["memory_notes"]

        # Attach agent's trailing stop requests + posture to tool_summary for Step 7.1
        persistent_tool_summary = result.tool_summary or {}
        if accumulated.get("trailing_stop_requests"):
            persistent_tool_summary["trailing_stop_requests"] = accumulated["trailing_stop_requests"]
        if accumulated.get("declared_posture"):
            persistent_tool_summary["declared_posture"] = accumulated["declared_posture"]

        # Convert to trades
        position_map = {p["ticker"]: p["shares"] for p in positions_for_agent}
        dynamic_tickers = await self._ticker_discovery.get_active_tickers(tenant_id=tenant_id)
        trades = self._strategy_b.agent_response_to_trades(
            response=response,
            total_value=total_value,
            current_positions=position_map,
            latest_prices=closes.iloc[-1],
            extra_tickers=dynamic_tickers,
            universe=portfolio_b_universe,
        )

        # Save decision
        await self._strategy_b.save_decision(
            self._db,
            today,
            response,
            trades,
            tenant_id=tenant_id,
            regime=regime_str,
            session_label=session,
        )

        # Save tool call logs
        if result.tool_calls:
            try:
                tool_log_entries = [
                    {
                        "turn": tc.turn,
                        "tool_name": tc.tool_name,
                        "tool_input": tc.tool_input,
                        "tool_output_preview": tc.tool_output_preview,
                        "success": tc.success,
                        "error": tc.error,
                    }
                    for tc in result.tool_calls
                ]
                await self._db.save_tool_call_logs(tool_log_entries, today, session_label=session, tenant_id=tenant_id)
            except Exception as e:
                log.warning("tool_call_log_save_failed", error=str(e))

        # Save agent memory
        try:
            await self._memory_manager.save_short_term(
                self._db,
                today.isoformat() if today else "",
                response,
                tenant_id=tenant_id,
            )
            await self._memory_manager.save_agent_notes(
                self._db,
                response.get("memory_notes", []),
                tenant_id=tenant_id,
            )
        except Exception as e:
            log.warning("memory_save_failed", error=str(e))

        # Process suggested tickers (passive path)
        await self._process_suggested_tickers(response, today, tenant_id=tenant_id)

        # Process tool-based ticker discoveries (already validated + saved by tool)
        try:
            await self._process_tool_discoveries(accumulated, today, tenant_id=tenant_id)
        except Exception as e:
            log.warning("tool_discovery_processing_failed", error=str(e))

        # Process watchlist updates
        try:
            await self._process_watchlist_updates(
                response.get("watchlist_updates", []),
                tenant_id,
                today,
            )
        except Exception as e:
            log.warning("watchlist_update_failed", error=str(e))

        # Record budget spend (if tiered mode active)
        if budget_tracker is not None:
            try:
                await budget_tracker.record_session(
                    tenant_id=tenant_id,
                    session_date=today or date.today(),
                    session_label=trigger_type,
                    session_id=result.session_id,
                    token_tracker=result.token_tracker,
                    session_profile=session_profile_str,
                )
            except Exception as e:
                log.warning("budget_record_failed", error=str(e))

        # Add tiered info to tool summary for Telegram
        if session_profile_str:
            persistent_tool_summary["session_profile"] = session_profile_str

        log.info(
            "portfolio_b_persistent_complete",
            trades=len(trades),
            session_id=result.session_id,
            compressed=result.compressed_count,
            tokens=result.token_tracker.total_input_tokens + result.token_tracker.total_output_tokens,
            cost_usd=round(result.token_tracker.total_cost_usd, 4),
        )
        try:
            from src.events.event_bus import Event, EventType, event_bus

            event_bus.publish(
                Event(
                    type=EventType.SESSION_COMPLETED,
                    tenant_id=tenant_id,
                    data={
                        "trades": len(trades),
                        "cost_usd": round(result.token_tracker.total_cost_usd, 4),
                    },
                )
            )
        except Exception as exc:
            log.debug("event_publish_failed", error=str(exc))
        return trades, response.get("reasoning", ""), persistent_tool_summary

    async def _check_trailing_stops(
        self,
        tenant_id: str,
        closes: pd.DataFrame,
        run_portfolio_a: bool,
        run_portfolio_b: bool,
    ) -> tuple[list[TradeSchema], list[dict]]:
        """Check trailing stops, update peaks, and generate sells for triggered stops.

        Returns:
            Tuple of (sell trades list, alert dicts for notifications).
        """
        sells: list[TradeSchema] = []
        alerts: list[dict] = []

        active_portfolios = _active_portfolio_names(
            run_portfolio_a,
            run_portfolio_b,
            tenant_id,
        )
        stops = await self._db.get_active_trailing_stops(tenant_id)

        for stop in stops:
            if stop.portfolio not in active_portfolios:
                continue
            if stop.ticker not in closes.columns:
                continue

            price = float(closes[stop.ticker].iloc[-1])
            if pd.isna(price):
                continue

            # Update peak if price is higher
            if price > stop.peak_price:
                new_stop_price = price * (1 - stop.trail_pct)
                await self._db.update_trailing_stop(
                    stop.id,
                    peak_price=price,
                    stop_price=new_stop_price,
                )
                log.debug(
                    "trailing_stop_peak_updated",
                    ticker=stop.ticker,
                    new_peak=round(price, 2),
                    new_stop=round(new_stop_price, 2),
                )
            elif price <= stop.stop_price:
                # TRIGGERED — generate full sell
                positions = await self._db.get_positions(
                    stop.portfolio,
                    tenant_id=tenant_id,
                )
                pos = next((p for p in positions if p.ticker == stop.ticker), None)
                if pos and pos.shares > 0:
                    sells.append(
                        TradeSchema(
                            portfolio=PortfolioName(stop.portfolio),
                            ticker=stop.ticker,
                            side=OrderSide.SELL,
                            shares=pos.shares,
                            price=price,
                            reason=f"Trailing stop triggered (stop=${stop.stop_price:.2f})",
                        )
                    )
                    alerts.append(
                        {
                            "ticker": stop.ticker,
                            "price": price,
                            "entry": stop.entry_price,
                            "peak": stop.peak_price,
                        }
                    )
                    await self._db.deactivate_trailing_stop(stop.id)
                    log.info(
                        "trailing_stop_triggered",
                        ticker=stop.ticker,
                        price=round(price, 2),
                        stop_price=round(stop.stop_price, 2),
                        portfolio=stop.portfolio,
                    )
                    try:
                        from src.events.event_bus import Event, EventType, event_bus

                        event_bus.publish(
                            Event(
                                type=EventType.TRAILING_STOP_TRIGGERED,
                                tenant_id=tenant_id,
                                data={
                                    "ticker": stop.ticker,
                                    "price": round(price, 2),
                                    "stop_price": round(stop.stop_price, 2),
                                    "portfolio": stop.portfolio,
                                },
                            )
                        )
                    except Exception as exc:
                        log.debug("event_publish_failed", error=str(exc))

        return sells, alerts

    @staticmethod
    def _format_correlation(corr_data: dict) -> str:
        """Format correlation data for the system prompt."""
        lines = [f"Correlation (avg: {corr_data['avg_correlation']:.2f}, {corr_data['matrix_size']} positions):"]
        for t1, t2, val in corr_data["high_pairs"]:
            lines.append(f"  {t1}-{t2}: {val:.2f} (HIGH)")
        return "\n".join(lines)

    @staticmethod
    def tenant_fully_configured(tenant: TenantRow) -> bool:
        """Check if a tenant has all required credentials to run the bot."""
        return bool(
            tenant.alpaca_api_key_enc
            and tenant.alpaca_api_secret_enc
            and tenant.telegram_bot_token_enc
            and tenant.telegram_chat_id_enc
        )

    def _notifier_available(self) -> bool:
        """Check if Telegram notifier is configured."""
        return bool(self._notifier._token and self._notifier._chat_id)

    @staticmethod
    async def _capture_alpaca_equity(executor) -> float | None:
        """Fetch account equity from Alpaca executor.

        Returns:
            Equity as float, or None if the executor is not Alpaca-based.
        """
        if not hasattr(executor, "_client"):
            return None
        try:
            account = await asyncio.to_thread(executor._client.get_account)
            return float(account.equity)
        except Exception as e:
            log.warning("equity_capture_failed", error=str(e))
            return None

    async def _detect_deposits(
        self,
        allocations: TenantAllocations,
        tenant_id: str,
    ) -> TenantAllocations:
        """Detect real cash deposits via Alpaca account activities API.

        Queries /v2/account/activities for CSD (Cash Deposit) and JNLC
        (Journal Credit) entries in the last 5 days.  Only confirmed broker
        transfers update the baseline — this prevents false positives from
        overnight price movements that inflate broker equity.

        A secondary equity-gap check guards against double-counting: after
        the snapshot step syncs tracked totals to broker equity, the gap
        drops to ~0 on subsequent runs in the same day.

        Returns:
            Updated TenantAllocations (or the original if no deposit).
        """
        if not hasattr(self._executor, "_client"):
            return allocations

        # --- Step 1: query Alpaca for actual cash-deposit activities ----
        try:
            after = (date.today() - timedelta(days=5)).isoformat()
            raw = await asyncio.to_thread(
                self._executor._client.get,
                "/v2/account/activities",
                {"activity_types": "CSD,JNLC", "after": after},
            )
        except Exception as e:
            log.warning("deposit_detection_fetch_failed", error=str(e))
            return allocations

        if not raw:
            return allocations

        activities = raw if isinstance(raw, list) else [raw]

        deposit_sum = 0.0
        for act in activities:
            try:
                amount = float(act.get("net_amount", 0))
                if amount > 0:
                    deposit_sum += amount
            except (ValueError, TypeError, AttributeError):
                continue

        if deposit_sum <= DEPOSIT_THRESHOLD:
            return allocations

        # --- Step 2: equity-gap guard against double-counting -----------
        try:
            account = await asyncio.to_thread(
                self._executor._client.get_account,
            )
            broker_equity = float(account.equity)
        except Exception as e:
            log.warning("deposit_detection_equity_fetch_failed", error=str(e))
            return allocations

        tracked_total = 0.0
        for pname in ("A", "B"):
            portfolio = await self._db.get_portfolio(pname, tenant_id=tenant_id)
            if portfolio:
                tracked_total += portfolio.total_value

        equity_gap = broker_equity - tracked_total
        if equity_gap <= DEPOSIT_THRESHOLD:
            # Tracked totals already include the deposit (processed earlier).
            return allocations

        # --- Step 3: confirmed deposit — apply to baseline ---------------
        delta = deposit_sum

        new_equity = allocations.initial_equity + delta
        new_a_cash = new_equity * allocations.portfolio_a_pct / 100
        new_b_cash = new_equity * allocations.portfolio_b_pct / 100

        # Distribute deposit cash into portfolios
        a_deposit = delta * allocations.portfolio_a_pct / 100
        b_deposit = delta * allocations.portfolio_b_pct / 100

        for pname, deposit_amount in [("A", a_deposit), ("B", b_deposit)]:
            portfolio = await self._db.get_portfolio(pname, tenant_id=tenant_id)
            if portfolio:
                await self._db.upsert_portfolio(
                    pname,
                    cash=portfolio.cash + deposit_amount,
                    total_value=portfolio.total_value + deposit_amount,
                    tenant_id=tenant_id,
                )

        # Update tenant record
        if tenant_id != "default":
            await self._db.update_tenant(
                tenant_id,
                {
                    "initial_equity": new_equity,
                    "portfolio_a_cash": new_a_cash,
                    "portfolio_b_cash": new_b_cash,
                },
            )

        log.info(
            "deposit_detected",
            delta=round(delta, 2),
            new_equity=round(new_equity, 2),
            tenant_id=tenant_id,
        )

        # Notify via Telegram
        if self._notifier_available():
            try:
                await self._notifier.send_message(
                    f"Deposit detected: +${delta:,.2f}\n"
                    f"New baseline: ${new_equity:,.2f}\n"
                    f"Portfolio A: ${new_a_cash:,.2f} | B: ${new_b_cash:,.2f}"
                )
            except (ConnectionError, TimeoutError, OSError) as e:
                log.warning("deposit_notification_failed", error=str(e))

        from src.utils.allocations import resolve_allocations

        return resolve_allocations(
            initial_equity=new_equity,
            portfolio_a_pct=allocations.portfolio_a_pct,
            portfolio_b_pct=allocations.portfolio_b_pct,
        )

    async def _reconcile_equity(
        self,
        tenant_id: str,
        run_portfolio_a: bool,
        run_portfolio_b: bool,
        allocations: TenantAllocations,
    ) -> float | None:
        """Reconcile internal portfolio cash against Alpaca broker equity.

        Corrects small pricing/cash drift ($10–$50) that accumulates from
        fill price differences, rounding, and dividends. Skips when the
        drift is below RECONCILE_THRESHOLD (noise) or above DEPOSIT_THRESHOLD
        positive (handled by _detect_deposits).

        Returns:
            Drift amount corrected, or None if no action was taken.
        """
        if not hasattr(self._executor, "_client"):
            return None

        try:
            account = await asyncio.to_thread(self._executor._client.get_account)
            broker_equity = float(account.equity)
        except Exception as e:
            log.warning("reconcile_equity_fetch_failed", error=str(e))
            return None

        # Sum tracked totals for enabled portfolios
        tracked_total = 0.0
        enabled = []
        if run_portfolio_a:
            enabled.append("A")
        if run_portfolio_b:
            enabled.append("B")
        if not enabled:
            return None

        for pname in enabled:
            portfolio = await self._db.get_portfolio(pname, tenant_id=tenant_id)
            if portfolio:
                tracked_total += portfolio.total_value

        drift = broker_equity - tracked_total

        if abs(drift) <= RECONCILE_THRESHOLD:
            return None

        # Positive drift above deposit threshold — let _detect_deposits handle it
        if drift > DEPOSIT_THRESHOLD:
            return None

        # Distribute drift proportionally across enabled portfolio cash
        if len(enabled) == 2:
            total_pct = allocations.portfolio_a_pct + allocations.portfolio_b_pct
            splits = {
                "A": allocations.portfolio_a_pct / total_pct,
                "B": allocations.portfolio_b_pct / total_pct,
            }
        else:
            splits = {enabled[0]: 1.0}

        for pname, share in splits.items():
            portfolio = await self._db.get_portfolio(pname, tenant_id=tenant_id)
            if portfolio:
                new_cash = portfolio.cash + drift * share
                new_total = portfolio.total_value + drift * share
                await self._db.upsert_portfolio(
                    pname,
                    cash=new_cash,
                    total_value=new_total,
                    tenant_id=tenant_id,
                )

        log.info(
            "equity_reconciled",
            drift=round(drift, 2),
            tenant_id=tenant_id,
            portfolios=enabled,
        )
        return drift

    async def _handle_rebalance(
        self,
        tenant_id: str,
        closes: pd.DataFrame,
        run_portfolio_a: bool,
        run_portfolio_b: bool,
    ) -> TenantAllocations | None:
        """Handle pending portfolio rebalance: liquidate and redistribute.

        Called in run_daily() after market data fetch. If the tenant has
        pending_rebalance=True, liquidate the appropriate positions and
        redistribute cash according to the new toggle state.

        Returns:
            New TenantAllocations if rebalance occurred, None otherwise.
        """
        tenant = await self._db.get_tenant(tenant_id)
        if tenant is None or not tenant.pending_rebalance:
            return None

        log.info(
            "rebalance_start",
            tenant_id=tenant_id,
            run_a=run_portfolio_a,
            run_b=run_portfolio_b,
        )

        # Determine which portfolios to liquidate
        portfolios_to_liquidate: list[str] = []
        if run_portfolio_a and run_portfolio_b:
            # Both enabled (fresh start) — liquidate everything
            portfolios_to_liquidate = ["A", "B"]
        elif run_portfolio_a and not run_portfolio_b:
            # Only A enabled — liquidate B
            portfolios_to_liquidate = ["B"]
        elif not run_portfolio_a and run_portfolio_b:
            # Only B enabled — liquidate A
            portfolios_to_liquidate = ["A"]
        else:
            # Both disabled — liquidate everything
            portfolios_to_liquidate = ["A", "B"]

        # Build latest prices from closes
        latest_prices: dict[str, float] = {}
        for t in closes.columns:
            if not pd.isna(closes[t].iloc[-1]):
                latest_prices[t] = float(closes[t].iloc[-1])

        # Generate SELL trades for positions in portfolios to liquidate
        from src.storage.models import OrderSide, PortfolioName, TradeSchema

        sell_trades: list[TradeSchema] = []
        for pname in portfolios_to_liquidate:
            positions = await self._db.get_positions(pname, tenant_id=tenant_id)
            for pos in positions:
                if pos.shares <= 0:
                    continue
                price = latest_prices.get(pos.ticker, pos.avg_price)
                sell_trades.append(
                    TradeSchema(
                        portfolio=PortfolioName(pname),
                        ticker=pos.ticker,
                        side=OrderSide.SELL,
                        shares=pos.shares,
                        price=price,
                        reason="Portfolio rebalance: liquidation",
                    )
                )

        # Execute sells
        if sell_trades:
            await self._executor.execute_trades(sell_trades, tenant_id=tenant_id)
            log.info("rebalance_liquidation_complete", trades=len(sell_trades))

        # Calculate total available cash across both portfolios
        total_cash = 0.0
        for pname in ("A", "B"):
            portfolio = await self._db.get_portfolio(pname, tenant_id=tenant_id)
            if portfolio:
                total_cash += portfolio.cash

        # Redistribute cash
        a_pct = tenant.portfolio_a_pct or 33.33
        b_pct = tenant.portfolio_b_pct or 66.67

        if run_portfolio_a and run_portfolio_b:
            a_cash = total_cash * a_pct / 100
            b_cash = total_cash * b_pct / 100
        elif run_portfolio_a:
            a_cash = total_cash
            b_cash = 0.0
        elif run_portfolio_b:
            a_cash = 0.0
            b_cash = total_cash
        else:
            a_cash = 0.0
            b_cash = 0.0

        # Update portfolio rows
        await self._db.upsert_portfolio("A", cash=a_cash, total_value=a_cash, tenant_id=tenant_id)
        await self._db.upsert_portfolio("B", cash=b_cash, total_value=b_cash, tenant_id=tenant_id)

        # Update tenant record: clear flag + update cash + initial_equity
        await self._db.update_tenant(
            tenant_id,
            {
                "pending_rebalance": False,
                "portfolio_a_cash": a_cash,
                "portfolio_b_cash": b_cash,
                "initial_equity": total_cash if total_cash > 0 else tenant.initial_equity,
            },
        )

        # Notify via Telegram
        if self._notifier_available():
            try:
                liquidated_str = ", ".join(portfolios_to_liquidate)
                await self._notifier.send_message(
                    f"Portfolio rebalance complete.\n"
                    f"Liquidated: Portfolio {liquidated_str} "
                    f"({len(sell_trades)} trades)\n"
                    f"New allocation: A=${a_cash:,.0f} | B=${b_cash:,.0f}\n"
                    f"Total: ${total_cash:,.0f}"
                )
            except (ConnectionError, TimeoutError, OSError) as e:
                log.warning("rebalance_notification_failed", error=str(e))

        log.info(
            "rebalance_complete",
            tenant_id=tenant_id,
            liquidated=portfolios_to_liquidate,
            sell_trades=len(sell_trades),
            a_cash=round(a_cash, 2),
            b_cash=round(b_cash, 2),
            total=round(total_cash, 2),
        )

        from src.utils.allocations import resolve_allocations

        return resolve_allocations(
            initial_equity=total_cash if total_cash > 0 else (tenant.initial_equity or 0),
            portfolio_a_pct=a_pct,
            portfolio_b_pct=b_pct,
        )

    async def _request_model_approval(self, complexity) -> str:
        """Send approval request via Telegram and wait for response.

        Returns:
            "opus", "sonnet", or "skip".
        """
        request_id = uuid.uuid4().hex
        msg_id = await self._notifier.send_approval_request(complexity, request_id)
        if msg_id is None:
            return "sonnet"
        return await self._notifier.wait_for_approval(request_id, PORTFOLIO_B.approval_timeout_seconds)

    async def _request_inverse_trade_approval(
        self,
        trade: TradeSchema,
        regime_result: RegimeResult | None,
    ) -> str:
        """Send inverse trade approval request via Telegram.

        Args:
            trade: The inverse ETF trade requiring approval.
            regime_result: Current regime result.

        Returns:
            "approve" or "reject". Defaults to "reject" if no notifier.
        """
        if not self._notifier_available():
            log.info("inverse_approval_auto_reject_no_notifier", ticker=trade.ticker)
            return "reject"

        request_id = uuid.uuid4().hex
        regime_str = regime_result.regime.value if regime_result else None
        msg_id = await self._notifier.send_inverse_trade_approval(trade, regime_str, request_id)
        if msg_id is None:
            return "reject"
        return await self._notifier.wait_for_inverse_approval(request_id, timeout_seconds=300)

    async def _request_large_trade_approval(
        self,
        trade: "TradeSchema",
        trade_pct: float,
        approval_reason: str,
        tenant_id: str,
    ) -> str:
        """Send large trade approval request via Telegram and publish SSE events.

        Args:
            trade: The trade requiring approval (>threshold% of portfolio).
            trade_pct: Trade value as % of portfolio.
            approval_reason: Human-readable reason for the approval.
            tenant_id: Tenant UUID.

        Returns:
            "approve" or "reject". Defaults to "reject" if no notifier.
        """
        # Publish SSE: approval requested
        try:
            from src.events.event_bus import Event, EventType, event_bus

            event_bus.publish(
                Event(
                    type=EventType.TRADE_APPROVAL_REQUESTED,
                    tenant_id=tenant_id,
                    data={
                        "ticker": trade.ticker,
                        "side": trade.side.value,
                        "shares": trade.shares,
                        "price": trade.price,
                        "value": round(trade.total, 2),
                        "portfolio_pct": round(trade_pct, 1),
                        "reason": approval_reason,
                    },
                )
            )
        except Exception as exc:
            log.debug("event_publish_failed", error=str(exc))

        if not self._notifier_available():
            log.info("large_trade_auto_reject_no_notifier", ticker=trade.ticker)
            self._publish_trade_approval_resolved(tenant_id, trade.ticker, False)
            return "reject"

        request_id = uuid.uuid4().hex
        msg_id = await self._notifier.send_large_trade_approval(trade, trade_pct, approval_reason, request_id)
        if msg_id is None:
            self._publish_trade_approval_resolved(tenant_id, trade.ticker, False)
            return "reject"
        choice = await self._notifier.wait_for_large_trade_approval(
            request_id, timeout_seconds=settings.trade_approval_timeout_s
        )
        self._publish_trade_approval_resolved(tenant_id, trade.ticker, choice == "approve")
        return choice

    def _publish_trade_approval_resolved(self, tenant_id: str, ticker: str, approved: bool) -> None:
        """Publish SSE event for trade approval resolution."""
        try:
            from src.events.event_bus import Event, EventType, event_bus

            event_bus.publish(
                Event(
                    type=EventType.TRADE_APPROVAL_RESOLVED,
                    tenant_id=tenant_id,
                    data={"ticker": ticker, "approved": approved},
                )
            )
        except Exception as exc:
            log.debug("event_publish_failed", error=str(exc))

    async def _process_suggested_tickers(
        self,
        response: dict,
        today: date,
        tenant_id: str = "default",
    ) -> None:
        """Validate agent-suggested tickers and send for Telegram approval.

        Args:
            response: Agent response dict potentially containing suggested_tickers.
            today: Current date.
            tenant_id: Tenant UUID for scoping.
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
                ticker=ticker,
                rationale=rationale,
                source="agent",
                today=today,
                tenant_id=tenant_id,
            )
            if row is None:
                continue

            # Send Telegram approval if available
            if self._notifier_available():
                choice = await self._request_ticker_approval(row)
                if choice == "approve":
                    await self._db.update_discovered_ticker_status(
                        ticker,
                        "approved",
                        tenant_id=tenant_id,
                    )
                    log.info("ticker_approved", ticker=ticker, tenant_id=tenant_id)
                else:
                    await self._db.update_discovered_ticker_status(
                        ticker,
                        "rejected",
                        tenant_id=tenant_id,
                    )
                    log.info("ticker_rejected", ticker=ticker, tenant_id=tenant_id)
            else:
                # No Telegram — auto-reject (requires human approval)
                await self._db.update_discovered_ticker_status(
                    ticker,
                    "rejected",
                    tenant_id=tenant_id,
                )
                log.info("ticker_auto_rejected_no_telegram", ticker=ticker)

    async def _process_tool_discoveries(
        self,
        accumulated: dict,
        today: date,
        tenant_id: str = "default",
    ) -> None:
        """Process ticker discoveries made via the discover_ticker tool.

        Tool-discovered tickers are already validated and saved to DB as 'proposed'.
        This method sends Telegram approval for each. If no Telegram, they stay
        as 'proposed' for dashboard approval.
        """
        proposals = accumulated.get("discovery_proposals", [])
        if not proposals:
            return

        for proposal in proposals:
            ticker = proposal.get("ticker", "").upper().strip()
            if not ticker:
                continue

            if self._notifier_available():
                row = await self._db.get_discovered_ticker(ticker, tenant_id=tenant_id)
                if row and row.status == "proposed":
                    choice = await self._request_ticker_approval(row)
                    status = "approved" if choice == "approve" else "rejected"
                    await self._db.update_discovered_ticker_status(ticker, status, tenant_id=tenant_id)
                    log.info("tool_discovery_resolved", ticker=ticker, status=status, tenant_id=tenant_id)
            else:
                # No Telegram — leave as 'proposed' for dashboard approval
                log.info("tool_discovery_pending_dashboard", ticker=ticker, tenant_id=tenant_id)

    async def _process_watchlist_updates(
        self,
        updates: list[dict],
        tenant_id: str,
        today: date,
    ) -> None:
        """Process watchlist add/remove actions from the agent response.

        Args:
            updates: List of watchlist update dicts from agent.
            tenant_id: Tenant UUID.
            today: Current date (for expiry calculation).
        """
        if not updates:
            return

        for update in updates:
            action = update.get("action", "").lower()
            ticker = update.get("ticker", "").upper().strip()
            if not ticker:
                continue

            if action == "add":
                await self._db.upsert_watchlist_item(
                    tenant_id=tenant_id,
                    ticker=ticker,
                    reason=update.get("reason", ""),
                    conviction=update.get("conviction", "medium"),
                    target_entry=update.get("target_entry"),
                    expires_at=today + timedelta(days=14),
                )
                log.info("watchlist_item_added", ticker=ticker, tenant_id=tenant_id)
            elif action == "remove":
                await self._db.remove_watchlist_item(tenant_id, ticker)
                log.info("watchlist_item_removed", ticker=ticker, tenant_id=tenant_id)

        # Publish SSE event for watchlist changes
        try:
            from src.events.event_bus import Event, EventType, event_bus

            additions = sum(1 for u in updates if u.get("action", "").lower() == "add")
            removals = sum(1 for u in updates if u.get("action", "").lower() == "remove")
            event_bus.publish(
                Event(
                    type=EventType.WATCHLIST_UPDATED,
                    tenant_id=tenant_id,
                    data={"additions": additions, "removals": removals},
                )
            )
        except Exception as exc:
            log.debug("event_publish_failed", error=str(exc))

    async def _request_ticker_approval(self, row) -> str:
        """Send ticker approval request via Telegram and wait for response.

        Returns:
            "approve" or "reject".
        """
        request_id = uuid.uuid4().hex
        msg_id = await self._notifier.send_ticker_proposal(row, request_id)
        if msg_id is None:
            return "reject"
        return await self._notifier.wait_for_ticker_approval(request_id, PORTFOLIO_B.approval_timeout_seconds)

    async def recovery_check(
        self,
        today: date,
        closes: pd.DataFrame,
        tenant_id: str = "default",
        allocations: TenantAllocations | None = None,
        run_portfolio_a: bool = True,
        run_portfolio_b: bool = True,
    ) -> list[str]:
        """Detect missed trading days and backfill snapshots.

        Compares DB snapshots against the business day calendar to find
        gaps. For each missed day, creates a snapshot using the closes
        data for that date.

        Args:
            today: Current trading date.
            closes: Full historical closes DataFrame.
            tenant_id: Tenant UUID for data isolation.
            allocations: Tenant allocations for initial value reference.
            run_portfolio_a: Whether Portfolio A is enabled.
            run_portfolio_b: Whether Portfolio B is enabled.

        Returns:
            List of recovered date strings (ISO format).
        """
        alloc = allocations or DEFAULT_ALLOCATIONS

        recovered_dates: list[str] = []

        recovery_portfolios = _active_portfolio_names(
            run_portfolio_a,
            run_portfolio_b,
            tenant_id,
        )
        all_portfolio_initials = {
            "A": alloc.portfolio_a_cash,
            "B": alloc.portfolio_b_cash,
        }
        for pname in recovery_portfolios:
            initial = all_portfolio_initials[pname]
            snapshots = await self._db.get_snapshots(pname, tenant_id=tenant_id)
            if not snapshots:
                continue

            snapshot_dates = {s.date for s in snapshots}
            last_snap_date = max(snapshot_dates)

            # Build expected trading days between last snapshot and today
            expected = trading_days_between(last_snap_date, today)

            missed = [d for d in expected if d not in snapshot_dates and d in closes.index.date]

            if not missed:
                continue

            log.warning(
                "missed_days_detected",
                portfolio=pname,
                missed=len(missed),
                dates=[str(d) for d in missed],
            )

            # Backfill snapshots
            prev_total = snapshots[-1].total_value
            for miss_date in missed:
                prices = {}
                for t in closes.columns:
                    mask = closes.index.date == miss_date
                    if mask.any():
                        val = closes.loc[mask, t].iloc[-1]
                        if pd.notna(val):
                            prices[t] = float(val)

                if not prices:
                    continue

                # Calculate value from existing positions
                portfolio = await self._db.get_portfolio(
                    pname,
                    tenant_id=tenant_id,
                )
                if portfolio is None:
                    continue

                positions = await self._db.get_positions(
                    pname,
                    tenant_id=tenant_id,
                )
                pos_value = sum(p.shares * prices.get(p.ticker, p.avg_price) for p in positions)
                total_value = portfolio.cash + pos_value

                daily_ret = None
                if prev_total > 0:
                    daily_ret = (total_value - prev_total) / prev_total * 100
                cum_ret = ((total_value - initial) / initial) * 100

                await self._db.save_snapshot(
                    portfolio=pname,
                    snapshot_date=miss_date,
                    total_value=total_value,
                    cash=portfolio.cash,
                    positions_value=pos_value,
                    daily_return_pct=daily_ret,
                    cumulative_return_pct=cum_ret,
                    tenant_id=tenant_id,
                )

                recovered_dates.append(str(miss_date))
                prev_total = total_value

        if recovered_dates:
            log.info(
                "recovery_backfill_complete",
                days=len(recovered_dates),
            )
            # Notify via Telegram
            if self._notifier_available():
                try:
                    msg = (
                        f"Recovery: backfilled {len(recovered_dates)} missed snapshot(s): {', '.join(recovered_dates)}"
                    )
                    await self._notifier.send_message(msg)
                except (ConnectionError, TimeoutError, OSError) as e:
                    log.warning("recovery_notification_failed", error=str(e))

        return recovered_dates

    async def _send_notifications(
        self,
        today: date,
        proposed_trades: list,
        executed_trades: list,
        summary: dict,
        session: str = "",
        regime_result: RegimeResult | None = None,
        tenant_id: str = "default",
        strategy_mode: str | None = None,
        allocations: TenantAllocations | None = None,
        run_portfolio_a: bool = True,
        run_portfolio_b: bool = True,
        trailing_stop_alerts: list[dict] | None = None,
    ) -> None:
        """Send daily brief and trade confirmation via Telegram."""
        alloc = allocations or DEFAULT_ALLOCATIONS
        active_strategy = strategy_mode or settings.agent.strategy_mode
        try:
            # Build portfolio summaries from snapshots (only enabled portfolios)
            portfolio_summaries = {}
            notify_portfolios = _active_portfolio_names(
                run_portfolio_a,
                run_portfolio_b,
                tenant_id,
            )
            for name in notify_portfolios:
                portfolio = await self._db.get_portfolio(name, tenant_id=tenant_id)
                snapshots = await self._db.get_snapshots(name, tenant_id=tenant_id)
                today_snap = next((s for s in snapshots if s.date == today), None)
                default_value = alloc.for_portfolio(name)
                portfolio_summaries[name] = {
                    "total_value": (
                        today_snap.total_value
                        if today_snap
                        else (portfolio.total_value if portfolio else default_value)
                    ),
                    "cash": portfolio.cash if portfolio else default_value,
                    "daily_return_pct": today_snap.daily_return_pct if today_snap else None,
                }

            # Add strategy-specific fields for enabled portfolios
            _empty_summary: dict = {"total_value": 0, "cash": 0, "daily_return_pct": None}
            if "A" in portfolio_summaries:
                positions_a = await self._db.get_positions("A", tenant_id=tenant_id)
                if positions_a:
                    top = max(positions_a, key=lambda p: p.shares * p.avg_price)
                    portfolio_summaries["A"]["top_ticker"] = top.ticker
                else:
                    portfolio_summaries["A"]["top_ticker"] = "cash"
                portfolio_summaries["A"]["reason"] = summary.get("a_reason", "")
            if "B" in portfolio_summaries:
                portfolio_summaries["B"]["reasoning"] = summary.get("b_reasoning", "") or "No changes recommended"

            from src.storage.models import TradeSchema

            proposed = [t for t in proposed_trades if isinstance(t, TradeSchema)]

            await self._notifier.send_daily_brief(
                brief_date=today,
                regime=regime_result.regime.value if regime_result else None,
                portfolio_a=portfolio_summaries.get("A", _empty_summary),
                portfolio_b=portfolio_summaries.get("B", _empty_summary),
                proposed_trades=proposed,
                commentary="",
                session=session,
                strategy_mode=active_strategy,
                run_portfolio_a=run_portfolio_a,
                run_portfolio_b=run_portfolio_b,
                trailing_stop_alerts=trailing_stop_alerts or [],
                agent_tool_summary=summary.get("b_tool_summary"),
            )

            # Only send trade confirmation for actually filled trades
            filled = [t for t in executed_trades if isinstance(t, TradeSchema)]
            if filled:
                await self._notifier.send_trade_confirmation(filled)

            log.info("notifications_sent", proposed=len(proposed), filled=len(filled))
        except Exception as e:
            log.error("notification_failed", error=str(e))
