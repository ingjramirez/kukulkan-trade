"""Kukulkan Trading Bot — main entry point.

Usage:
    python -m src.main              # Start scheduler (3x daily during market hours)
    python -m src.main --run-now    # Run pipeline immediately, then exit

Executor is controlled by EXECUTOR env var: "alpaca" or "paper" (default).
"""

import argparse
import asyncio
import logging
import signal

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config.settings import settings
from src.agent.memory import AgentMemoryManager
from src.notifications.telegram_bot import TelegramNotifier
from src.notifications.weekly_report import WeeklyReporter
from src.orchestrator import Orchestrator
from src.storage.database import Database
from src.utils.market_calendar import is_market_open

log = structlog.get_logger()


def setup_logging() -> None:
    """Configure structlog for the application with optional file rotation."""
    from logging.handlers import RotatingFileHandler

    level = logging.getLevelName(settings.log_level)

    # Console handler (always present)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    handlers: list[logging.Handler] = [console_handler]

    # File handler with rotation (10 MB per file, 5 backups)
    logs_dir = settings.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        logs_dir / "kukulkan.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    handlers.append(file_handler)

    # Configure root logger so structlog output reaches both handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = handlers

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


async def _create_executor(db: Database):
    """Create the appropriate executor based on settings.

    Returns:
        (executor, cleanup_fn) — cleanup_fn is called on shutdown (or None).
    """
    executor_type = settings.executor.lower()

    if executor_type == "alpaca":
        try:
            from alpaca.trading.client import TradingClient

            from src.execution.alpaca_executor import AlpacaExecutor

            client = TradingClient(
                api_key=settings.alpaca.api_key,
                secret_key=settings.alpaca.secret_key,
                paper=settings.alpaca.paper,
            )
            # Verify connection
            account = client.get_account()
            log.info(
                "alpaca_connected",
                status=str(account.status),
                equity=str(account.equity),
                paper=settings.alpaca.paper,
            )
            return AlpacaExecutor(db, client), None
        except Exception as e:
            log.warning("alpaca_not_available_using_paper_trader", error=str(e))

    # Default: PaperTrader
    from src.execution.paper_trader import PaperTrader

    log.info("using_paper_trader")
    return PaperTrader(db), None


async def run_once() -> None:
    """Run the daily pipeline once and exit."""
    db = Database(url=settings.database_url)
    await db.init_db()
    notifier = TelegramNotifier()

    executor, cleanup = await _create_executor(db)

    orchestrator = Orchestrator(db, notifier=notifier, executor=executor)
    results = await orchestrator.run_all_tenants()

    log.info("run_complete", results=results)

    if cleanup:
        await cleanup()
    await db.close()


async def run_scheduled() -> None:
    """Start the scheduler and run the pipeline daily at 4:30 PM ET."""
    db = Database(url=settings.database_url)
    await db.init_db()
    notifier = TelegramNotifier()

    executor, cleanup = await _create_executor(db)

    orchestrator = Orchestrator(db, notifier=notifier, executor=executor)
    scheduler = AsyncIOScheduler()

    # Shared lock: prevents scheduled pipeline and sentinel crisis session from running concurrently
    _pipeline_lock = asyncio.Lock()

    # Run 3x daily during market hours (US/Eastern)
    schedules = [
        ("morning", 10, 0, "Morning"),  # 10:00 AM — 30 min after open
        ("midday", 12, 30, "Midday"),  # 12:30 PM — midday rebalance
        ("before_close", 15, 45, "Closing"),  # 3:45 PM  — 15 min before close
    ]

    for label, hour, minute, session_name in schedules:

        async def scheduled_job(session=session_name):
            async with _pipeline_lock:
                try:
                    results = await orchestrator.run_all_tenants(session=session)
                    log.info("scheduled_run_complete", session=session, tenants=len(results))
                    # Record session time for sentinel cooldown (per-tenant)
                    try:
                        from src.agent.sentinel import record_session_time

                        record_session_time()
                    except Exception:
                        pass
                except Exception as e:
                    log.error("scheduled_run_failed", session=session, error=str(e))
                    await notifier.send_error(f"Pipeline failed ({session}): {e}")

        scheduler.add_job(
            scheduled_job,
            CronTrigger(
                hour=hour,
                minute=minute,
                day_of_week="mon-fri",
                timezone="US/Eastern",
            ),
            id=f"pipeline_{label}",
            name=f"Kukulkan {session_name}",
        )

    # Intraday snapshots: every 15 min during market hours (Mon-Fri 9:30-16:00 ET)
    from src.intraday import collect_intraday_snapshot
    from src.utils.market_time import MarketPhase

    async def intraday_snapshot_job():
        from datetime import date as _date

        today = _date.today()
        if not is_market_open(today):
            return
        try:
            tenants = await db.get_active_tenants()
            for tenant in tenants:
                if not Orchestrator.tenant_fully_configured(tenant):
                    continue
                try:
                    await collect_intraday_snapshot(db, tenant, market_phase=MarketPhase.MARKET)
                except Exception as e:
                    log.warning(
                        "intraday_snapshot_tenant_failed",
                        tenant_id=tenant.id,
                        error=str(e),
                    )
            log.debug("intraday_snapshot_job_complete")
        except Exception as e:
            log.error("intraday_snapshot_job_failed", error=str(e))

    scheduler.add_job(
        intraday_snapshot_job,
        CronTrigger(
            minute="*/15",
            hour="9-16",
            day_of_week="mon-fri",
            timezone="US/Eastern",
        ),
        id="intraday_snapshots",
        name="Kukulkan Intraday Snapshots",
    )

    # Pre-market snapshots: every 15 min, 7:00-9:15 ET Mon-Fri
    async def extended_snapshot_job(phase: MarketPhase):
        from datetime import date as _date

        today = _date.today()
        if not is_market_open(today):
            return
        try:
            tenants = await db.get_active_tenants()
            for tenant in tenants:
                if not Orchestrator.tenant_fully_configured(tenant):
                    continue
                try:
                    await collect_intraday_snapshot(db, tenant, market_phase=phase)
                except Exception as e:
                    log.warning(
                        "extended_snapshot_tenant_failed",
                        tenant_id=tenant.id,
                        phase=phase.value,
                        error=str(e),
                    )
            log.debug("extended_snapshot_job_complete", phase=phase.value)
        except Exception as e:
            log.error("extended_snapshot_job_failed", phase=phase.value, error=str(e))

    scheduler.add_job(
        extended_snapshot_job,
        CronTrigger(
            minute="*/15",
            hour="7-9",
            day_of_week="mon-fri",
            timezone="US/Eastern",
        ),
        args=[MarketPhase.PREMARKET],
        id="intraday_premarket",
        name="Kukulkan Pre-Market Snapshots",
    )

    # After-hours snapshots: every 15 min, 16:00-19:45 ET Mon-Fri
    scheduler.add_job(
        extended_snapshot_job,
        CronTrigger(
            minute="*/15",
            hour="16-19",
            day_of_week="mon-fri",
            timezone="US/Eastern",
        ),
        args=[MarketPhase.AFTERHOURS],
        id="intraday_afterhours",
        name="Kukulkan After-Hours Snapshots",
    )

    # Sentinel checks: every 30 min during market hours (Mon-Fri 10:30-15:30 ET)
    # Runs between the 3 scheduled sessions to catch stop proximity, regime shifts, fill issues

    async def sentinel_check_job():
        from datetime import date as _date

        today = _date.today()
        if not is_market_open(today):
            return
        if not settings.sentinel_enabled:
            return
        try:
            from src.agent.sentinel import SentinelRunner
            from src.notifications.telegram_factory import TelegramFactory

            tenants = await db.get_active_tenants()
            targets = tenants if tenants else [None]

            for tenant in targets:
                if tenant and not Orchestrator.tenant_fully_configured(tenant):
                    continue
                tid = tenant.id if tenant else "default"
                tenant_executor = executor if not tenant else None
                if tenant:
                    try:
                        from src.execution.client_factory import AlpacaClientFactory

                        client = AlpacaClientFactory.get_trading_client(tenant)
                        from src.execution.alpaca_executor import AlpacaExecutor

                        tenant_executor = AlpacaExecutor(db, client)
                    except Exception:
                        tenant_executor = None

                runner = SentinelRunner(db=db, executor=tenant_executor, tenant_id=tid)
                result = await runner.run_all_checks()

                # Publish SSE events for any alerts
                if result.alerts:
                    try:
                        from src.events.event_bus import EventType, event_bus

                        alert_dicts = [
                            {
                                "level": a.level.value,
                                "check_type": a.check_type,
                                "ticker": a.ticker,
                                "message": a.message,
                            }
                            for a in result.alerts
                        ]
                        event_bus.publish(
                            EventType.SENTINEL_ALERT,
                            tenant_id=tid,
                            data={
                                "max_level": result.max_level.value,
                                "alerts": alert_dicts,
                                "checks_run": result.checks_run,
                            },
                        )
                    except Exception as e:
                        log.warning("sentinel_sse_publish_failed", error=str(e))

                # Send Telegram alert if warning or critical (with per-ticker throttle)
                if result.max_level.value in ("warning", "critical"):
                    try:
                        from src.agent.sentinel import record_alert_sent, should_send_alert

                        tenant_notifier = notifier
                        if tenant:
                            try:
                                tenant_notifier = TelegramFactory.get_notifier(tenant)
                            except Exception:
                                pass
                        # Filter to only alerts we haven't recently sent
                        alert_dicts = [
                            {
                                "level": a.level.value,
                                "check_type": a.check_type,
                                "ticker": a.ticker,
                                "message": a.message,
                            }
                            for a in result.alerts
                            if should_send_alert(a.ticker, tenant_id=tid)
                        ]
                        if alert_dicts:
                            await tenant_notifier.send_sentinel_alert(alert_dicts, result.max_level.value)
                            for a in alert_dicts:
                                record_alert_sent(a["ticker"], tenant_id=tid)
                    except Exception as e:
                        log.warning("sentinel_telegram_failed", tenant_id=tid, error=str(e))

                # Escalation: trigger crisis session if critical + guards allow
                if result.needs_escalation:
                    from src.agent.sentinel import can_escalate, record_escalation

                    log.warning(
                        "sentinel_escalation_needed",
                        tenant_id=tid,
                        max_level=result.max_level.value,
                        alert_count=len(result.alerts),
                    )
                    try:
                        from src.events.event_bus import EventType, event_bus

                        event_bus.publish(
                            EventType.SENTINEL_ESCALATION,
                            tenant_id=tid,
                            data={"reason": "sentinel_critical", "alert_count": len(result.alerts)},
                        )
                    except Exception:
                        pass

                    if can_escalate(
                        max_per_day=settings.sentinel_max_escalations_per_day,
                        tenant_id=tid,
                    ):
                        async with _pipeline_lock:
                            try:
                                log.info("sentinel_crisis_session_starting", tenant_id=tid)
                                crisis_orchestrator = Orchestrator(db, notifier=notifier, executor=tenant_executor)
                                await crisis_orchestrator.run_daily(
                                    session="Sentinel-Crisis",
                                    tenant_id=tid,
                                    run_portfolio_a=False,
                                    run_portfolio_b=True,
                                )
                                record_escalation(tenant_id=tid)
                                log.info("sentinel_crisis_session_complete", tenant_id=tid)
                            except Exception as e:
                                log.error("sentinel_crisis_session_failed", tenant_id=tid, error=str(e))
                    else:
                        log.info("sentinel_escalation_blocked", tenant_id=tid)

            log.debug("sentinel_check_job_complete")
        except Exception as e:
            log.error("sentinel_check_job_failed", error=str(e))

    scheduler.add_job(
        sentinel_check_job,
        CronTrigger(
            minute="*/30",
            hour="10-15",
            day_of_week="mon-fri",
            timezone="US/Eastern",
        ),
        id="sentinel_checks",
        name="Kukulkan Sentinel Checks",
    )

    # Extended hours sentinel: pre-market (7:00-9:00 ET) and after-hours (16:30-19:30 ET)
    async def extended_sentinel_job(phase_str: str):
        from datetime import date as _date

        today = _date.today()
        if not is_market_open(today):
            return
        if not settings.sentinel_enabled:
            return
        try:
            from src.agent.sentinel import SentinelRunner
            from src.notifications.telegram_factory import TelegramFactory

            tenants = await db.get_active_tenants()
            for tenant in tenants:
                if not Orchestrator.tenant_fully_configured(tenant):
                    continue
                tid = tenant.id
                runner = SentinelRunner(db=db, tenant_id=tid, market_phase=phase_str)
                result = await runner.run_all_checks()

                # SSE event (always fires)
                if result.alerts:
                    try:
                        from src.events.event_bus import EventType, event_bus

                        alert_dicts = [
                            {
                                "level": a.level.value,
                                "check_type": a.check_type,
                                "ticker": a.ticker,
                                "message": a.message,
                                "market_phase": phase_str,
                            }
                            for a in result.alerts
                        ]
                        event_bus.publish(
                            EventType.SENTINEL_ALERT,
                            tenant_id=tid,
                            data={
                                "max_level": result.max_level.value,
                                "alerts": alert_dicts,
                                "market_phase": phase_str,
                            },
                        )
                    except Exception:
                        pass

                # Queue actions instead of crisis session
                if result.max_level.value in ("warning", "critical"):
                    from src.agent.sentinel import record_alert_sent, should_send_alert

                    for a in result.alerts:
                        if a.level.value in ("warning", "critical"):
                            action_type = "sell" if a.level == AlertLevel.CRITICAL else "review"
                            await db.save_sentinel_action(
                                tenant_id=tid,
                                action_type=action_type,
                                ticker=a.ticker,
                                reason=a.message,
                                source=f"{phase_str}_sentinel",
                                alert_level=a.level.value,
                            )

                    # Send or queue Telegram alert
                    try:
                        tenant_notifier = TelegramFactory.get_notifier(tenant)
                        unsent = [a for a in result.alerts if should_send_alert(a.ticker, tenant_id=tid)]
                        if unsent:
                            phase_label = "PRE-MARKET" if phase_str == "premarket" else "AFTER HOURS"
                            alert_msg = f"{phase_label} Sentinel Alert\n\n"
                            for a in unsent:
                                alert_msg += f"[{a.level.value.upper()}] {a.ticker}: {a.message}\n"
                            alert_msg += "\nAction queued for morning session."
                            await tenant_notifier.send_message_or_queue(
                                db=db,
                                tenant_id=tid,
                                message=alert_msg,
                                action_type="review",
                                ticker=unsent[0].ticker,
                                alert_level=result.max_level.value,
                                source=f"{phase_str}_sentinel",
                            )
                            for a in unsent:
                                record_alert_sent(a.ticker, tenant_id=tid)
                    except Exception as e:
                        log.warning("extended_sentinel_telegram_failed", tenant_id=tid, error=str(e))

            log.debug("extended_sentinel_job_complete", phase=phase_str)
        except Exception as e:
            log.error("extended_sentinel_job_failed", phase=phase_str, error=str(e))

    from src.agent.sentinel import AlertLevel

    # Pre-market sentinel: every 30 min, 7:00-9:00 ET (offset to avoid snapshot collision at :00/:15/:30/:45)
    scheduler.add_job(
        extended_sentinel_job,
        CronTrigger(minute="15,45", hour="7-9", day_of_week="mon-fri", timezone="US/Eastern"),
        args=["premarket"],
        id="sentinel_premarket",
        name="Kukulkan Pre-Market Sentinel",
    )

    # After-hours sentinel: every 30 min, 16:00-19:00 ET (offset to avoid snapshot collision)
    scheduler.add_job(
        extended_sentinel_job,
        CronTrigger(minute="15,45", hour="16-19", day_of_week="mon-fri", timezone="US/Eastern"),
        args=["afterhours"],
        id="sentinel_afterhours",
        name="Kukulkan After-Hours Sentinel",
    )

    # Weekend crypto sentinel: hourly Sat-Sun 8AM-8PM ET (BTC trades 24/7)
    async def crypto_sentinel_job():
        from datetime import date as _date

        today = _date.today()
        if today.weekday() < 5:  # Mon-Fri handled by regular sentinel
            return
        if not settings.sentinel_enabled:
            return
        try:
            from src.agent.sentinel import SentinelRunner
            from src.notifications.telegram_factory import TelegramFactory

            tenants = await db.get_active_tenants()
            for tenant in tenants:
                if not Orchestrator.tenant_fully_configured(tenant):
                    continue
                tid = tenant.id
                runner = SentinelRunner(db=db, tenant_id=tid, market_phase="weekend")
                result = await runner.run_all_checks(crypto_only=True)

                # SSE event (always fires)
                if result.alerts:
                    try:
                        from src.events.event_bus import EventType, event_bus

                        alert_dicts = [
                            {
                                "level": a.level.value,
                                "check_type": a.check_type,
                                "ticker": a.ticker,
                                "message": a.message,
                                "market_phase": "weekend",
                            }
                            for a in result.alerts
                        ]
                        event_bus.publish(
                            EventType.SENTINEL_ALERT,
                            tenant_id=tid,
                            data={
                                "max_level": result.max_level.value,
                                "alerts": alert_dicts,
                                "market_phase": "weekend",
                            },
                        )
                    except Exception:
                        pass

                # Queue actions + Telegram for warnings/criticals
                if result.max_level.value in ("warning", "critical"):
                    from src.agent.sentinel import record_alert_sent, should_send_alert

                    for a in result.alerts:
                        if a.level.value in ("warning", "critical"):
                            action_type = "sell" if a.level == AlertLevel.CRITICAL else "review"
                            await db.save_sentinel_action(
                                tenant_id=tid,
                                action_type=action_type,
                                ticker=a.ticker,
                                reason=a.message,
                                source="weekend_crypto_sentinel",
                                alert_level=a.level.value,
                            )

                    try:
                        tenant_notifier = TelegramFactory.get_notifier(tenant)
                        unsent = [a for a in result.alerts if should_send_alert(a.ticker, tenant_id=tid)]
                        if unsent:
                            alert_msg = "WEEKEND CRYPTO Sentinel Alert\n\n"
                            for a in unsent:
                                alert_msg += f"[{a.level.value.upper()}] {a.ticker}: {a.message}\n"
                            alert_msg += "\nAction queued for Monday session."
                            await tenant_notifier.send_message_or_queue(
                                db=db,
                                tenant_id=tid,
                                message=alert_msg,
                                action_type="review",
                                ticker=unsent[0].ticker,
                                alert_level=result.max_level.value,
                                source="weekend_crypto_sentinel",
                            )
                            for a in unsent:
                                record_alert_sent(a.ticker, tenant_id=tid)
                    except Exception as e:
                        log.warning("crypto_sentinel_telegram_failed", tenant_id=tid, error=str(e))

            log.debug("crypto_sentinel_job_complete")
        except Exception as e:
            log.error("crypto_sentinel_job_failed", error=str(e))

    scheduler.add_job(
        crypto_sentinel_job,
        CronTrigger(hour="8-20", minute="0", day_of_week="sat,sun", timezone="US/Eastern"),
        id="sentinel_crypto_weekend",
        name="Kukulkan Weekend Crypto Sentinel",
    )

    # Morning queue delivery: 8:00 AM ET (before morning session)
    async def morning_delivery_job():
        from datetime import date as _date

        today = _date.today()
        if not is_market_open(today):
            return
        try:
            from src.notifications.telegram_factory import TelegramFactory

            tenants = await db.get_active_tenants()
            for tenant in tenants:
                if not Orchestrator.tenant_fully_configured(tenant):
                    continue
                try:
                    tenant_notifier = TelegramFactory.get_notifier(tenant)
                    await tenant_notifier.deliver_morning_queue(db, tenant.id)
                except Exception as e:
                    log.warning("morning_delivery_failed", tenant_id=tenant.id, error=str(e))
            log.debug("morning_delivery_job_complete")
        except Exception as e:
            log.error("morning_delivery_job_failed", error=str(e))

    scheduler.add_job(
        morning_delivery_job,
        CronTrigger(hour=8, minute=0, day_of_week="mon-fri", timezone="US/Eastern"),
        id="morning_delivery",
        name="Kukulkan Morning Queue Delivery",
    )

    # Gap risk alert: 2:45 PM ET (15 min before close session)
    async def gap_risk_alert_job():
        from datetime import date as _date

        today = _date.today()
        if not is_market_open(today):
            return
        try:
            from src.analysis.gap_risk import GapRiskAnalyzer
            from src.notifications.telegram_factory import TelegramFactory

            analyzer = GapRiskAnalyzer()
            tenants = await db.get_active_tenants()
            for tenant in tenants:
                if not Orchestrator.tenant_fully_configured(tenant):
                    continue
                try:
                    assessment = await analyzer.analyze(db, tenant.id)
                    if assessment.rating in ("HIGH", "EXTREME"):
                        message = (
                            f"Pre-Close Gap Risk: {assessment.rating}\n"
                            f"Risk score: {assessment.aggregate_risk_score}\n\n"
                        )
                        if assessment.earnings_tonight:
                            message += f"Earnings tonight: {', '.join(assessment.earnings_tonight)}\n\n"
                        for p in assessment.positions[:5]:
                            if p.recommendation:
                                message += f"  {p.ticker} (score {p.gap_risk_score}): {p.recommendation}\n"
                        message += "\nClose session at 3:45 PM will review these."

                        tenant_notifier = TelegramFactory.get_notifier(tenant)
                        await tenant_notifier.send_message_or_queue(
                            db=db,
                            tenant_id=tenant.id,
                            message=message,
                            action_type="review",
                            ticker=assessment.earnings_tonight[0] if assessment.earnings_tonight else "",
                            alert_level="warning",
                            source="gap_risk",
                        )
                except Exception as e:
                    log.warning("gap_risk_alert_tenant_failed", tenant_id=tenant.id, error=str(e))
            log.debug("gap_risk_alert_job_complete")
        except Exception as e:
            log.error("gap_risk_alert_job_failed", error=str(e))

    scheduler.add_job(
        gap_risk_alert_job,
        CronTrigger(hour=14, minute=45, day_of_week="mon-fri", timezone="US/Eastern"),
        id="gap_risk_alert",
        name="Kukulkan Gap Risk Alert",
    )

    # Signal engine: rank all tickers every 10 min during market hours (zero API cost)
    # Hoisted outside job so _prev_ranks state persists across invocations
    from src.analysis.signal_engine import SignalEngine, signals_to_db_rows

    _signal_engine = SignalEngine()

    async def signal_engine_job():
        from datetime import date as _date

        today = _date.today()
        if not is_market_open(today):
            return
        try:
            engine = _signal_engine
            tenants = await db.get_active_tenants()
            tenant_list = tenants if tenants else []
            if not tenant_list:
                # Fallback: run for default tenant
                tenant_list = [type("T", (), {"id": "default", "is_active": True})()]

            for tenant in tenant_list:
                try:
                    closes, volumes = await db.get_cached_closes_and_volumes()
                    if closes.empty:
                        continue
                    signals = await engine.run(tenant.id, closes, volumes)
                    if signals:
                        rows = signals_to_db_rows(tenant.id, signals)
                        await db.save_signal_batch(rows)
                        # Publish SSE event
                        try:
                            from src.events.event_bus import Event, EventType, event_bus

                            alerts_count = sum(1 for s in signals if s.alerts)
                            event_bus.publish(
                                Event(
                                    type=EventType.SIGNAL_RANKINGS_UPDATED,
                                    tenant_id=tenant.id,
                                    data={
                                        "scored_at": signals[0].scored_at.isoformat(),
                                        "total_tickers": len(signals),
                                        "alerts_triggered": alerts_count,
                                        "top_ticker": signals[0].ticker,
                                        "top_score": signals[0].composite_score,
                                    },
                                )
                            )
                        except Exception as e:
                            log.debug("signal_sse_publish_failed", error=str(e))
                except Exception as e:
                    log.warning("signal_engine_tenant_failed", tenant_id=tenant.id, error=str(e))
            log.debug("signal_engine_job_complete")
        except Exception as e:
            log.error("signal_engine_job_failed", error=str(e))

    scheduler.add_job(
        signal_engine_job,
        CronTrigger(minute="*/10", hour="9-16", day_of_week="mon-fri", timezone="US/Eastern"),
        id="signal_engine",
        name="Kukulkan Signal Engine (10-min ticker ranking)",
    )

    # Fear & Greed Index: fetch twice daily (9:30 AM + 4:30 PM ET)
    async def fear_greed_job():
        from datetime import date as _date

        today = _date.today()
        if not is_market_open(today):
            return
        try:
            from src.data.fear_greed import fetch_and_save

            tenants = await db.get_active_tenants()
            tenant_list = tenants if tenants else []
            if not tenant_list:
                tenant_list = [type("T", (), {"id": "default", "is_active": True})()]
            for tenant in tenant_list:
                try:
                    await fetch_and_save(db, tenant.id)
                except Exception as e:
                    log.warning("fear_greed_tenant_failed", tenant_id=tenant.id, error=str(e))
            log.debug("fear_greed_job_complete")
        except Exception as e:
            log.error("fear_greed_job_failed", error=str(e))

    scheduler.add_job(
        fear_greed_job,
        CronTrigger(hour="9,16", minute=30, day_of_week="mon-fri", timezone="US/Eastern"),
        id="fear_greed",
        name="Kukulkan Fear & Greed Index",
    )

    # Weekly memory compaction (Sunday 6 PM ET)
    memory_manager = AgentMemoryManager()

    async def weekly_compaction():
        try:
            from src.agent.claude_agent import _build_decision_review
            from src.analysis.outcome_tracker import OutcomeTracker
            from src.analysis.track_record import TrackRecord

            tenants = await db.get_active_tenants()
            if tenants:
                for tenant in tenants:
                    if Orchestrator.tenant_fully_configured(tenant):
                        try:
                            outcome_summary = None
                            track_record_text = None
                            try:
                                tracker = OutcomeTracker(db)
                                outcomes = await tracker.get_recent_outcomes(days=7, tenant_id=tenant.id)
                                if outcomes:
                                    outcome_summary = _build_decision_review(outcomes)
                                    stats = TrackRecord().compute(outcomes, min_trades=1)
                                    track_record_text = TrackRecord.format_for_prompt(stats)
                            except Exception as e:
                                log.warning("compaction_feedback_failed", tenant_id=tenant.id, error=str(e))

                            await memory_manager.run_weekly_compaction(
                                db,
                                tenant_id=tenant.id,
                                outcome_summary=outcome_summary,
                                track_record_text=track_record_text,
                            )
                        except Exception as e:
                            log.error(
                                "weekly_compaction_tenant_failed",
                                tenant_id=tenant.id,
                                error=str(e),
                            )
            else:
                outcome_summary = None
                track_record_text = None
                try:
                    tracker = OutcomeTracker(db)
                    outcomes = await tracker.get_recent_outcomes(days=7, tenant_id="default")
                    if outcomes:
                        outcome_summary = _build_decision_review(outcomes)
                        stats = TrackRecord().compute(outcomes, min_trades=1)
                        track_record_text = TrackRecord.format_for_prompt(stats)
                except Exception as e:
                    log.warning("compaction_feedback_failed_default", error=str(e))

                await memory_manager.run_weekly_compaction(
                    db,
                    outcome_summary=outcome_summary,
                    track_record_text=track_record_text,
                )
            await db.delete_expired_memories()
            log.info("weekly_compaction_job_complete")
        except Exception as e:
            log.error("weekly_compaction_job_failed", error=str(e))

    scheduler.add_job(
        weekly_compaction,
        CronTrigger(
            day_of_week="sun",
            hour=18,
            minute=0,
            timezone="US/Eastern",
        ),
        id="weekly_memory_compaction",
        name="Kukulkan Weekly Memory Compaction",
    )

    # Weekly performance report (Friday 5 PM ET)
    async def weekly_report():
        from datetime import date as _date

        today = _date.today()
        if not is_market_open(today):
            log.info("weekly_report_skipped_holiday")
            return
        try:
            tenants = await db.get_active_tenants()
            if tenants:
                from src.notifications.telegram_factory import TelegramFactory

                for tenant in tenants:
                    if not Orchestrator.tenant_fully_configured(tenant):
                        log.info(
                            "weekly_report_tenant_skipped",
                            tenant_id=tenant.id,
                        )
                        continue
                    try:
                        from src.utils.allocations import resolve_from_tenant

                        tenant_notifier = TelegramFactory.get_notifier(tenant)
                        tenant_alloc = resolve_from_tenant(tenant)
                        reporter = WeeklyReporter(
                            db,
                            tenant_notifier,
                            tenant_id=tenant.id,
                            allocations=tenant_alloc,
                            run_portfolio_a=tenant.run_portfolio_a,
                            run_portfolio_b=tenant.run_portfolio_b,
                        )
                        await reporter.generate_and_send(today)
                    except Exception as e:
                        log.error(
                            "weekly_report_tenant_failed",
                            tenant_id=tenant.id,
                            error=str(e),
                        )
            else:
                reporter = WeeklyReporter(db, notifier)
                await reporter.generate_and_send(today)
            log.info("weekly_report_job_complete")
        except Exception as e:
            log.error("weekly_report_job_failed", error=str(e))

    scheduler.add_job(
        weekly_report,
        CronTrigger(
            day_of_week="fri",
            hour=17,
            minute=0,
            timezone="US/Eastern",
        ),
        id="weekly_report",
        name="Kukulkan Weekly Report",
    )

    # Weekly self-improvement loop (Sunday 4 PM ET)
    async def weekly_improvement_loop():
        try:
            from src.analysis.improvement_pipeline import WeeklyImprovementPipeline

            tenants = await db.get_active_tenants()
            targets = tenants if tenants else []
            if not targets:
                targets = [None]

            for tenant in targets:
                tid = tenant.id if tenant else "default"
                if tenant and not Orchestrator.tenant_fully_configured(tenant):
                    continue
                try:
                    pipeline = WeeklyImprovementPipeline(db)
                    notifier_instance = None
                    if tenant:
                        try:
                            from src.notifications.telegram_bot import TelegramNotifier
                            from src.utils.crypto import decrypt_value

                            if tenant.telegram_bot_token_enc and tenant.telegram_chat_id_enc:
                                notifier_instance = TelegramNotifier(
                                    bot_token=decrypt_value(tenant.telegram_bot_token_enc),
                                    chat_id=decrypt_value(tenant.telegram_chat_id_enc),
                                )
                        except Exception:
                            pass
                    result = await pipeline.run(tenant_id=tid, notifier=notifier_instance)
                    log.info("weekly_improvement_tenant_done", tenant_id=tid, result=result)
                except Exception as e:
                    log.error("weekly_improvement_tenant_failed", tenant_id=tid, error=str(e))

            log.info("weekly_improvement_loop_complete")
        except Exception as e:
            log.error("weekly_improvement_loop_failed", error=str(e))

    scheduler.add_job(
        weekly_improvement_loop,
        CronTrigger(
            day_of_week="sun",
            hour=16,
            minute=0,
            timezone="US/Eastern",
        ),
        id="weekly_improvement_loop",
        name="Kukulkan Weekly Self-Improvement",
    )

    # Weekly playbook + calibration generation (Sunday 5 PM ET)
    async def weekly_playbook_calibration():
        try:
            from src.analysis.conviction_calibrator import ConvictionCalibrator
            from src.analysis.outcome_tracker import OutcomeTracker
            from src.analysis.playbook_generator import PlaybookGenerator

            tenants = await db.get_active_tenants()
            targets = tenants if tenants else []
            # Include default tenant if no tenants configured
            if not targets:
                targets = [None]

            for tenant in targets:
                tid = tenant.id if tenant else "default"
                if tenant and not Orchestrator.tenant_fully_configured(tenant):
                    continue
                try:
                    tracker = OutcomeTracker(db)
                    outcomes = await tracker.get_recent_outcomes(days=90, tenant_id=tid)
                    if not outcomes:
                        continue

                    # Generate and save playbook
                    playbook_cells = PlaybookGenerator().generate(outcomes)
                    if playbook_cells:
                        cell_dicts = [
                            {
                                "regime": c.regime,
                                "sector": c.sector,
                                "total_trades": c.total,
                                "wins": c.wins,
                                "losses": c.losses,
                                "win_rate_pct": c.win_rate_pct,
                                "avg_pnl_pct": c.avg_pnl_pct,
                                "recommendation": c.recommendation,
                            }
                            for c in playbook_cells
                        ]
                        await db.save_playbook_snapshot(cell_dicts, tenant_id=tid)

                    # Generate and save calibration
                    cal_buckets = ConvictionCalibrator().calibrate(outcomes)
                    if cal_buckets:
                        bucket_dicts = [
                            {
                                "conviction_level": b.conviction,
                                "total_trades": b.total,
                                "wins": b.wins,
                                "losses": b.losses,
                                "win_rate_pct": b.win_rate_pct,
                                "avg_pnl_pct": b.avg_pnl_pct,
                                "assessment": b.assessment,
                                "suggested_multiplier": b.suggested_multiplier,
                            }
                            for b in cal_buckets
                        ]
                        await db.save_conviction_calibration(bucket_dicts, tenant_id=tid)

                    log.info("playbook_calibration_generated", tenant_id=tid)
                except Exception as e:
                    log.error("playbook_calibration_tenant_failed", tenant_id=tid, error=str(e))

            log.info("weekly_playbook_calibration_complete")
        except Exception as e:
            log.error("weekly_playbook_calibration_failed", error=str(e))

    scheduler.add_job(
        weekly_playbook_calibration,
        CronTrigger(
            day_of_week="sun",
            hour=17,
            minute=0,
            timezone="US/Eastern",
        ),
        id="weekly_playbook_calibration",
        name="Kukulkan Weekly Playbook & Calibration",
    )

    # Weekly intraday snapshot + signal cleanup (Sunday 7 PM ET)
    async def intraday_cleanup_job():
        try:
            deleted = await db.purge_old_intraday_snapshots(days=90)
            if deleted:
                log.info("intraday_cleanup_complete", deleted=deleted)
            # Cleanup old signal data (keep last 48h for debugging)
            tenants = await db.get_active_tenants()
            for tenant in tenants or []:
                try:
                    sig_deleted = await db.cleanup_old_signals(tenant.id, keep_hours=48)
                    if sig_deleted:
                        log.info("signal_cleanup_complete", tenant_id=tenant.id, deleted=sig_deleted)
                except Exception as e:
                    log.warning("signal_cleanup_failed", tenant_id=tenant.id, error=str(e))
            # Cleanup ChromaDB news older than 180 days (6-month retention)
            try:
                from config.settings import settings as _settings
                from src.storage.vector_store import VectorStore

                _vs = VectorStore(host=_settings.chroma.host, port=_settings.chroma.port)
                chroma_deleted = _vs.cleanup_old(days=180)
                if chroma_deleted:
                    log.info("chromadb_news_cleanup_complete", deleted=chroma_deleted)
            except Exception as e:
                log.warning("chromadb_news_cleanup_failed", error=str(e))
        except Exception as e:
            log.error("intraday_cleanup_failed", error=str(e))

    scheduler.add_job(
        intraday_cleanup_job,
        CronTrigger(
            day_of_week="sun",
            hour=19,
            minute=0,
            timezone="US/Eastern",
        ),
        id="intraday_cleanup",
        name="Kukulkan Intraday Cleanup",
    )

    scheduler.start()
    log.info(
        "scheduler_started",
        schedule="Mon-Fri at 10:00, 12:30, 15:45 ET; "
        "Sentinel every 30min 10-15 ET; "
        "Intraday every 15min 9-16 ET; "
        "Fri 17:00 ET report; Sun 17:00 ET playbook+calibration; "
        "Sun 18:00 ET compaction; "
        "Sun 19:00 ET cleanup",
    )

    # Keep running until interrupted
    stop_event = asyncio.Event()

    def handle_signal(sig, frame):
        log.info("shutdown_signal_received", signal=sig)
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    await stop_event.wait()

    scheduler.shutdown()
    if cleanup:
        await cleanup()
    await db.close()
    log.info("kukulkan_shutdown_complete")


def main() -> None:
    """Parse args and run the appropriate mode."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Kukulkan Trading Bot")
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Run the daily pipeline immediately and exit",
    )
    args = parser.parse_args()

    mode = "run-now" if args.run_now else "scheduled"
    log.info("kukulkan_starting", mode=mode, executor=settings.executor)

    if args.run_now:
        asyncio.run(run_once())
    else:
        asyncio.run(run_scheduled())


if __name__ == "__main__":
    main()
