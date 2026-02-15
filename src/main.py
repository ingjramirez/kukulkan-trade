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
from src.agent.claude_agent import ClaudeAgent
from src.agent.memory import AgentMemoryManager
from src.notifications.telegram_bot import TelegramNotifier
from src.notifications.weekly_report import WeeklyReporter
from src.orchestrator import Orchestrator
from src.storage.database import Database
from src.utils.market_calendar import is_market_open

log = structlog.get_logger()


def setup_logging() -> None:
    """Configure structlog for the application."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(settings.log_level)),
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

    # Run 3x daily during market hours (US/Eastern)
    schedules = [
        ("morning", 10, 0, "Morning"),  # 10:00 AM — 30 min after open
        ("midday", 12, 30, "Midday"),  # 12:30 PM — midday rebalance
        ("before_close", 15, 45, "Closing"),  # 3:45 PM  — 15 min before close
    ]

    for label, hour, minute, session_name in schedules:

        async def scheduled_job(session=session_name):
            try:
                results = await orchestrator.run_all_tenants(session=session)
                log.info("scheduled_run_complete", session=session, tenants=len(results))
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

    async def intraday_snapshot_job():
        from datetime import date as _date

        today = _date.today()
        if not is_market_open(today):
            return
        try:
            tenants = await db.get_active_tenants()
            for tenant in tenants:
                if not Orchestrator._tenant_fully_configured(tenant):
                    continue
                try:
                    await collect_intraday_snapshot(db, tenant)
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

    # Weekly memory compaction (Sunday 6 PM ET)
    memory_manager = AgentMemoryManager()
    agent = ClaudeAgent()

    async def weekly_compaction():
        try:
            from src.agent.claude_agent import _build_decision_review
            from src.analysis.outcome_tracker import OutcomeTracker
            from src.analysis.track_record import TrackRecord

            tenants = await db.get_active_tenants()
            if tenants:
                for tenant in tenants:
                    if Orchestrator._tenant_fully_configured(tenant):
                        try:
                            # Compute outcome feedback for evaluation
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
                                agent,
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
                # Default tenant path
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
                    agent,
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
                    if not Orchestrator._tenant_fully_configured(tenant):
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

    # Weekly intraday snapshot cleanup (Sunday 7 PM ET)
    async def intraday_cleanup_job():
        try:
            deleted = await db.purge_old_intraday_snapshots(days=90)
            if deleted:
                log.info("intraday_cleanup_complete", deleted=deleted)
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

    # Weekly conversation message cleanup (Sunday 7:30 PM ET)
    async def conversation_cleanup_job():
        try:
            from src.agent.conversation_store import ConversationStore

            store = ConversationStore(db)
            tenants = await db.get_active_tenants()
            total_cleaned = 0
            if tenants:
                for tenant in tenants:
                    cleaned = await store.cleanup_old_messages(tenant.id, days=30)
                    total_cleaned += cleaned
            else:
                cleaned = await store.cleanup_old_messages("default", days=30)
                total_cleaned += cleaned
            if total_cleaned:
                log.info("conversation_cleanup_complete", cleaned=total_cleaned)
        except Exception as e:
            log.error("conversation_cleanup_failed", error=str(e))

    scheduler.add_job(
        conversation_cleanup_job,
        CronTrigger(
            day_of_week="sun",
            hour=19,
            minute=30,
            timezone="US/Eastern",
        ),
        id="conversation_cleanup",
        name="Kukulkan Conversation Cleanup",
    )

    scheduler.start()
    log.info(
        "scheduler_started",
        schedule="Mon-Fri at 10:00, 12:30, 15:45 ET; "
        "Intraday every 15min 9-16 ET; "
        "Fri 17:00 ET report; Sun 18:00 ET compaction; "
        "Sun 19:00 ET cleanup; Sun 19:30 ET conversation cleanup",
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
