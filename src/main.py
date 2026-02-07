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
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level)
        ),
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
    summary = await orchestrator.run_daily()

    log.info("run_complete", summary=summary)

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
        ("morning",      10,  0, "Morning"),       # 10:00 AM — 30 min after open
        ("midday",       12, 30, "Midday"),         # 12:30 PM — midday rebalance
        ("before_close", 15, 45, "Closing"),        # 3:45 PM  — 15 min before close
    ]

    for label, hour, minute, session_name in schedules:
        async def scheduled_job(session=session_name):
            try:
                summary = await orchestrator.run_daily(session=session)
                log.info("scheduled_run_complete", session=session, summary=summary)
            except Exception as e:
                log.error("scheduled_run_failed", session=session, error=str(e))
                await notifier.send_error(f"Pipeline failed ({session}): {e}")

        scheduler.add_job(
            scheduled_job,
            CronTrigger(
                hour=hour, minute=minute,
                day_of_week="mon-fri",
                timezone="US/Eastern",
            ),
            id=f"pipeline_{label}",
            name=f"Kukulkan {session_name}",
        )

    # Weekly memory compaction (Sunday 6 PM ET)
    memory_manager = AgentMemoryManager()
    agent = ClaudeAgent()

    async def weekly_compaction():
        try:
            await memory_manager.run_weekly_compaction(db, agent)
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
    reporter = WeeklyReporter(db, notifier)

    async def weekly_report():
        from datetime import date as _date

        today = _date.today()
        if not is_market_open(today):
            log.info("weekly_report_skipped_holiday")
            return
        try:
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

    scheduler.start()
    log.info(
        "scheduler_started",
        schedule="Mon-Fri at 10:00, 12:30, 15:45 ET; "
        "Fri 17:00 ET report; Sun 18:00 ET compaction",
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
