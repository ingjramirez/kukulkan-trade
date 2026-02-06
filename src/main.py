"""Atlas Trading Bot — main entry point.

Usage:
    python -m src.main              # Start scheduler (runs daily at 4:30 PM ET)
    python -m src.main --run-now    # Run pipeline immediately, then exit
"""

import argparse
import asyncio
import signal
import sys

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config.settings import settings
from src.orchestrator import Orchestrator
from src.storage.database import Database

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
            structlog.get_level_from_name(settings.log_level)
        ),
    )


async def run_once() -> None:
    """Run the daily pipeline once and exit."""
    db = Database(url=settings.database_url)
    await db.init_db()

    orchestrator = Orchestrator(db)
    summary = await orchestrator.run_daily()

    log.info("run_complete", summary=summary)
    await db.close()


async def run_scheduled() -> None:
    """Start the scheduler and run the pipeline daily at 4:30 PM ET."""
    db = Database(url=settings.database_url)
    await db.init_db()

    orchestrator = Orchestrator(db)
    scheduler = AsyncIOScheduler()

    async def scheduled_job():
        try:
            summary = await orchestrator.run_daily()
            log.info("scheduled_run_complete", summary=summary)
        except Exception as e:
            log.error("scheduled_run_failed", error=str(e))

    # Run daily at 4:30 PM Eastern (after market close)
    scheduler.add_job(
        scheduled_job,
        CronTrigger(hour=16, minute=30, timezone="US/Eastern"),
        id="daily_pipeline",
        name="Atlas Daily Pipeline",
    )

    scheduler.start()
    log.info("scheduler_started", schedule="daily at 4:30 PM ET")

    # Keep running until interrupted
    stop_event = asyncio.Event()

    def handle_signal(sig, frame):
        log.info("shutdown_signal_received", signal=sig)
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    await stop_event.wait()

    scheduler.shutdown()
    await db.close()
    log.info("atlas_shutdown_complete")


def main() -> None:
    """Parse args and run the appropriate mode."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Atlas Trading Bot")
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Run the daily pipeline immediately and exit",
    )
    args = parser.parse_args()

    log.info("atlas_starting", mode="run-now" if args.run_now else "scheduled")

    if args.run_now:
        asyncio.run(run_once())
    else:
        asyncio.run(run_scheduled())


if __name__ == "__main__":
    main()
