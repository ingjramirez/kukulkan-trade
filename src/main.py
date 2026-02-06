"""Atlas Trading Bot — main entry point.

Usage:
    python -m src.main              # Start scheduler (runs daily at 4:30 PM ET)
    python -m src.main --run-now    # Run pipeline immediately, then exit

Executor is controlled by EXECUTOR env var: "alpaca", "ibkr", or "paper" (default).
"""

import argparse
import asyncio
import logging
import signal
import sys

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config.settings import settings
from src.notifications.telegram_bot import TelegramNotifier
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

    elif executor_type == "ibkr":
        try:
            from src.execution.ibkr_client import IBKRClient
            from src.execution.ibkr_executor import IBKRExecutor

            client = IBKRClient(
                host=settings.ibkr.host,
                port=settings.ibkr.port,
                client_id=settings.ibkr.client_id,
            )
            connected = await client.connect()
            if connected:
                log.info("ibkr_connected", host=settings.ibkr.host, port=settings.ibkr.port)
                return IBKRExecutor(db, client), client.disconnect
            else:
                log.warning("ibkr_connection_failed_using_paper_trader")
        except Exception as e:
            log.warning("ibkr_not_available_using_paper_trader", error=str(e))

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

    async def scheduled_job():
        try:
            summary = await orchestrator.run_daily()
            log.info("scheduled_run_complete", summary=summary)
        except Exception as e:
            log.error("scheduled_run_failed", error=str(e))
            await notifier.send_error(f"Pipeline failed: {e}")

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
    if cleanup:
        await cleanup()
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

    log.info("atlas_starting", mode="run-now" if args.run_now else "scheduled", executor=settings.executor)

    if args.run_now:
        asyncio.run(run_once())
    else:
        asyncio.run(run_scheduled())


if __name__ == "__main__":
    main()
