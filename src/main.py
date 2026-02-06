"""Atlas Trading Bot — main entry point."""

import structlog

from config.settings import settings

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


def main() -> None:
    """Start the Atlas trading bot."""
    setup_logging()
    log.info("atlas_starting", log_level=settings.log_level)
    # TODO: Initialize scheduler, connect to IBKR, start strategies
    log.info("atlas_ready")


if __name__ == "__main__":
    main()
