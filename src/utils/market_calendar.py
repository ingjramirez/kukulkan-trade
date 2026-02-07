"""Market calendar utilities using exchange_calendars."""

from datetime import date, timedelta

import exchange_calendars as xcals
import structlog

log = structlog.get_logger()

# NYSE calendar covers all US equity trading days
_nyse = xcals.get_calendar("XNYS")


def is_market_open(check_date: date | None = None) -> bool:
    """Check if the US stock market is open on the given date.

    Args:
        check_date: Date to check. Defaults to today.

    Returns:
        True if the market is open (regular trading day).
    """
    check_date = check_date or date.today()
    is_open = _nyse.is_session(check_date)
    if not is_open:
        log.info("market_closed", date=str(check_date))
    return is_open


def is_early_close(check_date: date | None = None) -> bool:
    """Check if the market closes early (1 PM ET) on the given date.

    Early close days: day before Independence Day, Black Friday,
    Christmas Eve, etc.
    """
    check_date = check_date or date.today()
    if not _nyse.is_session(check_date):
        return False
    close_time = _nyse.session_close(check_date)
    # Normal close is 4 PM ET (21:00 UTC). Early close is 1 PM ET (18:00 UTC).
    return close_time.hour < 21


def next_trading_day(from_date: date | None = None) -> date:
    """Get the next trading day after the given date."""
    from_date = from_date or date.today()
    end = from_date + timedelta(days=10)
    sessions = _nyse.sessions_in_range(
        from_date.isoformat(), end.isoformat(),
    )
    for s in sessions:
        if s.date() > from_date:
            return s.date()
    return from_date  # fallback


def trading_days_between(start: date, end: date) -> list[date]:
    """Get actual trading days between two dates (exclusive on both ends).

    Uses the NYSE calendar, which accounts for holidays.

    Args:
        start: Start date (exclusive).
        end: End date (exclusive).

    Returns:
        List of trading dates between start and end.
    """
    if start >= end:
        return []
    sessions = _nyse.sessions_in_range(start.isoformat(), end.isoformat())
    return [s.date() for s in sessions if start < s.date() < end]
