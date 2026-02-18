"""Market phase detection: pre-market, market hours, after-hours, closed.

Uses US/Eastern timezone for phase boundaries. Leverages exchange_calendars
for holiday detection via the existing market_calendar module.
"""

from datetime import datetime, time
from enum import Enum
from zoneinfo import ZoneInfo

ET = ZoneInfo("US/Eastern")


class MarketPhase(str, Enum):
    PREMARKET = "premarket"  # 7:00 - 9:30 ET
    MARKET = "market"  # 9:30 - 16:00 ET
    AFTERHOURS = "afterhours"  # 16:00 - 20:00 ET
    CLOSED = "closed"  # 20:00 - 7:00 ET, weekends, holidays


def get_market_phase(now: datetime | None = None) -> MarketPhase:
    """Determine current market phase based on US/Eastern time.

    Args:
        now: Override datetime (must be tz-aware or naive ET assumed).
             Defaults to current time.

    Returns:
        Current MarketPhase.
    """
    if now is None:
        now = datetime.now(ET)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=ET)
    else:
        now = now.astimezone(ET)

    # Weekends are always closed
    if now.weekday() >= 5:
        return MarketPhase.CLOSED

    # Check holidays via exchange_calendars
    if not is_trading_day(now):
        return MarketPhase.CLOSED

    t = now.time()
    if time(7, 0) <= t < time(9, 30):
        return MarketPhase.PREMARKET
    elif time(9, 30) <= t < time(16, 0):
        return MarketPhase.MARKET
    elif time(16, 0) <= t < time(20, 0):
        return MarketPhase.AFTERHOURS
    else:
        return MarketPhase.CLOSED


def is_trading_day(dt: datetime | None = None) -> bool:
    """Check if the given date is a US equity trading day.

    Uses exchange_calendars (NYSE) for accurate holiday detection.
    """
    if dt is None:
        dt = datetime.now(ET)

    if dt.weekday() >= 5:
        return False

    try:
        from src.utils.market_calendar import is_market_open

        return is_market_open(dt.date() if isinstance(dt, datetime) else dt)
    except Exception:
        # Fallback: assume weekdays are trading days
        return True
