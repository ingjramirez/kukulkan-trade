"""Tests for MarketPhase detection and is_trading_day."""

from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from src.utils.market_time import ET, MarketPhase, get_market_phase, is_trading_day


def _et(hour: int, minute: int = 0, weekday: int = 0) -> datetime:
    """Create an ET datetime on a specific weekday (0=Mon).

    Base: 2026-02-17 is Tuesday (not a holiday).
    weekday offset: 0=Tue, 1=Wed, ..., 3=Fri, 4=Sat, 5=Sun.
    """
    day = 17 + weekday
    return datetime(2026, 2, day, hour, minute, tzinfo=ET)


class TestGetMarketPhase:
    def test_premarket_phase(self) -> None:
        assert get_market_phase(_et(7, 0)) == MarketPhase.PREMARKET
        assert get_market_phase(_et(8, 30)) == MarketPhase.PREMARKET
        assert get_market_phase(_et(9, 29)) == MarketPhase.PREMARKET

    def test_market_phase(self) -> None:
        assert get_market_phase(_et(9, 30)) == MarketPhase.MARKET
        assert get_market_phase(_et(12, 0)) == MarketPhase.MARKET
        assert get_market_phase(_et(15, 59)) == MarketPhase.MARKET

    def test_afterhours_phase(self) -> None:
        assert get_market_phase(_et(16, 0)) == MarketPhase.AFTERHOURS
        assert get_market_phase(_et(18, 0)) == MarketPhase.AFTERHOURS
        assert get_market_phase(_et(19, 59)) == MarketPhase.AFTERHOURS

    def test_closed_phase_night(self) -> None:
        assert get_market_phase(_et(20, 0)) == MarketPhase.CLOSED
        assert get_market_phase(_et(23, 0)) == MarketPhase.CLOSED
        assert get_market_phase(_et(5, 0)) == MarketPhase.CLOSED
        assert get_market_phase(_et(6, 59)) == MarketPhase.CLOSED

    def test_closed_phase_weekend(self) -> None:
        saturday = _et(12, 0, weekday=4)  # Saturday (17+4=21)
        sunday = _et(12, 0, weekday=5)  # Sunday (17+5=22)
        assert get_market_phase(saturday) == MarketPhase.CLOSED
        assert get_market_phase(sunday) == MarketPhase.CLOSED

    def test_holiday_returns_closed(self) -> None:
        # Patch is_market_open to return False (holiday)
        with patch("src.utils.market_time.is_trading_day", return_value=False):
            assert get_market_phase(_et(12, 0)) == MarketPhase.CLOSED

    def test_naive_datetime_assumed_et(self) -> None:
        naive = datetime(2026, 2, 17, 12, 0)  # Tuesday noon
        assert get_market_phase(naive) == MarketPhase.MARKET

    def test_other_timezone_converted(self) -> None:
        # 12 PM CT = 1 PM ET → market hours
        ct = ZoneInfo("US/Central")
        dt = datetime(2026, 2, 17, 12, 0, tzinfo=ct)
        assert get_market_phase(dt) == MarketPhase.MARKET


class TestIsTradingDay:
    def test_weekday_is_trading_day(self) -> None:
        tuesday = datetime(2026, 2, 17, tzinfo=ET)
        assert is_trading_day(tuesday) is True

    def test_weekend_not_trading_day(self) -> None:
        saturday = datetime(2026, 2, 21, tzinfo=ET)
        assert is_trading_day(saturday) is False

    def test_fallback_on_import_error(self) -> None:
        with patch("src.utils.market_calendar.is_market_open", side_effect=ImportError):
            tuesday = datetime(2026, 2, 17, tzinfo=ET)
            assert is_trading_day(tuesday) is True
