"""Tests for market calendar utilities."""

from datetime import date

from src.utils.market_calendar import (
    is_early_close,
    is_market_open,
    next_trading_day,
    trading_days_between,
)


class TestIsMarketOpen:
    def test_regular_weekday(self):
        """Monday Feb 9 2026 is a regular trading day."""
        assert is_market_open(date(2026, 2, 9)) is True

    def test_saturday(self):
        """Saturday is not a trading day."""
        assert is_market_open(date(2026, 2, 7)) is False

    def test_sunday(self):
        """Sunday is not a trading day."""
        assert is_market_open(date(2026, 2, 8)) is False

    def test_mlk_day(self):
        """MLK Day 2026 (Jan 19) — market closed."""
        assert is_market_open(date(2026, 1, 19)) is False

    def test_christmas(self):
        """Christmas 2025 — market closed."""
        assert is_market_open(date(2025, 12, 25)) is False

    def test_presidents_day(self):
        """Presidents Day 2026 (Feb 16) — market closed."""
        assert is_market_open(date(2026, 2, 16)) is False

    def test_regular_friday(self):
        """A regular Friday should be open."""
        assert is_market_open(date(2026, 2, 6)) is True


class TestIsEarlyClose:
    def test_black_friday(self):
        """Black Friday 2025 (Nov 28) is an early close day."""
        assert is_early_close(date(2025, 11, 28)) is True

    def test_regular_day(self):
        """Regular trading day is not an early close."""
        assert is_early_close(date(2026, 2, 9)) is False

    def test_closed_day_returns_false(self):
        """A fully closed day is not an early close."""
        assert is_early_close(date(2026, 2, 7)) is False  # Saturday


class TestNextTradingDay:
    def test_from_friday(self):
        """Next trading day after Friday is Monday."""
        result = next_trading_day(date(2026, 2, 6))  # Friday
        assert result == date(2026, 2, 9)  # Monday

    def test_from_saturday(self):
        """Next trading day after Saturday is Monday."""
        result = next_trading_day(date(2026, 2, 7))  # Saturday
        assert result == date(2026, 2, 9)  # Monday

    def test_skips_holiday(self):
        """Next trading day after MLK weekend skips Monday."""
        # MLK Day 2026 = Monday Jan 19
        result = next_trading_day(date(2026, 1, 16))  # Friday before MLK
        assert result == date(2026, 1, 20)  # Tuesday

    def test_from_regular_weekday(self):
        """Next trading day after Monday is Tuesday."""
        result = next_trading_day(date(2026, 2, 9))  # Monday
        assert result == date(2026, 2, 10)  # Tuesday


class TestTradingDaysBetween:
    def test_regular_week(self):
        """Mon-Fri should have 3 trading days between (exclusive)."""
        days = trading_days_between(date(2026, 2, 9), date(2026, 2, 13))
        assert len(days) == 3  # Tue, Wed, Thu
        assert date(2026, 2, 9) not in days  # exclusive start
        assert date(2026, 2, 13) not in days  # exclusive end

    def test_over_weekend(self):
        """No extra days for weekend."""
        days = trading_days_between(date(2026, 2, 6), date(2026, 2, 9))
        assert len(days) == 0  # Fri to Mon, both exclusive

    def test_over_holiday(self):
        """Holiday is excluded from trading days."""
        # MLK Day = Jan 19 2026 (Monday)
        days = trading_days_between(date(2026, 1, 16), date(2026, 1, 21))
        # Jan 16 (Fri) to Jan 21 (Wed), exclusive: only Tue Jan 20
        assert len(days) == 1
        assert days[0] == date(2026, 1, 20)

    def test_same_date(self):
        """Same start and end returns empty."""
        days = trading_days_between(date(2026, 2, 9), date(2026, 2, 9))
        assert days == []

    def test_end_before_start(self):
        """End before start returns empty."""
        days = trading_days_between(date(2026, 2, 13), date(2026, 2, 9))
        assert days == []
