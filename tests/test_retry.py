"""Tests for retry decorators in src/utils/retry.py."""

from unittest.mock import patch

import pytest

from src.utils.retry import (
    TRANSIENT_EXCEPTIONS,
    retry_broker_read,
    retry_market_data,
    retry_news_api,
)


class TestRetryMarketData:
    """Tests for retry_market_data decorator."""

    def test_transient_exception_triggers_retry(self):
        """ConnectionError should be retried up to 3 times."""
        call_count = 0

        @retry_market_data
        def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "success"

        result = flaky_fn()
        assert result == "success"
        assert call_count == 3

    def test_timeout_error_triggers_retry(self):
        """TimeoutError should be retried."""
        call_count = 0

        @retry_market_data
        def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("timeout")
            return "ok"

        assert flaky_fn() == "ok"
        assert call_count == 2

    def test_non_transient_exception_not_retried(self):
        """ValueError should NOT be retried — raised immediately."""
        call_count = 0

        @retry_market_data
        def bad_fn():
            nonlocal call_count
            call_count += 1
            raise ValueError("bad data")

        with pytest.raises(ValueError, match="bad data"):
            bad_fn()
        assert call_count == 1

    def test_key_error_not_retried(self):
        """KeyError should NOT be retried."""
        call_count = 0

        @retry_market_data
        def bad_fn():
            nonlocal call_count
            call_count += 1
            raise KeyError("missing")

        with pytest.raises(KeyError):
            bad_fn()
        assert call_count == 1

    def test_max_attempts_respected(self):
        """After 3 attempts, the transient exception is re-raised."""
        call_count = 0

        @retry_market_data
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("down")

        with pytest.raises(ConnectionError, match="down"):
            always_fails()
        assert call_count == 3

    def test_logging_callback_fires(self):
        """The before_sleep callback should log retry attempts."""
        call_count = 0

        @retry_market_data
        def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("transient")
            return "ok"

        with patch("src.utils.retry.log") as mock_log:
            flaky_fn()
            mock_log.warning.assert_called_once()
            call_args = mock_log.warning.call_args
            assert call_args[0][0] == "retry_attempt"


class TestRetryNewsApi:
    """Tests for retry_news_api decorator (2 attempts)."""

    def test_max_2_attempts(self):
        """retry_news_api should stop after 2 attempts."""
        call_count = 0

        @retry_news_api
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise IOError("network")

        with pytest.raises(IOError):
            always_fails()
        assert call_count == 2

    def test_success_on_second_attempt(self):
        """Should succeed if second attempt works."""
        call_count = 0

        @retry_news_api
        def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise OSError("flaky")
            return "articles"

        assert flaky_fn() == "articles"
        assert call_count == 2


class TestRetryBrokerRead:
    """Tests for retry_broker_read decorator."""

    def test_transient_retried(self):
        """Broker read retries on transient errors."""
        call_count = 0

        @retry_broker_read
        def fetch_order():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("reset")
            return {"status": "filled"}

        result = fetch_order()
        assert result == {"status": "filled"}
        assert call_count == 2


class TestAsyncRetry:
    """Tests that retry decorators work with async functions."""

    @pytest.mark.asyncio
    async def test_async_retry(self):
        """retry_market_data should work with async functions."""
        call_count = 0

        @retry_market_data
        async def async_flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("transient")
            return "async_ok"

        result = await async_flaky()
        assert result == "async_ok"
        assert call_count == 2


class TestTransientExceptions:
    """Verify all transient exception types are covered."""

    @pytest.mark.parametrize("exc_type", TRANSIENT_EXCEPTIONS)
    def test_all_transient_types_retried(self, exc_type):
        """Each transient exception type triggers retry."""
        call_count = 0

        @retry_market_data
        def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise exc_type("test")
            return "ok"

        assert fn() == "ok"
        assert call_count == 2
