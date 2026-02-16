"""Retry decorators for transient network failures.

Uses tenacity to retry on transient exceptions (ConnectionError, TimeoutError,
IOError, OSError) with exponential backoff. Non-transient exceptions (ValueError,
KeyError, etc.) are never retried.

IMPORTANT: Never apply retry decorators to order submission methods —
double-execution risk.
"""

import structlog
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = structlog.get_logger()

TRANSIENT_EXCEPTIONS = (ConnectionError, TimeoutError, IOError, OSError)


def _log_before_retry(retry_state: RetryCallState) -> None:
    """Log a warning before each retry attempt."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    log.warning(
        "retry_attempt",
        fn=retry_state.fn.__name__ if retry_state.fn else "unknown",
        attempt=retry_state.attempt_number,
        error=str(exc) if exc else "unknown",
    )


retry_market_data = retry(
    retry=retry_if_exception_type(TRANSIENT_EXCEPTIONS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
    before_sleep=_log_before_retry,
)

retry_news_api = retry(
    retry=retry_if_exception_type(TRANSIENT_EXCEPTIONS),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    reraise=True,
    before_sleep=_log_before_retry,
)

retry_macro_data = retry(
    retry=retry_if_exception_type(TRANSIENT_EXCEPTIONS),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    reraise=True,
    before_sleep=_log_before_retry,
)

retry_broker_read = retry(
    retry=retry_if_exception_type(TRANSIENT_EXCEPTIONS),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    reraise=True,
    before_sleep=_log_before_retry,
)
