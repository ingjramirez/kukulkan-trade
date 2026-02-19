"""Proactive rate-limit pacer using a sliding-window token tracker.

Prevents 429 errors by tracking token consumption within a 60-second
sliding window and sleeping when the estimated next request would
exceed the tokens-per-minute limit.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque

import structlog

log = structlog.get_logger()

# Pacing bounds
_MIN_WAIT_SECS = 1.0
_MAX_WAIT_SECS = 60.0


class RequestPacer:
    """Proactive rate-limit pacer using a sliding-window token tracker."""

    def __init__(self, tokens_per_minute: int = 25_500) -> None:
        self._tpm_limit = tokens_per_minute
        self._window: deque[tuple[float, int]] = deque()  # (timestamp, tokens)

    def _prune(self, now: float) -> None:
        """Remove entries older than 60 seconds."""
        cutoff = now - 60.0
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

    def _window_total(self) -> int:
        """Sum tokens in the current window."""
        return sum(tokens for _, tokens in self._window)

    async def wait_if_needed(self, estimated_tokens: int) -> float:
        """Wait until the sliding window has capacity for estimated_tokens.

        Returns seconds waited (0.0 if no wait needed).
        """
        now = time.monotonic()
        self._prune(now)

        current = self._window_total()
        if current + estimated_tokens <= self._tpm_limit:
            return 0.0

        # Calculate how long to wait for oldest entries to expire
        needed = current + estimated_tokens - self._tpm_limit
        wait_secs = 0.0

        for ts, tokens in self._window:
            wait_secs = ts + 60.0 - now
            needed -= tokens
            if needed <= 0:
                break

        wait_secs = max(_MIN_WAIT_SECS, min(wait_secs, _MAX_WAIT_SECS))
        log.info(
            "request_pacer_waiting", wait_secs=round(wait_secs, 1), window_tokens=current, estimated=estimated_tokens
        )
        await asyncio.sleep(wait_secs)
        return wait_secs

    def record(self, actual_tokens: int) -> None:
        """Record actual tokens consumed after API response."""
        self._window.append((time.monotonic(), actual_tokens))

    @staticmethod
    def estimate_tokens(messages: list[dict], system_prompt: str | list[dict] = "") -> int:
        """Estimate input tokens from messages + system prompt. ~4 chars/token."""
        total_chars = 0

        # System prompt
        if isinstance(system_prompt, str):
            total_chars += len(system_prompt)
        elif isinstance(system_prompt, list):
            for block in system_prompt:
                if isinstance(block, dict):
                    total_chars += len(block.get("text", ""))

        # Messages
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text", "") or block.get("content", "")
                        if isinstance(text, str):
                            total_chars += len(text)
                        inp = block.get("input")
                        if isinstance(inp, dict):
                            total_chars += len(str(inp))

        return total_chars // 4
