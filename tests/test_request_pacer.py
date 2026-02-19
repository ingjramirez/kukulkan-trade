"""Tests for RequestPacer — sliding-window rate-limit pacer."""

import time
from unittest.mock import AsyncMock, patch

import pytest

from src.agent.request_pacer import _MAX_WAIT_SECS, _MIN_WAIT_SECS, RequestPacer


@pytest.mark.asyncio
async def test_no_wait_under_limit():
    """No wait when estimated tokens fit within the window."""
    pacer = RequestPacer(tokens_per_minute=30_000)
    wait = await pacer.wait_if_needed(10_000)
    assert wait == 0.0


@pytest.mark.asyncio
async def test_waits_when_near_limit():
    """Pacer waits when window is nearly full."""
    pacer = RequestPacer(tokens_per_minute=10_000)
    # Fill the window
    pacer.record(8_000)

    with patch("src.agent.request_pacer.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        wait = await pacer.wait_if_needed(5_000)

    assert wait > 0
    mock_sleep.assert_called_once()


@pytest.mark.asyncio
async def test_waits_until_window_expires():
    """Pacer calculates wait time based on oldest entry expiry."""
    pacer = RequestPacer(tokens_per_minute=10_000)

    # Inject an old entry that's 50 seconds old
    old_ts = time.monotonic() - 50.0
    pacer._window.append((old_ts, 8_000))

    with patch("src.agent.request_pacer.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        wait = await pacer.wait_if_needed(5_000)

    # Should wait ~10 seconds (60 - 50) for the old entry to expire
    assert wait > 0
    assert mock_sleep.call_args[0][0] <= 15.0  # within bounds


def test_record_tracks_usage():
    """record() adds entries to the sliding window."""
    pacer = RequestPacer()
    pacer.record(5_000)
    pacer.record(3_000)
    assert len(pacer._window) == 2
    assert pacer._window_total() == 8_000


def test_estimate_tokens_string_system():
    """Estimate tokens from string system prompt + messages."""
    tokens = RequestPacer.estimate_tokens(
        messages=[{"role": "user", "content": "A" * 400}],
        system_prompt="B" * 800,
    )
    # (400 + 800) / 4 = 300
    assert tokens == 300


def test_estimate_tokens_list_system():
    """Estimate tokens from list-based system prompt (cache blocks)."""
    tokens = RequestPacer.estimate_tokens(
        messages=[{"role": "user", "content": "A" * 400}],
        system_prompt=[{"type": "text", "text": "B" * 800}],
    )
    assert tokens == 300


def test_estimate_tokens_tool_blocks():
    """Estimate handles tool_use and tool_result content blocks."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "X" * 100},
                {"type": "tool_use", "id": "t1", "name": "get_price", "input": {"ticker": "NVDA"}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Y" * 200}],
        },
    ]
    tokens = RequestPacer.estimate_tokens(messages)
    assert tokens > 0


def test_old_entries_pruned():
    """Entries older than 60s are pruned."""
    pacer = RequestPacer(tokens_per_minute=10_000)
    # Inject an entry from 61 seconds ago
    old_ts = time.monotonic() - 61.0
    pacer._window.append((old_ts, 5_000))
    pacer._prune(time.monotonic())
    assert len(pacer._window) == 0


@pytest.mark.asyncio
async def test_minimum_wait():
    """Wait time is at least _MIN_WAIT_SECS."""
    pacer = RequestPacer(tokens_per_minute=10_000)
    # Inject an entry that's 59.5 seconds old — would expire in ~0.5s
    old_ts = time.monotonic() - 59.5
    pacer._window.append((old_ts, 9_000))

    with patch("src.agent.request_pacer.asyncio.sleep", new_callable=AsyncMock):
        wait = await pacer.wait_if_needed(5_000)

    assert wait >= _MIN_WAIT_SECS


@pytest.mark.asyncio
async def test_maximum_wait():
    """Wait time is capped at _MAX_WAIT_SECS."""
    pacer = RequestPacer(tokens_per_minute=100)  # very low limit
    # Fill with recent entries
    now = time.monotonic()
    for i in range(10):
        pacer._window.append((now - i * 0.1, 50))

    with patch("src.agent.request_pacer.asyncio.sleep", new_callable=AsyncMock):
        wait = await pacer.wait_if_needed(200)

    assert wait <= _MAX_WAIT_SECS


def test_safety_margin_applied():
    """Default TPM is 85% of 30K = 25500."""
    pacer = RequestPacer()
    assert pacer._tpm_limit == 25_500
