"""Tests for SessionCompressor — Haiku compression and Sonnet validation."""

from unittest.mock import MagicMock, patch

import pytest

from src.agent.session_compressor import (
    CompressionError,
    CompressionValidation,
    SessionCompressor,
    run_compression_validation,
)

SAMPLE_MESSAGES = [
    {"role": "user", "content": "Good morning. VIX 18.2. Regime: BULL."},
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me check the portfolio."},
            {"type": "tool_use", "id": "t1", "name": "get_portfolio_state", "input": {}},
        ],
    },
    {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": '{"cash": 25000, "positions": 4}'}],
    },
    {
        "role": "assistant",
        "content": "Portfolio healthy. Buying NVDA 50 shares at $118 — half-size position. "
        "Plan to add Wednesday if $115 support holds. Setting 7% trailing stop.",
    },
]


def _mock_response(text: str):
    """Create a mock Anthropic response with text content."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


@pytest.fixture
def compressor():
    return SessionCompressor(api_key="test-key")


async def test_compress_returns_summary(compressor: SessionCompressor):
    """compress() returns summary text from Haiku."""
    mock_summary = (
        "Morning session in BULL regime (VIX 18.2). Portfolio had $25K cash, 4 positions. "
        "Bought NVDA 50 shares at $118 as half-size position. Plan to add Wednesday if "
        "$115 support holds. Set 7% trailing stop."
    )

    with patch.object(compressor, "_get_client") as mock_client:
        mock_client.return_value.messages.create.return_value = _mock_response(mock_summary)
        result = await compressor.compress(SAMPLE_MESSAGES)

    assert "NVDA" in result
    assert "$118" in result
    assert "50 shares" in result


async def test_compress_preserves_trade_details(compressor: SessionCompressor):
    """Mock Haiku summary preserves trade details."""
    summary = "Bought NVDA 50 shares at $118.00. Set 7% trailing stop at $109.74."

    with patch.object(compressor, "_get_client") as mock_client:
        mock_client.return_value.messages.create.return_value = _mock_response(summary)
        result = await compressor.compress(SAMPLE_MESSAGES)

    assert "NVDA" in result
    assert "$118" in result


async def test_compress_preserves_multi_day_plans(compressor: SessionCompressor):
    """Mock Haiku summary preserves multi-day plans."""
    summary = "Bought NVDA half-size. Plan to add Wednesday if $115 support holds."

    with patch.object(compressor, "_get_client") as mock_client:
        mock_client.return_value.messages.create.return_value = _mock_response(summary)
        result = await compressor.compress(SAMPLE_MESSAGES)

    assert "Wednesday" in result
    assert "$115" in result


async def test_compress_handles_empty_session(compressor: SessionCompressor):
    """compress() raises CompressionError for empty sessions."""
    with pytest.raises(CompressionError, match="empty session"):
        await compressor.compress([])


async def test_compress_handles_haiku_empty_response(compressor: SessionCompressor):
    """compress() raises CompressionError if Haiku returns empty text."""
    with patch.object(compressor, "_get_client") as mock_client:
        mock_client.return_value.messages.create.return_value = _mock_response("")
        with pytest.raises(CompressionError, match="empty summary"):
            await compressor.compress(SAMPLE_MESSAGES)


async def test_validate_compression_passes_good_summary(compressor: SessionCompressor):
    """validate_compression returns acceptable for good summaries."""
    validation_json = '{"missing_facts": [], "fidelity_score": 0.98, "assessment": "Good"}'

    with patch.object(compressor, "_get_client") as mock_client:
        mock_client.return_value.messages.create.return_value = _mock_response(validation_json)
        result = await compressor.validate_compression(
            SAMPLE_MESSAGES,
            "Bought NVDA 50 at $118. Plan to add Wed if $115 holds. 7% stop.",
        )

    assert isinstance(result, CompressionValidation)
    assert result.fidelity_score == 0.98
    assert result.is_acceptable is True
    assert result.missing_facts == []


async def test_validate_compression_detects_missing_facts(compressor: SessionCompressor):
    """validate_compression identifies missing trading facts."""
    validation_json = (
        '{"missing_facts": ["Missing trailing stop level of 7%"], '
        '"fidelity_score": 0.85, "assessment": "Acceptable"}'
    )

    with patch.object(compressor, "_get_client") as mock_client:
        mock_client.return_value.messages.create.return_value = _mock_response(validation_json)
        result = await compressor.validate_compression(
            SAMPLE_MESSAGES,
            "Bought NVDA 50 at $118. Plan to add Wed.",
        )

    assert result.fidelity_score == 0.85
    assert result.is_acceptable is False
    assert len(result.missing_facts) == 1
    assert "trailing stop" in result.missing_facts[0].lower()


async def test_validate_handles_malformed_json(compressor: SessionCompressor):
    """validate_compression handles malformed JSON from Sonnet."""
    with patch.object(compressor, "_get_client") as mock_client:
        mock_client.return_value.messages.create.return_value = _mock_response("not json at all")
        result = await compressor.validate_compression(SAMPLE_MESSAGES, "summary")

    assert result.fidelity_score == 0.0
    assert result.is_acceptable is False


async def test_compression_validation_gate_aggregate_score(compressor: SessionCompressor):
    """run_compression_validation computes aggregate fidelity score."""
    sessions = [
        {"session_id": "s1", "messages": SAMPLE_MESSAGES},
        {"session_id": "s2", "messages": SAMPLE_MESSAGES},
    ]

    call_count = 0

    def mock_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 1:
            # Compression call (odd)
            return _mock_response("Summary of session.")
        else:
            # Validation call (even)
            return _mock_response('{"missing_facts": [], "fidelity_score": 0.96, "assessment": "Good"}')

    with patch.object(compressor, "_get_client") as mock_client:
        mock_client.return_value.messages.create.side_effect = mock_create
        result = await run_compression_validation(compressor, sessions)

    assert len(result["results"]) == 2
    assert result["avg_fidelity"] == 0.96
    assert result["all_acceptable"] is True


def test_format_messages_for_compression():
    """_format_messages_for_compression produces readable text."""
    text = SessionCompressor._format_messages_for_compression(SAMPLE_MESSAGES)
    assert "[user]:" in text
    assert "[assistant]:" in text or "[assistant → tool:" in text
    assert "VIX 18.2" in text
    assert "NVDA" in text
