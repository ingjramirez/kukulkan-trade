"""Tests for OpusValidator — trade review using Opus."""

from unittest.mock import MagicMock, patch

import pytest

from src.agent.opus_validator import OpusValidator, ValidationResult


@pytest.fixture
def validator():
    return OpusValidator(api_key="test-key")


def _mock_response(text: str, input_tokens: int = 2000, output_tokens: int = 500):
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    resp.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return resp


class TestValidationResult:
    def test_approved_result(self):
        r = ValidationResult(approved=True, summary="All good")
        assert r.approved is True

    def test_rejected_result(self):
        r = ValidationResult(approved=False, concerns=["Too concentrated"], summary="Risky")
        assert r.approved is False
        assert len(r.concerns) == 1

    def test_frozen_dataclass(self):
        r = ValidationResult(approved=True)
        with pytest.raises(AttributeError):
            r.approved = False


class TestOpusValidatorParsing:
    def test_parse_approved(self, validator):
        text = '{"approved": true, "concerns": [], "summary": "Trades look good"}'
        result = validator._parse_response(text, 0.02)
        assert result.approved is True
        assert result.concerns == []
        assert result.cost_usd == 0.02

    def test_parse_rejected(self, validator):
        text = '{"approved": false, "concerns": ["Sector too concentrated"], "summary": "Needs revision"}'
        result = validator._parse_response(text, 0.02)
        assert result.approved is False
        assert len(result.concerns) == 1

    def test_parse_with_markdown_fences(self, validator):
        text = '```json\n{"approved": true, "concerns": [], "summary": "OK"}\n```'
        result = validator._parse_response(text, 0.01)
        assert result.approved is True

    def test_parse_malformed_json_approves_by_default(self, validator):
        text = "Not valid JSON"
        result = validator._parse_response(text, 0.01)
        assert result.approved is True
        assert len(result.concerns) == 1  # "Failed to parse"


class TestOpusValidatorValidate:
    @pytest.mark.asyncio
    async def test_skip_when_no_trades(self, validator):
        result = await validator.validate(trades=[], reasoning="No trades", market_context="BULL", posture="balanced")
        assert result.approved is True
        assert result.summary == "No trades to validate"

    @pytest.mark.asyncio
    async def test_validate_with_trades(self, validator):
        mock_resp = _mock_response('{"approved": true, "concerns": [], "summary": "Looks good"}')
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_resp

        with patch.object(validator, "_get_client", return_value=mock_client):
            result = await validator.validate(
                trades=[{"ticker": "NVDA", "side": "BUY", "shares": 10}],
                reasoning="Strong momentum",
                market_context="Regime: BULL, VIX: 15",
            )
        assert result.approved is True
        assert result.cost_usd > 0

    @pytest.mark.asyncio
    async def test_validate_failure_approves_by_default(self, validator):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API error")

        with patch.object(validator, "_get_client", return_value=mock_client):
            result = await validator.validate(
                trades=[{"ticker": "AAPL", "side": "BUY", "shares": 5}],
                reasoning="Earnings play",
                market_context="BULL",
            )
        assert result.approved is True  # Fail open
        assert result.cost_usd == 0.0
