"""Tests for HaikuScanner — market triage using Haiku."""

from unittest.mock import MagicMock, patch

import pytest

from src.agent.haiku_scanner import HaikuScanner, ScanResult


@pytest.fixture
def scanner():
    return HaikuScanner(api_key="test-key")


def _mock_response(text: str, input_tokens: int = 500, output_tokens: int = 200):
    """Build a mock Anthropic response."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    resp.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return resp


class TestScanResult:
    def test_routine_result(self):
        r = ScanResult(verdict="ROUTINE", anomalies=[], summary="All normal")
        assert r.verdict == "ROUTINE"
        assert r.anomalies == []

    def test_frozen_dataclass(self):
        r = ScanResult(verdict="ROUTINE")
        with pytest.raises(AttributeError):
            r.verdict = "INVESTIGATE"


class TestHaikuScannerParsing:
    def test_parse_routine(self, scanner):
        text = '{"verdict": "ROUTINE", "anomalies": [], "summary": "Markets stable"}'
        result = scanner._parse_response(text, 0.001)
        assert result.verdict == "ROUTINE"
        assert result.anomalies == []
        assert result.summary == "Markets stable"
        assert result.cost_usd == 0.001

    def test_parse_investigate(self, scanner):
        text = '{"verdict": "INVESTIGATE", "anomalies": ["NVDA -3%"], "summary": "Position down"}'
        result = scanner._parse_response(text, 0.002)
        assert result.verdict == "INVESTIGATE"
        assert len(result.anomalies) == 1

    def test_parse_urgent(self, scanner):
        text = '{"verdict": "URGENT", "anomalies": ["VIX spike", "SPY -4%"], "summary": "Selloff"}'
        result = scanner._parse_response(text, 0.002)
        assert result.verdict == "URGENT"
        assert len(result.anomalies) == 2

    def test_parse_with_markdown_fences(self, scanner):
        text = '```json\n{"verdict": "ROUTINE", "anomalies": [], "summary": "OK"}\n```'
        result = scanner._parse_response(text, 0.001)
        assert result.verdict == "ROUTINE"

    def test_parse_invalid_verdict_defaults_to_investigate(self, scanner):
        text = '{"verdict": "MAYBE", "anomalies": [], "summary": ""}'
        result = scanner._parse_response(text, 0.001)
        assert result.verdict == "INVESTIGATE"

    def test_parse_malformed_json_defaults_to_investigate(self, scanner):
        text = "This is not JSON at all"
        result = scanner._parse_response(text, 0.001)
        assert result.verdict == "INVESTIGATE"
        assert len(result.anomalies) == 1  # "Failed to parse scan response"


class TestHaikuScannerScan:
    @pytest.mark.asyncio
    async def test_scan_routine(self, scanner):
        mock_resp = _mock_response('{"verdict": "ROUTINE", "anomalies": [], "summary": "OK"}')
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_resp

        with patch.object(scanner, "_get_client", return_value=mock_client):
            result = await scanner.scan(
                market_data={"regime": "BULL", "vix": 15},
                portfolio_summary={"cash": 10000, "positions_count": 3},
            )
        assert result.verdict == "ROUTINE"
        assert result.cost_usd > 0

    @pytest.mark.asyncio
    async def test_scan_failure_defaults_to_investigate(self, scanner):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API error")

        with patch.object(scanner, "_get_client", return_value=mock_client):
            result = await scanner.scan(
                market_data={"regime": "BULL"},
                portfolio_summary={"cash": 5000},
            )
        assert result.verdict == "INVESTIGATE"
        assert result.cost_usd == 0.0
