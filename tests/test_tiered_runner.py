"""Tests for TieredModelRunner — Haiku→Sonnet→Opus flow."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.agent_runner import AgentRunResult
from src.agent.haiku_scanner import ScanResult
from src.agent.opus_validator import ValidationResult
from src.agent.session_profiles import SessionProfile
from src.agent.tiered_runner import TieredModelRunner
from src.agent.token_tracker import TokenTracker


@pytest.fixture
def token_tracker():
    return TokenTracker(session_budget_usd=5.0)


@pytest.fixture
def mock_scanner():
    scanner = MagicMock()
    scanner.scan = AsyncMock()
    return scanner


@pytest.fixture
def mock_validator():
    validator = MagicMock()
    validator.validate = AsyncMock()
    return validator


@pytest.fixture
def mock_runner():
    runner = MagicMock()
    runner.run = AsyncMock()
    return runner


@pytest.fixture
def tiered(mock_scanner, mock_validator, mock_runner, token_tracker):
    return TieredModelRunner(
        scanner=mock_scanner,
        validator=mock_validator,
        agent_runner=mock_runner,
        token_tracker=token_tracker,
    )


def _agent_result(trades: list | None = None, reasoning: str = "test reasoning") -> AgentRunResult:
    return AgentRunResult(
        response={
            "regime_assessment": "BULL",
            "reasoning": reasoning,
            "trades": trades or [],
            "risk_notes": "",
        },
        tool_calls=[],
        turns=2,
        token_tracker=TokenTracker(),
        raw_messages=[{"role": "user", "content": "test"}],
    )


def _routine_scan() -> ScanResult:
    return ScanResult(verdict="ROUTINE", anomalies=[], summary="All normal", cost_usd=0.001)


def _investigate_scan() -> ScanResult:
    return ScanResult(verdict="INVESTIGATE", anomalies=["NVDA -3%"], summary="Position down", cost_usd=0.001)


def _urgent_scan() -> ScanResult:
    return ScanResult(verdict="URGENT", anomalies=["VIX spike"], summary="Market stress", cost_usd=0.001)


# ── FULL profile ─────────────────────────────────────────────────────────────


class TestFullProfile:
    @pytest.mark.asyncio
    async def test_full_no_trades_skips_validation(self, tiered, mock_scanner, mock_runner, mock_validator):
        mock_scanner.scan.return_value = _routine_scan()
        mock_runner.run.return_value = _agent_result(trades=[])

        result = await tiered.run(
            system_prompt="test",
            user_message="test",
            session_profile=SessionProfile.FULL,
            market_data={"regime": "BULL"},
            portfolio_summary={"cash": 10000},
        )
        assert result.scan_result is not None
        assert result.validation_result is None  # No trades → no validation
        mock_validator.validate.assert_not_called()

    @pytest.mark.asyncio
    async def test_full_with_trades_calls_validation(self, tiered, mock_scanner, mock_runner, mock_validator):
        mock_scanner.scan.return_value = _investigate_scan()
        trades = [{"ticker": "NVDA", "side": "BUY", "shares": 10}]
        mock_runner.run.return_value = _agent_result(trades=trades)
        mock_validator.validate.return_value = ValidationResult(approved=True, summary="OK", cost_usd=0.02)

        result = await tiered.run(
            system_prompt="test",
            user_message="test",
            session_profile=SessionProfile.FULL,
            market_data={"regime": "BULL"},
            portfolio_summary={"cash": 10000},
        )
        assert result.scan_result is not None
        assert result.validation_result is not None
        assert result.validation_result.approved is True
        mock_validator.validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_full_opus_concerns_trigger_followup(self, tiered, mock_scanner, mock_runner, mock_validator):
        mock_scanner.scan.return_value = _investigate_scan()
        trades = [{"ticker": "NVDA", "side": "BUY", "shares": 100}]
        mock_runner.run.return_value = _agent_result(trades=trades)
        mock_validator.validate.return_value = ValidationResult(
            approved=False, concerns=["Too concentrated"], summary="Risky", cost_usd=0.02
        )

        # Second run returns revised trades
        revised = _agent_result(trades=[{"ticker": "NVDA", "side": "BUY", "shares": 20}])
        mock_runner.run.side_effect = [_agent_result(trades=trades), revised]

        await tiered.run(
            system_prompt="test",
            user_message="test",
            session_profile=SessionProfile.FULL,
            market_data={"regime": "BULL"},
            portfolio_summary={"cash": 10000},
        )
        # Runner called twice (original + followup)
        assert mock_runner.run.call_count == 2


# ── LIGHT profile ────────────────────────────────────────────────────────────


class TestLightProfile:
    @pytest.mark.asyncio
    async def test_light_routine_skips_investigation(self, tiered, mock_scanner, mock_runner):
        mock_scanner.scan.return_value = _routine_scan()

        result = await tiered.run(
            system_prompt="test",
            user_message="test",
            session_profile=SessionProfile.LIGHT,
            market_data={"regime": "BULL"},
            portfolio_summary={"cash": 10000},
        )
        assert result.skipped_investigation is True
        assert result.response["trades"] == []
        mock_runner.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_light_investigate_escalates_to_full(self, tiered, mock_scanner, mock_runner, mock_validator):
        mock_scanner.scan.return_value = _investigate_scan()
        mock_runner.run.return_value = _agent_result(trades=[])

        result = await tiered.run(
            system_prompt="test",
            user_message="test",
            session_profile=SessionProfile.LIGHT,
            market_data={"regime": "BULL"},
            portfolio_summary={"cash": 10000},
        )
        assert result.skipped_investigation is False
        mock_runner.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_light_urgent_escalates_to_full(self, tiered, mock_scanner, mock_runner, mock_validator):
        mock_scanner.scan.return_value = _urgent_scan()
        mock_runner.run.return_value = _agent_result(trades=[])

        result = await tiered.run(
            system_prompt="test",
            user_message="test",
            session_profile=SessionProfile.LIGHT,
            market_data={"regime": "BEAR"},
            portfolio_summary={"cash": 5000},
        )
        assert result.skipped_investigation is False
        mock_runner.run.assert_called_once()


# ── CRISIS profile ───────────────────────────────────────────────────────────


class TestCrisisProfile:
    @pytest.mark.asyncio
    async def test_crisis_skips_scan(self, tiered, mock_scanner, mock_runner):
        mock_runner.run.return_value = _agent_result(trades=[])

        result = await tiered.run(
            system_prompt="test",
            user_message="test",
            session_profile=SessionProfile.CRISIS,
            market_data={"regime": "BEAR"},
            portfolio_summary={"cash": 5000},
        )
        assert result.scan_result is None
        mock_scanner.scan.assert_not_called()
        mock_runner.run.assert_called_once()


# ── REVIEW profile ───────────────────────────────────────────────────────────


class TestReviewProfile:
    @pytest.mark.asyncio
    async def test_review_skips_scan_and_validation(self, tiered, mock_scanner, mock_runner, mock_validator):
        mock_runner.run.return_value = _agent_result(trades=[])

        result = await tiered.run(
            system_prompt="test",
            user_message="test",
            session_profile=SessionProfile.REVIEW,
            market_data={},
            portfolio_summary={},
        )
        assert result.scan_result is None
        assert result.validation_result is None
        mock_scanner.scan.assert_not_called()
        mock_validator.validate.assert_not_called()
        mock_runner.run.assert_called_once()


# ── BUDGET_SAVING profile ───────────────────────────────────────────────────


class TestBudgetSavingProfile:
    @pytest.mark.asyncio
    async def test_budget_saving_haiku_only(self, tiered, mock_scanner, mock_runner, mock_validator):
        mock_scanner.scan.return_value = _investigate_scan()

        result = await tiered.run(
            system_prompt="test",
            user_message="test",
            session_profile=SessionProfile.BUDGET_SAVING,
            market_data={"regime": "BULL"},
            portfolio_summary={"cash": 10000},
        )
        assert result.skipped_investigation is True
        assert result.response["trades"] == []
        mock_runner.run.assert_not_called()
        mock_validator.validate.assert_not_called()


# ── Token tracking ───────────────────────────────────────────────────────────


class TestTokenTracking:
    @pytest.mark.asyncio
    async def test_scan_cost_recorded(self, tiered, mock_scanner, mock_runner, token_tracker):
        mock_scanner.scan.return_value = _routine_scan()

        await tiered.run(
            system_prompt="test",
            user_message="test",
            session_profile=SessionProfile.BUDGET_SAVING,
            market_data={"regime": "BULL"},
            portfolio_summary={"cash": 10000},
        )
        # Token tracker should have at least one entry from the scan
        assert token_tracker.total_cost_usd > 0
