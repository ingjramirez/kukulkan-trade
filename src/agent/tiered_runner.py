"""Tiered model runner: Haiku scan → Sonnet investigate → Opus validate.

Wraps the existing AgentRunner with pre-scan (Haiku) and post-validation (Opus)
steps. Session profile controls which steps run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from src.agent.agent_runner import AgentRunner
from src.agent.haiku_scanner import HaikuScanner, ScanResult
from src.agent.opus_validator import OpusValidator, ValidationResult
from src.agent.session_profiles import SessionProfile
from src.agent.token_tracker import MODEL_PRICING, TokenTracker

log = structlog.get_logger()


@dataclass
class TieredRunResult:
    """Result from a tiered model run — extends AgentRunResult metadata."""

    response: dict
    tool_calls: list = field(default_factory=list)
    turns: int = 0
    token_tracker: TokenTracker = field(default_factory=TokenTracker)
    raw_messages: list = field(default_factory=list)
    scan_result: ScanResult | None = None
    validation_result: ValidationResult | None = None
    session_profile: SessionProfile = SessionProfile.FULL
    skipped_investigation: bool = False


class TieredModelRunner:
    """Orchestrates Haiku → Sonnet → Opus tiered flow.

    Args:
        scanner: HaikuScanner instance for pre-scan.
        validator: OpusValidator instance for trade validation.
        agent_runner: AgentRunner (Sonnet) for investigation.
        token_tracker: Shared tracker for all model costs.
    """

    def __init__(
        self,
        scanner: HaikuScanner,
        validator: OpusValidator,
        agent_runner: AgentRunner,
        token_tracker: TokenTracker,
    ) -> None:
        self._scanner = scanner
        self._validator = validator
        self._runner = agent_runner
        self._token_tracker = token_tracker

    async def run(
        self,
        system_prompt: str | list[dict],
        user_message: str,
        session_profile: SessionProfile,
        market_data: dict,
        portfolio_summary: dict,
        posture: str = "balanced",
        messages_override: list[dict] | None = None,
    ) -> TieredRunResult:
        """Execute the tiered model pipeline.

        Args:
            system_prompt: System prompt (str or cached list[dict]).
            user_message: User message with trigger data.
            session_profile: Controls which models run.
            market_data: For Haiku scan.
            portfolio_summary: For Haiku scan.
            posture: Current agent posture.
            messages_override: For persistent agent context.

        Returns:
            TieredRunResult with all model outputs and metadata.
        """
        if session_profile == SessionProfile.BUDGET_SAVING:
            return await self._run_budget_saving(market_data, portfolio_summary, posture)
        elif session_profile == SessionProfile.CRISIS:
            return await self._run_crisis(system_prompt, user_message, messages_override)
        elif session_profile == SessionProfile.REVIEW:
            return await self._run_review(system_prompt, user_message, messages_override)
        elif session_profile == SessionProfile.LIGHT:
            return await self._run_light(
                system_prompt, user_message, market_data, portfolio_summary, posture, messages_override
            )
        else:  # FULL
            return await self._run_full(
                system_prompt, user_message, market_data, portfolio_summary, posture, messages_override
            )

    async def _run_budget_saving(
        self,
        market_data: dict,
        portfolio_summary: dict,
        posture: str,
    ) -> TieredRunResult:
        """Haiku scan only — no trades, just summary."""
        scan = await self._scanner.scan(market_data, portfolio_summary, posture)
        self._record_scan_cost(scan)

        return TieredRunResult(
            response={
                "regime_assessment": scan.summary,
                "reasoning": f"Budget saving mode. Scan verdict: {scan.verdict}. {scan.summary}",
                "trades": [],
                "risk_notes": "; ".join(scan.anomalies) if scan.anomalies else "No anomalies",
            },
            scan_result=scan,
            session_profile=SessionProfile.BUDGET_SAVING,
            skipped_investigation=True,
            token_tracker=self._token_tracker,
        )

    async def _run_crisis(
        self,
        system_prompt: str | list[dict],
        user_message: str,
        messages_override: list[dict] | None,
    ) -> TieredRunResult:
        """Skip scan, go straight to full Sonnet investigation."""
        result = await self._runner.run(
            system_prompt=system_prompt,
            user_message=user_message,
            messages_override=messages_override,
        )
        return TieredRunResult(
            response=result.response,
            tool_calls=result.tool_calls,
            turns=result.turns,
            token_tracker=self._token_tracker,
            raw_messages=result.raw_messages,
            session_profile=SessionProfile.CRISIS,
        )

    async def _run_review(
        self,
        system_prompt: str | list[dict],
        user_message: str,
        messages_override: list[dict] | None,
    ) -> TieredRunResult:
        """Skip scan and validation, Sonnet investigation only."""
        result = await self._runner.run(
            system_prompt=system_prompt,
            user_message=user_message,
            messages_override=messages_override,
        )
        return TieredRunResult(
            response=result.response,
            tool_calls=result.tool_calls,
            turns=result.turns,
            token_tracker=self._token_tracker,
            raw_messages=result.raw_messages,
            session_profile=SessionProfile.REVIEW,
        )

    async def _run_light(
        self,
        system_prompt: str | list[dict],
        user_message: str,
        market_data: dict,
        portfolio_summary: dict,
        posture: str,
        messages_override: list[dict] | None,
    ) -> TieredRunResult:
        """Haiku scan → skip if ROUTINE, else full pipeline."""
        scan = await self._scanner.scan(market_data, portfolio_summary, posture)
        self._record_scan_cost(scan)

        if scan.verdict == "ROUTINE":
            log.info("light_session_routine_skip", summary=scan.summary)
            return TieredRunResult(
                response={
                    "regime_assessment": scan.summary,
                    "reasoning": f"Light session — scan ROUTINE. {scan.summary}",
                    "trades": [],
                    "risk_notes": "No anomalies detected. Skipping investigation.",
                },
                scan_result=scan,
                session_profile=SessionProfile.LIGHT,
                skipped_investigation=True,
                token_tracker=self._token_tracker,
            )

        # INVESTIGATE or URGENT: proceed to full pipeline
        log.info("light_session_escalated", verdict=scan.verdict, anomalies=scan.anomalies)
        return await self._run_full_with_scan(
            system_prompt, user_message, market_data, portfolio_summary, posture, messages_override, scan
        )

    async def _run_full(
        self,
        system_prompt: str | list[dict],
        user_message: str,
        market_data: dict,
        portfolio_summary: dict,
        posture: str,
        messages_override: list[dict] | None,
    ) -> TieredRunResult:
        """Full pipeline: Haiku scan → Sonnet investigate → Opus validate."""
        scan = await self._scanner.scan(market_data, portfolio_summary, posture)
        self._record_scan_cost(scan)

        return await self._run_full_with_scan(
            system_prompt, user_message, market_data, portfolio_summary, posture, messages_override, scan
        )

    async def _run_full_with_scan(
        self,
        system_prompt: str | list[dict],
        user_message: str,
        market_data: dict,
        portfolio_summary: dict,
        posture: str,
        messages_override: list[dict] | None,
        scan: ScanResult,
    ) -> TieredRunResult:
        """Shared full pipeline after scan is complete."""
        # 2. Sonnet investigation
        result = await self._runner.run(
            system_prompt=system_prompt,
            user_message=user_message,
            messages_override=messages_override,
        )

        trades = result.response.get("trades", [])
        validation = None

        # 3. Opus validation (only if trades proposed)
        if trades:
            reasoning = result.response.get("reasoning", "")
            market_context = f"Regime: {market_data.get('regime', 'unknown')}, VIX: {market_data.get('vix', 'N/A')}"
            validation = await self._validator.validate(trades, reasoning, market_context, posture)
            self._record_validation_cost(validation)

            # 4. If Opus flags concerns — one more Sonnet turn
            if not validation.approved and validation.concerns:
                log.info("opus_concerns_followup", concerns=validation.concerns)
                concern_msg = (
                    "Risk review flagged concerns with your proposed trades:\n"
                    + "\n".join(f"- {c}" for c in validation.concerns)
                    + "\n\nPlease address these concerns and revise your trades if needed. "
                    "Respond with your final JSON output."
                )
                followup_messages = list(result.raw_messages)
                followup_messages.append({"role": "user", "content": concern_msg})

                followup = await self._runner.run(
                    system_prompt=system_prompt,
                    user_message=concern_msg,
                    messages_override=followup_messages,
                )
                result = followup

        return TieredRunResult(
            response=result.response,
            tool_calls=result.tool_calls,
            turns=result.turns,
            token_tracker=self._token_tracker,
            raw_messages=result.raw_messages,
            scan_result=scan,
            validation_result=validation,
            session_profile=SessionProfile.FULL,
        )

    def _record_scan_cost(self, scan: ScanResult) -> None:
        """Record Haiku scan cost on the shared token tracker."""
        if scan.cost_usd > 0:
            # Estimate tokens from cost for the tracker
            pricing = MODEL_PRICING.get("claude-haiku-4-5-20251001", (1.0, 5.0))
            # Approximate: assume 80% input, 20% output by cost
            est_input = int(scan.cost_usd * 0.8 * 1_000_000 / pricing[0]) if pricing[0] > 0 else 0
            est_output = int(scan.cost_usd * 0.2 * 1_000_000 / pricing[1]) if pricing[1] > 0 else 0
            self._token_tracker.record(
                model="claude-haiku-4-5-20251001",
                input_tokens=est_input,
                output_tokens=est_output,
                turn=0,  # Pre-investigation turn
            )

    def _record_validation_cost(self, validation: ValidationResult) -> None:
        """Record Opus validation cost on the shared token tracker."""
        if validation.cost_usd > 0:
            pricing = MODEL_PRICING.get("claude-opus-4-6", (5.0, 25.0))
            est_input = int(validation.cost_usd * 0.8 * 1_000_000 / pricing[0]) if pricing[0] > 0 else 0
            est_output = int(validation.cost_usd * 0.2 * 1_000_000 / pricing[1]) if pricing[1] > 0 else 0
            self._token_tracker.record(
                model="claude-opus-4-6",
                input_tokens=est_input,
                output_tokens=est_output,
                turn=99,  # Post-investigation turn
            )
