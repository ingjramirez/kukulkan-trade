"""Opus-based trade validator for the tiered model runner.

Reviews proposed trades for risk quality before execution.
Only called when Sonnet proposes trades — no-trade sessions skip validation.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger()

VALIDATE_MAX_TOKENS = 1000

VALIDATE_SYSTEM_PROMPT = (
    "You are a senior risk reviewer for an educational trading bot.\n\n"
    "Review proposed trades for:\n"
    "1. Portfolio quality — diversification, sector concentration, position sizing\n"
    "2. Risk alignment — do trades match the stated market regime and posture?\n"
    "3. Conviction quality — is the reasoning sound and supported by data?\n"
    "4. Timing concerns — earnings, ex-dividend dates, market events\n\n"
    "Approve trades that pass review. Flag specific concerns for any that don't.\n\n"
    "Respond with ONLY a JSON object:\n"
    '{"approved": true/false, "concerns": ["specific concern 1", "concern 2"], '
    '"summary": "one sentence overall assessment"}'
)


@dataclass(frozen=True)
class ValidationResult:
    """Result from Opus trade validation."""

    approved: bool
    concerns: list[str] = field(default_factory=list)
    summary: str = ""
    cost_usd: float = 0.0


class OpusValidator:
    """Validates proposed trades using Opus before execution."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @property
    def _validate_model(self) -> str:
        """Get validate model from settings, with fallback."""
        try:
            from config.settings import settings

            return settings.agent.validate_model
        except Exception:
            return "claude-opus-4-6"

    def _get_client(self):
        """Lazy client initialization."""
        import anthropic

        from config.settings import settings

        return anthropic.Anthropic(api_key=self._api_key, max_retries=settings.agent.max_retries)

    async def validate(
        self,
        trades: list[dict],
        reasoning: str,
        market_context: str,
        posture: str = "balanced",
    ) -> ValidationResult:
        """Validate proposed trades with Opus.

        Args:
            trades: List of proposed trade dicts from Sonnet.
            reasoning: Sonnet's reasoning for the trades.
            market_context: Current market regime and data summary.
            posture: Current agent posture.

        Returns:
            ValidationResult with approval status and concerns.
        """
        if not trades:
            return ValidationResult(approved=True, summary="No trades to validate")

        user_message = (
            f"Proposed trades:\n{json.dumps(trades, default=str, indent=2)}\n\n"
            f"Reasoning: {reasoning}\n\n"
            f"Market context: {market_context}\n\n"
            f"Current posture: {posture}"
        )

        try:
            validate_model = self._validate_model
            client = self._get_client()
            response = await asyncio.to_thread(
                client.messages.create,
                model=validate_model,
                max_tokens=VALIDATE_MAX_TOKENS,
                system=VALIDATE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text = block.text
                    break

            # Estimate cost
            from src.agent.token_tracker import MODEL_PRICING

            pricing = MODEL_PRICING.get(validate_model, (5.0, 25.0))
            cost = (response.usage.input_tokens * pricing[0] + response.usage.output_tokens * pricing[1]) / 1_000_000

            return self._parse_response(text, cost)

        except Exception as e:
            log.warning("opus_validation_failed", error=str(e))
            # Fail open: approve trades if validation itself fails
            return ValidationResult(
                approved=True,
                concerns=[f"Validation failed: {e}"],
                summary="Validation error, approving by default",
                cost_usd=0.0,
            )

    @staticmethod
    def _parse_response(text: str, cost: float) -> ValidationResult:
        """Parse Opus's JSON response."""
        try:
            cleaned = text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            data = json.loads(cleaned)
            return ValidationResult(
                approved=bool(data.get("approved", True)),
                concerns=data.get("concerns", []),
                summary=data.get("summary", ""),
                cost_usd=cost,
            )
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            log.warning("opus_validation_parse_failed", error=str(e), raw=text[:200])
            return ValidationResult(
                approved=True,
                concerns=["Failed to parse validation response"],
                summary="Parse error, approving by default",
                cost_usd=cost,
            )
