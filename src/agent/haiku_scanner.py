"""Haiku-based market scanner for the tiered model runner.

Performs a fast, low-cost scan of market conditions to decide
whether a full Sonnet investigation is needed.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger()

SCAN_MODEL = "claude-haiku-4-5-20251001"
SCAN_MAX_TOKENS = 500

SCAN_SYSTEM_PROMPT = (
    "You are a market scanner. Your job is to quickly classify the current trading session.\n\n"
    "Classify as one of:\n"
    "- ROUTINE: No significant moves, no action needed. Market is within normal ranges.\n"
    "- INVESTIGATE: Some anomalies detected that warrant investigation "
    "(>2% moves in held positions, sector rotation signals, unusual volume, VIX spikes).\n"
    "- URGENT: Significant market event requiring immediate attention "
    "(circuit breakers, earnings surprises on held positions, regime change).\n\n"
    "Respond with ONLY a JSON object:\n"
    '{"verdict": "ROUTINE|INVESTIGATE|URGENT", "anomalies": ["list of anomalies if any"], '
    '"summary": "one sentence summary"}'
)


@dataclass(frozen=True)
class ScanResult:
    """Result from a Haiku market scan."""

    verdict: str  # ROUTINE, INVESTIGATE, URGENT
    anomalies: list[str] = field(default_factory=list)
    summary: str = ""
    cost_usd: float = 0.0


class HaikuScanner:
    """Fast market scanner using Haiku for triage."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def _get_client(self):
        """Lazy client initialization."""
        import anthropic

        from config.settings import settings

        return anthropic.Anthropic(api_key=self._api_key, max_retries=settings.agent.max_retries)

    async def scan(
        self,
        market_data: dict,
        portfolio_summary: dict,
        posture: str = "balanced",
    ) -> ScanResult:
        """Run a fast market scan to classify the session.

        Args:
            market_data: Dict with regime, vix, spy data, etc.
            portfolio_summary: Dict with positions, cash, P&L.
            posture: Current agent posture (balanced/defensive/aggressive).

        Returns:
            ScanResult with verdict, anomalies, and cost.
        """
        user_message = (
            f"Market: {json.dumps(market_data, default=str)}\n"
            f"Portfolio: {json.dumps(portfolio_summary, default=str)}\n"
            f"Current posture: {posture}"
        )

        try:
            client = self._get_client()
            response = await asyncio.to_thread(
                client.messages.create,
                model=SCAN_MODEL,
                max_tokens=SCAN_MAX_TOKENS,
                system=SCAN_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text = block.text
                    break

            # Estimate cost
            from src.agent.token_tracker import MODEL_PRICING

            pricing = MODEL_PRICING.get(SCAN_MODEL, (1.0, 5.0))
            cost = (response.usage.input_tokens * pricing[0] + response.usage.output_tokens * pricing[1]) / 1_000_000

            return self._parse_response(text, cost)

        except Exception as e:
            log.warning("haiku_scan_failed", error=str(e))
            # Safe default: investigate to avoid missing something
            return ScanResult(
                verdict="INVESTIGATE",
                anomalies=[f"Scan failed: {e}"],
                summary="Scan failed, defaulting to investigation",
                cost_usd=0.0,
            )

    @staticmethod
    def _parse_response(text: str, cost: float) -> ScanResult:
        """Parse Haiku's JSON response."""
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
            verdict = data.get("verdict", "INVESTIGATE").upper()
            if verdict not in ("ROUTINE", "INVESTIGATE", "URGENT"):
                verdict = "INVESTIGATE"

            return ScanResult(
                verdict=verdict,
                anomalies=data.get("anomalies", []),
                summary=data.get("summary", ""),
                cost_usd=cost,
            )
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            log.warning("haiku_scan_parse_failed", error=str(e), raw=text[:200])
            return ScanResult(
                verdict="INVESTIGATE",
                anomalies=["Failed to parse scan response"],
                summary="Parse error, defaulting to investigation",
                cost_usd=cost,
            )
