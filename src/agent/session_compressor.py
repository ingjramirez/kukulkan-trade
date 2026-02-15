"""Compresses old agent sessions into ~500-token summaries using Haiku.

Includes a validation system that uses Sonnet to verify compression quality
before the compression pipeline goes live.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import structlog

log = structlog.get_logger()

COMPRESSION_MODEL = "claude-haiku-4-5-20251001"
VALIDATION_MODEL = "claude-sonnet-4-5-20250929"
TARGET_TOKENS = 500

COMPRESSION_PROMPT = """Compress this trading session into a ~500-token summary.

PRESERVE (these are critical):
- Every trade executed: ticker, side, shares, price, reason
- Position changes: entries, exits, size adjustments
- Key observations and theses stated by the agent
- Multi-day plans ("plan to add Wednesday", "watching for support at $115")
- Specific price levels mentioned as targets or stops
- Regime assessment and any posture changes
- Risk concerns raised

OMIT:
- Tool call details (input/output JSON)
- Repetitive portfolio state reads
- Boilerplate analysis

Format: Narrative paragraph, not bullet points. Use specific numbers.

Session to compress:
{session_text}"""

VALIDATION_PROMPT = """Compare this compressed summary to the original trading session.

List any TRADING-RELEVANT facts that are MISSING from the summary:
- Trades executed (ticker, side, shares, price)
- Position changes
- Theses or multi-day plans
- Specific price levels or targets
- Risk concerns

Respond in JSON:
{{
  "missing_facts": ["fact 1", "fact 2"],
  "fidelity_score": 0.95,
  "assessment": "Good/Acceptable/Poor"
}}

Original session:
{original_text}

Compressed summary:
{summary}"""


@dataclass(frozen=True)
class CompressionValidation:
    """Result of validating a compression against the original session."""

    missing_facts: list[str]
    fidelity_score: float  # 0.0 to 1.0
    is_acceptable: bool  # fidelity > 0.95


class SessionCompressor:
    """Compresses old sessions into ~500-token summaries using Haiku."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    def _get_client(self):
        """Lazy client initialization."""
        import anthropic

        return anthropic.Anthropic(api_key=self._api_key)

    async def compress(self, session_messages: list[dict]) -> str:
        """Compress a full session to a ~500-token summary using Haiku.

        Args:
            session_messages: Full Anthropic messages array for the session.

        Returns:
            The summary text.

        Raises:
            CompressionError: If Haiku returns empty or fails.
        """
        session_text = self._format_messages_for_compression(session_messages)
        if not session_text.strip():
            raise CompressionError("Cannot compress empty session")

        prompt = COMPRESSION_PROMPT.format(session_text=session_text)

        import asyncio

        client = self._get_client()
        response = await asyncio.to_thread(
            client.messages.create,
            model=COMPRESSION_MODEL,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )

        summary = ""
        for block in response.content:
            if block.type == "text":
                summary = block.text
                break

        if not summary.strip():
            raise CompressionError("Haiku returned empty summary")

        return summary.strip()

    async def validate_compression(
        self,
        original_messages: list[dict],
        summary: str,
    ) -> CompressionValidation:
        """Use Sonnet to validate compression quality.

        Args:
            original_messages: Full message history of the original session.
            summary: Compressed summary to validate.

        Returns:
            CompressionValidation with missing facts and fidelity score.
        """
        original_text = self._format_messages_for_compression(original_messages)
        prompt = VALIDATION_PROMPT.format(original_text=original_text, summary=summary)

        import asyncio

        client = self._get_client()
        response = await asyncio.to_thread(
            client.messages.create,
            model=VALIDATION_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        text = ""
        for block in response.content:
            if block.type == "text":
                text = block.text
                break

        return self._parse_validation(text)

    @staticmethod
    def _format_messages_for_compression(messages: list[dict]) -> str:
        """Convert messages array to a readable text format for compression."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if isinstance(content, str):
                lines.append(f"[{role}]: {content}")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            lines.append(f"[{role}]: {block.get('text', '')}")
                        elif block.get("type") == "tool_use":
                            name = block.get("name", "unknown_tool")
                            inp = block.get("input", {})
                            lines.append(f"[{role} → tool:{name}]: {json.dumps(inp, default=str)[:200]}")
                        elif block.get("type") == "tool_result":
                            result = block.get("content", "")
                            lines.append(f"[tool_result]: {str(result)[:200]}")
        return "\n".join(lines)

    @staticmethod
    def _parse_validation(text: str) -> CompressionValidation:
        """Parse Sonnet's validation response JSON."""
        try:
            # Handle markdown fences
            cleaned = text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]

            data = json.loads(cleaned.strip())
            missing = data.get("missing_facts", [])
            score = float(data.get("fidelity_score", 0.0))
            return CompressionValidation(
                missing_facts=missing,
                fidelity_score=score,
                is_acceptable=score >= 0.95,
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            return CompressionValidation(
                missing_facts=["Failed to parse validation response"],
                fidelity_score=0.0,
                is_acceptable=False,
            )


class CompressionError(Exception):
    """Raised when session compression fails."""


async def run_compression_validation(
    compressor: SessionCompressor,
    sessions: list[dict],
) -> dict:
    """Validate compression on real sessions before enabling auto-compression.

    Args:
        compressor: SessionCompressor instance with API key.
        sessions: List of session dicts with "messages" key.

    Returns:
        Dict with per-session results and aggregate fidelity score.
    """
    results = []
    for session in sessions:
        messages = session.get("messages", [])
        if not messages:
            continue
        try:
            summary = await compressor.compress(messages)
            validation = await compressor.validate_compression(messages, summary)
            results.append(
                {
                    "session_id": session.get("session_id", "unknown"),
                    "summary": summary,
                    "fidelity_score": validation.fidelity_score,
                    "is_acceptable": validation.is_acceptable,
                    "missing_facts": validation.missing_facts,
                }
            )
        except Exception as e:
            results.append(
                {
                    "session_id": session.get("session_id", "unknown"),
                    "error": str(e),
                    "fidelity_score": 0.0,
                    "is_acceptable": False,
                }
            )

    if not results:
        return {"results": [], "avg_fidelity": 0.0, "all_acceptable": False}

    avg_fidelity = sum(r.get("fidelity_score", 0) for r in results) / len(results)
    all_acceptable = all(r.get("is_acceptable", False) for r in results)

    return {
        "results": results,
        "avg_fidelity": round(avg_fidelity, 3),
        "all_acceptable": all_acceptable,
    }


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Session compressor CLI")
    parser.add_argument("--validate", action="store_true", help="Validate compression on recent sessions")
    parser.add_argument("--compress-session", type=str, help="Compress a single session by ID")
    parser.add_argument("--tenant", type=str, default="default", help="Tenant ID")
    args = parser.parse_args()

    if args.validate:
        print("Compression validation requires loading sessions from DB.")
        print("Usage: python -m src.agent.session_compressor --validate --tenant default")
        print("(Full implementation connects to DB and runs validation)")
        sys.exit(0)
    elif args.compress_session:
        print(f"Would compress session: {args.compress_session}")
        sys.exit(0)
    else:
        parser.print_help()
