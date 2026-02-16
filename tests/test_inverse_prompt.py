"""Tests for inverse ETF system prompt additions."""

from src.agent.claude_agent import build_system_prompt
from src.agent.context_manager import _SYSTEM_IDENTITY, ContextManager


class TestBuildSystemPrompt:
    def test_includes_inverse_etf_rules(self) -> None:
        prompt = build_system_prompt()
        assert "Inverse ETF Rules:" in prompt
        assert "SH" in prompt
        assert "PSQ" in prompt
        assert "RWM" in prompt
        assert "TBF" in prompt

    def test_includes_decay_warning(self) -> None:
        prompt = build_system_prompt()
        assert "decay" in prompt.lower()

    def test_includes_regime_gating(self) -> None:
        prompt = build_system_prompt()
        assert "CORRECTION" in prompt
        assert "CRISIS" in prompt

    def test_inverse_etf_context_appears(self) -> None:
        ctx = "- SH (Short S&P 500): 200 shares, $3,000, P&L -1.5%"
        prompt = build_system_prompt(inverse_etf_context=ctx)
        assert "## Inverse ETF Positions" in prompt
        assert "SH (Short S&P 500)" in prompt

    def test_inverse_etf_context_omitted_when_none(self) -> None:
        prompt = build_system_prompt(inverse_etf_context=None)
        assert "## Inverse ETF Positions" not in prompt


class TestContextManager:
    def test_guardrails_mention_inverse(self) -> None:
        assert "Inverse ETF" in _SYSTEM_IDENTITY
        assert "SH, PSQ, RWM" in _SYSTEM_IDENTITY

    def test_build_system_prompt_includes_guardrails(self) -> None:
        cm = ContextManager()
        prompt = cm.build_system_prompt(pinned_context="")
        assert "inverse ETF" in prompt.lower() or "Inverse ETF" in prompt

    def test_build_cached_system_prompt_includes_guardrails(self) -> None:
        cm = ContextManager()
        blocks = cm.build_cached_system_prompt(pinned_context="")
        # The identity block should contain inverse ETF mentions
        identity_text = blocks[0]["text"]
        assert "Inverse ETF" in identity_text
