"""Tests for inverse ETF system prompt additions."""

from src.agent.claude_agent import build_system_prompt


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
