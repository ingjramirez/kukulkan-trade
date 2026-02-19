"""Tests for BTC-related agent system prompt changes."""

from src.agent.claude_agent import SYSTEM_PROMPT, build_system_prompt


class TestSystemPromptBtc:
    def test_default_prompt_mentions_btc(self) -> None:
        assert "BTC-USD" in SYSTEM_PROMPT

    def test_default_prompt_mentions_bitcoin(self) -> None:
        assert "Bitcoin" in SYSTEM_PROMPT

    def test_built_prompt_mentions_btc_trading(self) -> None:
        prompt = build_system_prompt()
        assert "BTC-USD" in prompt
        assert "Bitcoin Trading" in prompt

    def test_built_prompt_mentions_fractional(self) -> None:
        prompt = build_system_prompt()
        assert "fractional" in prompt

    def test_built_prompt_still_has_inverse_etf_rules(self) -> None:
        prompt = build_system_prompt()
        assert "Inverse ETF Rules" in prompt

    def test_built_prompt_has_safety_nets(self) -> None:
        prompt = build_system_prompt()
        assert "safety net" in prompt.lower()

    def test_built_prompt_mentions_247_trading(self) -> None:
        prompt = build_system_prompt()
        assert "24/7" in prompt
