"""Tests for agent settings in config/settings.py."""

from config.settings import AgentSettings


class TestAgentSettings:
    def test_defaults(self) -> None:
        s = AgentSettings()
        assert s.strategy_mode == "conservative"
        assert s.agent_tool_model == "claude-sonnet-4-6"
        assert s.agent_max_turns == 8
        assert s.agent_session_budget == 0.50

    def test_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("AGENT_AGENT_TOOL_MODEL", "claude-opus-4-6")
        monkeypatch.setenv("AGENT_AGENT_MAX_TURNS", "12")
        monkeypatch.setenv("AGENT_AGENT_SESSION_BUDGET", "1.25")
        s = AgentSettings()
        assert s.agent_tool_model == "claude-opus-4-6"
        assert s.agent_max_turns == 12
        assert s.agent_session_budget == 1.25

    def test_strategy_mode_default(self) -> None:
        s = AgentSettings()
        assert s.strategy_mode in ("conservative", "standard", "aggressive")
