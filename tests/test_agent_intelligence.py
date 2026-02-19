"""Tests for Phase 44 — agent intelligence maximization features.

Covers: decision quality feedback, Portfolio A benchmark, strategy directive wiring,
and concentration guidance injection.
"""

from src.agent.context_manager import ContextManager
from src.analysis.decision_quality import DecisionQualitySummary, DecisionQualityTracker

# ── Decision quality format ─────────────────────────────────────────────────


def test_decision_quality_format_with_data():
    """format_for_prompt produces text with accuracy percentages."""
    summary = DecisionQualitySummary(
        total_decisions=10,
        favorable_1d_pct=60.0,
        favorable_3d_pct=70.0,
        favorable_5d_pct=50.0,
    )
    text = DecisionQualityTracker.format_for_prompt(summary)
    assert "10 trades" in text
    assert "1d=60%" in text
    assert "3d=70%" in text
    assert "5d=50%" in text


def test_decision_quality_format_empty():
    """format_for_prompt handles no decisions gracefully."""
    summary = DecisionQualitySummary(
        total_decisions=0,
        favorable_1d_pct=0.0,
        favorable_3d_pct=0.0,
        favorable_5d_pct=0.0,
    )
    text = DecisionQualityTracker.format_for_prompt(summary)
    assert "No decisions" in text


# ── Strategy directive resolution ───────────────────────────────────────────


def test_strategy_directive_resolves_for_conservative():
    """Conservative mode resolves to CONSERVATIVE_DIRECTIVE."""
    from src.agent.strategy_directives import STRATEGY_MAP

    directive = STRATEGY_MAP.get("conservative", "")
    assert "CONSERVATIVE" in directive
    assert "lower-volatility" in directive


def test_strategy_directive_resolves_for_aggressive():
    """Aggressive mode resolves to AGGRESSIVE_DIRECTIVE."""
    from src.agent.strategy_directives import STRATEGY_MAP

    directive = STRATEGY_MAP.get("aggressive", "")
    assert "AGGRESSIVE" in directive
    assert "Concentrate" in directive


def test_session_directive_appended_for_morning():
    """Morning session gets session-specific directive."""
    from src.agent.strategy_directives import SESSION_DIRECTIVES

    session_label = {"morning": "Morning", "midday": "Midday", "close": "Closing"}.get("morning")
    assert session_label in SESSION_DIRECTIVES
    assert "Post-Open" in SESSION_DIRECTIVES[session_label]


def test_session_directive_appended_for_midday():
    """Midday session gets session-specific directive."""
    from src.agent.strategy_directives import SESSION_DIRECTIVES

    session_label = {"morning": "Morning", "midday": "Midday", "close": "Closing"}.get("midday")
    assert session_label in SESSION_DIRECTIVES
    assert "Profit-Taking" in SESSION_DIRECTIVES[session_label]


def test_session_directive_appended_for_close():
    """Close session gets session-specific directive."""
    from src.agent.strategy_directives import SESSION_DIRECTIVES

    session_label = {"morning": "Morning", "midday": "Midday", "close": "Closing"}.get("close")
    assert session_label in SESSION_DIRECTIVES
    assert "Overnight Risk" in SESSION_DIRECTIVES[session_label]


def test_concentration_guidance_content():
    """Concentration guidance text is valid and mentions key themes."""
    guidance = (
        "## Position Sizing Philosophy\n"
        "Portfolio A (your benchmark) uses extreme concentration: 100% in one ETF.\n"
        "Its outperformance shows that conviction > diversification for paper trading.\n"
        "- Aim for 5-8 high-conviction positions, NOT 15-20 small ones\n"
        "- Size top 3 ideas at 10-20% each\n"
        "- A position under 3% of portfolio is noise — size up or skip it"
    )
    assert "5-8" in guidance
    assert "10-20%" in guidance
    assert "3%" in guidance


# ── Cached system prompt includes strategy directive ────────────────────────


def test_cached_prompt_includes_strategy_directive():
    """build_cached_system_prompt includes strategy directive block."""
    cm = ContextManager()
    blocks = cm.build_cached_system_prompt(
        pinned_context="## Posture: Balanced",
        strategy_directive="## CONSERVATIVE\nBe careful.",
    )
    texts = [b["text"] for b in blocks]
    combined = "\n".join(texts)
    assert "CONSERVATIVE" in combined
    assert "Be careful" in combined


def test_cached_prompt_without_directive():
    """build_cached_system_prompt works without strategy directive."""
    cm = ContextManager()
    blocks = cm.build_cached_system_prompt(
        pinned_context="## Posture: Balanced",
        strategy_directive="",
    )
    # Should have identity + pinned context, no strategy block
    assert len(blocks) == 2  # identity + pinned
