"""Tests for feedback integration in build_system_prompt."""

from src.agent.claude_agent import _build_decision_review, build_system_prompt
from src.analysis.outcome_tracker import TradeOutcome
from src.analysis.track_record import TrackRecord, TrackRecordStats


def _make_outcome(ticker: str = "XLK", pnl_pct: float = 5.0) -> TradeOutcome:
    return TradeOutcome(
        ticker=ticker,
        side="BUY",
        entry_price=100.0,
        current_price=105.0,
        exit_price=None,
        pnl_pct=pnl_pct,
        hold_days=5,
        sector="Technology",
        sector_etf_pct=3.0,
        spy_pct=2.0,
        alpha_vs_sector=2.0,
        alpha_vs_spy=3.0,
        conviction="high",
        reasoning="test",
    )


def test_build_system_prompt_with_decision_review():
    outcomes = [_make_outcome("XLK", 5.0), _make_outcome("XLE", -2.0)]
    review = _build_decision_review(outcomes)
    prompt = build_system_prompt(decision_review=review)
    assert "Recent Decision Outcomes" in prompt
    assert "XLK" in prompt
    assert "XLE" in prompt


def test_build_system_prompt_with_track_record():
    stats = TrackRecordStats(
        total_trades=10,
        wins=6,
        losses=3,
        scratches=1,
        win_rate_pct=66.7,
        avg_pnl_pct=2.5,
        avg_alpha_vs_spy=1.0,
    )
    text = TrackRecord.format_for_prompt(stats)
    prompt = build_system_prompt(track_record=text)
    assert "Win Rate Analysis" in prompt
    assert "67%" in prompt


def test_prompt_section_ordering():
    """Verify feedback sections appear after performance but before memory."""
    prompt = build_system_prompt(
        performance_stats="perf stats here",
        decision_review="review here",
        track_record="track record here",
        memory_context="## Memory\nmemory here",
    )
    perf_idx = prompt.index("perf stats here")
    review_idx = prompt.index("review here")
    track_idx = prompt.index("track record here")
    memory_idx = prompt.index("memory here")
    assert perf_idx < review_idx < track_idx < memory_idx


def test_build_system_prompt_empty_feedback():
    """Empty feedback doesn't add sections."""
    prompt = build_system_prompt(decision_review=None, track_record=None)
    assert "Recent Decision Outcomes" not in prompt
    assert "Win Rate Analysis" not in prompt
