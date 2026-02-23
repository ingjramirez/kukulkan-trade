"""Tests for WeeklyPerformanceData, WeeklyDataCollector, and ImprovementAnalyzer."""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.analysis.outcome_tracker import TradeOutcome
from src.analysis.track_record import TrackRecordStats
from src.analysis.weekly_improvement import (
    TRAILING_STOP_MULTIPLIER_MAX,
    TRAILING_STOP_MULTIPLIER_MIN,
    ImprovementAnalyzer,
    ImprovementProposal,
    ProposedChange,
    WeeklyDataCollector,
    WeeklyPerformanceData,
)
from src.storage.database import Database

# ── WeeklyPerformanceData tests ──────────────────────────────────


def test_weekly_performance_data_defaults():
    data = WeeklyPerformanceData(
        tenant_id="default",
        week_start=date(2026, 2, 10),
        week_end=date(2026, 2, 17),
    )
    assert data.outcomes == []
    assert data.track_record is None
    assert data.current_strategy_mode == "conservative"
    assert data.current_trailing_stop_multiplier == 1.0
    assert data.current_ticker_exclusions == []
    assert data.total_cost_usd == 0.0  # always 0 after Claude Max migration


def test_proposed_change_fields():
    change = ProposedChange(
        category="strategy_mode",
        parameter="strategy_mode",
        old_value="conservative",
        new_value="standard",
        reason="Better win rate",
    )
    assert change.category == "strategy_mode"
    assert change.old_value == "conservative"


def test_improvement_proposal_defaults():
    proposal = ImprovementProposal()
    assert proposal.changes == []
    assert proposal.summary == ""
    assert proposal.raw_json == {}


# ── ImprovementAnalyzer._parse_response tests ────────────────────


def _make_data(**kwargs) -> WeeklyPerformanceData:
    defaults = {
        "tenant_id": "default",
        "week_start": date(2026, 2, 10),
        "week_end": date(2026, 2, 17),
        "current_strategy_mode": "conservative",
        "current_trailing_stop_multiplier": 1.0,
        "current_ticker_exclusions": [],
    }
    defaults.update(kwargs)
    return WeeklyPerformanceData(**defaults)


def test_parse_empty_changes():
    analyzer = ImprovementAnalyzer()
    result = analyzer._parse_response(
        '{"changes": [], "summary": "All good"}',
        _make_data(),
    )
    assert result.changes == []
    assert result.summary == "All good"


def test_parse_strategy_mode_change():
    analyzer = ImprovementAnalyzer()
    result = analyzer._parse_response(
        '{"changes": [{"category": "strategy_mode", "new_value": "standard", "reason": "test"}], "summary": "ok"}',
        _make_data(),
    )
    assert len(result.changes) == 1
    assert result.changes[0].category == "strategy_mode"
    assert result.changes[0].new_value == "standard"
    assert result.changes[0].old_value == "conservative"


def test_parse_invalid_strategy_mode_rejected():
    analyzer = ImprovementAnalyzer()
    result = analyzer._parse_response(
        '{"changes": [{"category": "strategy_mode", "new_value": "yolo", "reason": "test"}], "summary": "ok"}',
        _make_data(),
    )
    assert len(result.changes) == 0


def test_parse_same_strategy_mode_skipped():
    analyzer = ImprovementAnalyzer()
    result = analyzer._parse_response(
        '{"changes": [{"category": "strategy_mode", "new_value": "conservative", "reason": "test"}], "summary": "ok"}',
        _make_data(current_strategy_mode="conservative"),
    )
    assert len(result.changes) == 0


def test_parse_trailing_stop_change():
    analyzer = ImprovementAnalyzer()
    result = analyzer._parse_response(
        '{"changes": [{"category": "trailing_stop", "new_value": "0.8", "reason": "tighter"}], "summary": "ok"}',
        _make_data(),
    )
    assert len(result.changes) == 1
    assert result.changes[0].parameter == "trailing_stop_multiplier"
    assert result.changes[0].new_value == "0.8"


def test_parse_trailing_stop_clamped():
    analyzer = ImprovementAnalyzer()
    # Too low
    result = analyzer._parse_response(
        '{"changes": [{"category": "trailing_stop", "new_value": "0.1", "reason": "test"}], "summary": "ok"}',
        _make_data(),
    )
    assert len(result.changes) == 1
    assert result.changes[0].new_value == str(TRAILING_STOP_MULTIPLIER_MIN)

    # Too high
    result = analyzer._parse_response(
        '{"changes": [{"category": "trailing_stop", "new_value": "5.0", "reason": "test"}], "summary": "ok"}',
        _make_data(),
    )
    assert len(result.changes) == 1
    assert result.changes[0].new_value == str(TRAILING_STOP_MULTIPLIER_MAX)


def test_parse_trailing_stop_same_value_skipped():
    analyzer = ImprovementAnalyzer()
    result = analyzer._parse_response(
        '{"changes": [{"category": "trailing_stop", "new_value": "1.0", "reason": "test"}], "summary": "ok"}',
        _make_data(current_trailing_stop_multiplier=1.0),
    )
    assert len(result.changes) == 0


def test_parse_universe_exclude():
    analyzer = ImprovementAnalyzer()
    j = '{"changes": [{"category": "universe_exclude", "new_value": "TSLA", "reason": "loser"}], "summary": "ok"}'
    result = analyzer._parse_response(j, _make_data())
    assert len(result.changes) == 1
    assert result.changes[0].new_value == "TSLA"
    assert result.changes[0].parameter == "ticker_exclusion:TSLA"


def test_parse_universe_exclude_already_excluded():
    analyzer = ImprovementAnalyzer()
    result = analyzer._parse_response(
        '{"changes": [{"category": "universe_exclude", "new_value": "TSLA", "reason": "test"}], "summary": "ok"}',
        _make_data(current_ticker_exclusions=["TSLA"]),
    )
    assert len(result.changes) == 0


def test_parse_universe_exclude_max_5():
    analyzer = ImprovementAnalyzer()
    changes_json = [{"category": "universe_exclude", "new_value": f"T{i}", "reason": "test"} for i in range(7)]
    result = analyzer._parse_response(
        f'{{"changes": {__import__("json").dumps(changes_json)}, "summary": "ok"}}',
        _make_data(),
    )
    assert len([c for c in result.changes if c.category == "universe_exclude"]) == 5


def test_parse_learning():
    analyzer = ImprovementAnalyzer()
    j = (
        '{"changes": [{"category": "learning", "parameter": "sector_insight",'
        ' "new_value": "Tech does well in bull", "reason": "observed"}], "summary": "ok"}'
    )
    result = analyzer._parse_response(j, _make_data())
    assert len(result.changes) == 1
    assert result.changes[0].category == "learning"
    assert result.changes[0].new_value == "Tech does well in bull"


def test_parse_learning_max_5():
    analyzer = ImprovementAnalyzer()
    import json

    changes_json = [
        {"category": "learning", "parameter": f"l{i}", "new_value": f"insight {i}", "reason": "test"} for i in range(7)
    ]
    result = analyzer._parse_response(
        json.dumps({"changes": changes_json, "summary": "ok"}),
        _make_data(),
    )
    assert len([c for c in result.changes if c.category == "learning"]) == 5


def test_parse_markdown_fenced_json():
    analyzer = ImprovementAnalyzer()
    fenced = '```json\n{"changes": [], "summary": "clean"}\n```'
    result = analyzer._parse_response(fenced, _make_data())
    assert result.summary == "clean"


def test_parse_invalid_json():
    analyzer = ImprovementAnalyzer()
    result = analyzer._parse_response("not json at all", _make_data())
    assert result.changes == []
    assert "Failed to parse" in result.summary


def test_parse_trailing_stop_invalid_value():
    analyzer = ImprovementAnalyzer()
    result = analyzer._parse_response(
        '{"changes": [{"category": "trailing_stop", "new_value": "abc", "reason": "test"}], "summary": "ok"}',
        _make_data(),
    )
    assert len(result.changes) == 0


# ── ImprovementAnalyzer._build_prompt tests ──────────────────────


def test_build_prompt_basic():
    analyzer = ImprovementAnalyzer()
    data = _make_data(universe_size=70)
    prompt = analyzer._build_prompt(data)
    assert "Weekly Performance Review" in prompt
    assert "Total trades: 0" in prompt
    assert "70" in prompt


def test_build_prompt_with_track_record():
    analyzer = ImprovementAnalyzer()
    stats = TrackRecordStats(
        total_trades=10,
        wins=6,
        losses=3,
        scratches=1,
        win_rate_pct=60.0,
        avg_pnl_pct=1.5,
        avg_alpha_vs_spy=0.8,
        best_sector="Technology",
        worst_sector="Energy",
    )
    data = _make_data(track_record=stats)
    prompt = analyzer._build_prompt(data)
    assert "Win rate: 60.0%" in prompt
    assert "Best sector: Technology" in prompt


def test_build_prompt_with_previous_changes():
    analyzer = ImprovementAnalyzer()
    data = _make_data()
    prev = [{"parameter": "strategy_mode", "old_value": "conservative", "new_value": "standard", "reason": "test"}]
    prompt = analyzer._build_prompt(data, previous_changes=prev)
    assert "Recent Parameter Changes" in prompt
    assert "strategy_mode" in prompt


# ── WeeklyDataCollector tests ─────────────────────────────────────


@pytest.fixture
async def db():
    d = Database(url="sqlite+aiosqlite:///:memory:")
    await d.init_db()
    yield d
    await d.close()


async def test_collector_returns_data(db: Database):
    with (
        patch("src.analysis.weekly_improvement.OutcomeTracker") as mock_tracker_cls,
        patch("src.analysis.weekly_improvement.TrackRecord") as mock_record_cls,
    ):
        mock_tracker = MagicMock()
        mock_tracker.get_recent_outcomes = AsyncMock(return_value=[])
        mock_tracker_cls.return_value = mock_tracker

        mock_record = MagicMock()
        mock_record_cls.return_value = mock_record

        collector = WeeklyDataCollector(db)
        data = await collector.collect("default")

        assert data.tenant_id == "default"
        assert data.current_strategy_mode == "conservative"
        assert data.current_trailing_stop_multiplier == 1.0
        assert data.outcomes == []


async def test_collector_with_outcomes(db: Database):
    outcome = TradeOutcome(
        ticker="AAPL",
        side="BUY",
        entry_price=150.0,
        current_price=160.0,
        exit_price=None,
        pnl_pct=6.67,
        hold_days=5,
        sector="Technology",
        sector_etf_pct=2.0,
        spy_pct=1.0,
        alpha_vs_sector=4.67,
        alpha_vs_spy=5.67,
        conviction="high",
        reasoning="strong momentum",
    )

    with (
        patch("src.analysis.weekly_improvement.OutcomeTracker") as mock_tracker_cls,
        patch("src.analysis.weekly_improvement.TrackRecord") as mock_record_cls,
    ):
        mock_tracker = MagicMock()
        mock_tracker.get_recent_outcomes = AsyncMock(return_value=[outcome])
        mock_tracker_cls.return_value = mock_tracker

        mock_stats = TrackRecordStats(
            total_trades=1,
            wins=1,
            losses=0,
            scratches=0,
            win_rate_pct=100.0,
            avg_pnl_pct=6.67,
            avg_alpha_vs_spy=5.67,
        )
        mock_record = MagicMock()
        mock_record.compute.return_value = mock_stats
        mock_record_cls.return_value = mock_record

        collector = WeeklyDataCollector(db)
        data = await collector.collect("default")

        assert len(data.outcomes) == 1
        assert data.track_record is not None
        assert data.track_record.win_rate_pct == 100.0


# ── ImprovementAnalyzer.analyze (mocked Claude CLI) ───────────────


@patch("src.agent.claude_invoker.claude_cli_json", new_callable=AsyncMock)
async def test_analyzer_analyze_success(mock_cli):
    analyzer = ImprovementAnalyzer()
    data = _make_data()

    mock_cli.return_value = {"changes": [], "summary": "looks good"}

    result = await analyzer.analyze(data)
    assert result.summary == "looks good"
    assert result.changes == []


@patch("src.agent.claude_invoker.claude_cli_json", new_callable=AsyncMock)
async def test_analyzer_analyze_api_error(mock_cli):
    analyzer = ImprovementAnalyzer()
    data = _make_data()

    mock_cli.side_effect = Exception("CLI error")
    result = await analyzer.analyze(data)
    assert "failed" in result.summary.lower()
