"""Tests for WeeklyImprovementPipeline — the full collect→analyze→apply→save→notify loop."""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.analysis.improvement_pipeline import (
    WeeklyImprovementPipeline,
    _format_report,
    _format_telegram_report,
)
from src.analysis.outcome_tracker import TradeOutcome
from src.analysis.track_record import TrackRecordStats
from src.analysis.weekly_improvement import (
    ImprovementAnalyzer,
    ImprovementProposal,
    ProposedChange,
    WeeklyPerformanceData,
)
from src.storage.database import Database


@pytest.fixture
async def db():
    d = Database(url="sqlite+aiosqlite:///:memory:")
    await d.init_db()
    yield d
    await d.close()


def _make_outcome(**kwargs) -> TradeOutcome:
    defaults = dict(
        ticker="AAPL", side="BUY", entry_price=150.0, current_price=160.0,
        exit_price=None, pnl_pct=6.67, hold_days=5, sector="Technology",
        sector_etf_pct=2.0, spy_pct=1.0, alpha_vs_sector=4.67, alpha_vs_spy=5.67,
        conviction="high", reasoning="strong momentum",
    )
    defaults.update(kwargs)
    return TradeOutcome(**defaults)


def _make_data(**kwargs) -> WeeklyPerformanceData:
    defaults = {
        "tenant_id": "default",
        "week_start": date(2026, 2, 10),
        "week_end": date(2026, 2, 17),
        "outcomes": [_make_outcome()],
        "track_record": TrackRecordStats(
            total_trades=1, wins=1, losses=0, scratches=0,
            win_rate_pct=100.0, avg_pnl_pct=6.67, avg_alpha_vs_spy=5.67,
        ),
        "current_strategy_mode": "conservative",
        "current_trailing_stop_multiplier": 1.0,
        "total_cost_usd": 2.50,
    }
    defaults.update(kwargs)
    return WeeklyPerformanceData(**defaults)


# ── Full Pipeline Tests ──────────────────────────────────────────


async def test_pipeline_skip_no_trades(db: Database):
    """Pipeline skips if no trades in the past week."""
    mock_analyzer = MagicMock(spec=ImprovementAnalyzer)

    with patch("src.analysis.improvement_pipeline.WeeklyDataCollector") as mock_collector_cls:
        mock_collector = MagicMock()
        mock_collector.collect = AsyncMock(return_value=_make_data(outcomes=[]))
        mock_collector_cls.return_value = mock_collector

        pipeline = WeeklyImprovementPipeline(db, analyzer=mock_analyzer)
        result = await pipeline.run("default")

        assert result["status"] == "skipped"
        assert result["total_trades"] == 0
        mock_analyzer.analyze.assert_not_called()


async def test_pipeline_full_run(db: Database):
    """Pipeline runs full loop: collect → analyze → apply → save."""
    proposal = ImprovementProposal(
        changes=[
            ProposedChange(
                category="learning",
                parameter="weekly_note",
                old_value=None,
                new_value="Tech is strong",
                reason="observed",
            )
        ],
        summary="Solid week",
        raw_json={
            "changes": [{"category": "learning", "parameter": "weekly_note", "new_value": "Tech is strong"}],
            "summary": "Solid week",
        },
    )

    mock_analyzer = MagicMock(spec=ImprovementAnalyzer)
    mock_analyzer.analyze = AsyncMock(return_value=proposal)

    with patch("src.analysis.improvement_pipeline.WeeklyDataCollector") as mock_collector_cls:
        mock_collector = MagicMock()
        mock_collector.collect = AsyncMock(return_value=_make_data())
        mock_collector_cls.return_value = mock_collector

        pipeline = WeeklyImprovementPipeline(db, analyzer=mock_analyzer)
        result = await pipeline.run("default")

        assert result["status"] == "completed"
        assert result["total_trades"] == 1
        assert result["changes_applied"] == 1
        assert result["snapshot_id"] > 0


async def test_pipeline_no_changes_proposed(db: Database):
    """Pipeline handles empty proposal gracefully."""
    proposal = ImprovementProposal(summary="All good, no changes needed")

    mock_analyzer = MagicMock(spec=ImprovementAnalyzer)
    mock_analyzer.analyze = AsyncMock(return_value=proposal)

    with patch("src.analysis.improvement_pipeline.WeeklyDataCollector") as mock_collector_cls:
        mock_collector = MagicMock()
        mock_collector.collect = AsyncMock(return_value=_make_data())
        mock_collector_cls.return_value = mock_collector

        pipeline = WeeklyImprovementPipeline(db, analyzer=mock_analyzer)
        result = await pipeline.run("default")

        assert result["status"] == "completed"
        assert result["changes_applied"] == 0


async def test_pipeline_sends_telegram(db: Database):
    """Pipeline sends Telegram notification when notifier provided."""
    proposal = ImprovementProposal(summary="Good week")
    mock_analyzer = MagicMock(spec=ImprovementAnalyzer)
    mock_analyzer.analyze = AsyncMock(return_value=proposal)

    mock_notifier = MagicMock()
    mock_notifier.send_message = AsyncMock()

    with patch("src.analysis.improvement_pipeline.WeeklyDataCollector") as mock_collector_cls:
        mock_collector = MagicMock()
        mock_collector.collect = AsyncMock(return_value=_make_data())
        mock_collector_cls.return_value = mock_collector

        pipeline = WeeklyImprovementPipeline(db, analyzer=mock_analyzer)
        await pipeline.run("default", notifier=mock_notifier)

        mock_notifier.send_message.assert_called_once()
        call_args = mock_notifier.send_message.call_args
        assert "Weekly Self-Improvement Report" in call_args.args[0]
        assert call_args.kwargs["parse_mode"] == "HTML"


async def test_pipeline_telegram_failure_non_blocking(db: Database):
    """Telegram failure should not stop the pipeline."""
    proposal = ImprovementProposal(summary="ok")
    mock_analyzer = MagicMock(spec=ImprovementAnalyzer)
    mock_analyzer.analyze = AsyncMock(return_value=proposal)

    mock_notifier = MagicMock()
    mock_notifier.send_message = AsyncMock(side_effect=Exception("Telegram down"))

    with patch("src.analysis.improvement_pipeline.WeeklyDataCollector") as mock_collector_cls:
        mock_collector = MagicMock()
        mock_collector.collect = AsyncMock(return_value=_make_data())
        mock_collector_cls.return_value = mock_collector

        pipeline = WeeklyImprovementPipeline(db, analyzer=mock_analyzer)
        result = await pipeline.run("default", notifier=mock_notifier)

        assert result["status"] == "completed"  # Not error


async def test_pipeline_saves_snapshot(db: Database):
    """Pipeline saves snapshot to database."""
    proposal = ImprovementProposal(
        summary="Solid",
        raw_json={"changes": [], "summary": "Solid"},
    )
    mock_analyzer = MagicMock(spec=ImprovementAnalyzer)
    mock_analyzer.analyze = AsyncMock(return_value=proposal)

    with patch("src.analysis.improvement_pipeline.WeeklyDataCollector") as mock_collector_cls:
        mock_collector = MagicMock()
        mock_collector.collect = AsyncMock(return_value=_make_data())
        mock_collector_cls.return_value = mock_collector

        pipeline = WeeklyImprovementPipeline(db, analyzer=mock_analyzer)
        result = await pipeline.run("default")

        snap_id = result["snapshot_id"]
        snapshot = await db.get_improvement_snapshot(snap_id, "default")
        assert snapshot is not None
        assert snapshot.total_trades == 1
        assert snapshot.win_rate_pct == 100.0
        assert snapshot.report_text is not None


async def test_pipeline_error_handling(db: Database):
    """Pipeline returns error status on exception."""
    mock_analyzer = MagicMock(spec=ImprovementAnalyzer)

    with patch("src.analysis.improvement_pipeline.WeeklyDataCollector") as mock_collector_cls:
        mock_collector = MagicMock()
        mock_collector.collect = AsyncMock(side_effect=Exception("DB down"))
        mock_collector_cls.return_value = mock_collector

        pipeline = WeeklyImprovementPipeline(db, analyzer=mock_analyzer)
        result = await pipeline.run("default")

        assert result["status"] == "error"
        assert "DB down" in result["error"]


async def test_pipeline_loads_previous_changes(db: Database):
    """Pipeline passes previous changelog to analyzer."""
    await db.insert_parameter_changelog(
        tenant_id="default",
        parameter="strategy_mode",
        old_value="conservative",
        new_value="standard",
        reason="prev change",
    )

    proposal = ImprovementProposal(summary="noted")
    mock_analyzer = MagicMock(spec=ImprovementAnalyzer)
    mock_analyzer.analyze = AsyncMock(return_value=proposal)

    with patch("src.analysis.improvement_pipeline.WeeklyDataCollector") as mock_collector_cls:
        mock_collector = MagicMock()
        mock_collector.collect = AsyncMock(return_value=_make_data())
        mock_collector_cls.return_value = mock_collector

        pipeline = WeeklyImprovementPipeline(db, analyzer=mock_analyzer)
        await pipeline.run("default")

        # Check that analyzer received previous changes
        call_args = mock_analyzer.analyze.call_args
        prev = call_args.kwargs.get("previous_changes") or call_args.args[1]
        assert len(prev) == 1
        assert prev[0]["parameter"] == "strategy_mode"


# ── Report Formatting Tests ──────────────────────────────────────


def test_format_report_basic():
    data = _make_data()
    proposal = ImprovementProposal(summary="Good week")
    applied = []

    report = _format_report(data, proposal, applied)
    assert "Weekly Improvement Report" in report
    assert "Trades: 1" in report
    assert "Win rate: 100.0%" in report
    assert "No changes proposed" in report


def test_format_report_with_changes():
    data = _make_data()
    proposal = ImprovementProposal(summary="Changes needed")
    applied = [
        {"parameter": "strategy_mode", "old_value": "conservative", "new_value": "standard", "status": "applied"},
    ]

    report = _format_report(data, proposal, applied)
    assert "[applied] strategy_mode" in report


def test_format_telegram_report_basic():
    data = _make_data()
    proposal = ImprovementProposal(summary="Solid performance")
    applied = []

    report = _format_telegram_report(data, proposal, applied)
    assert "<b>Weekly Self-Improvement Report</b>" in report
    assert "Solid performance" in report
    assert "No parameter changes this week" in report


def test_format_telegram_report_with_changes():
    data = _make_data()
    proposal = ImprovementProposal(summary="Adjusting")
    applied = [
        {
            "parameter": "strategy_mode", "old_value": "conservative",
            "new_value": "standard", "status": "applied", "reason": "test",
        },
        {
            "parameter": "trailing_stop_multiplier", "old_value": "1.0",
            "new_value": "0.8", "status": "blocked_flipflop", "reason": "blocked",
        },
    ]

    report = _format_telegram_report(data, proposal, applied)
    assert "<b>Changes Applied (1)</b>" in report
    assert "<b>Blocked" in report
    assert "strategy_mode" in report


def test_format_telegram_report_escapes_html():
    data = _make_data()
    proposal = ImprovementProposal(summary="Use <b>caution</b> & watch")
    applied = []

    report = _format_telegram_report(data, proposal, applied)
    assert "&lt;b&gt;caution&lt;/b&gt;" in report
    assert "&amp;" in report
