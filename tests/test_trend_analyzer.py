"""Tests for TrendAnalyzer — linear regression on improvement snapshots."""

from datetime import date, timedelta

import pytest

from src.analysis.trend_analyzer import (
    TrendAnalyzer,
    _classify,
    _linear_slope,
)
from src.storage.database import Database


@pytest.fixture
async def db():
    d = Database(url="sqlite+aiosqlite:///:memory:")
    await d.init_db()
    yield d
    await d.close()


# ── _linear_slope tests ──────────────────────────────────────────


def test_slope_constant():
    assert _linear_slope([50.0, 50.0, 50.0, 50.0]) == 0.0


def test_slope_increasing():
    slope = _linear_slope([40.0, 50.0, 60.0, 70.0])
    assert slope == pytest.approx(10.0)


def test_slope_decreasing():
    slope = _linear_slope([70.0, 60.0, 50.0, 40.0])
    assert slope == pytest.approx(-10.0)


def test_slope_single_value():
    assert _linear_slope([50.0]) == 0.0


def test_slope_empty():
    assert _linear_slope([]) == 0.0


def test_slope_two_points():
    slope = _linear_slope([40.0, 60.0])
    assert slope == pytest.approx(20.0)


# ── _classify tests ──────────────────────────────────────────────


def test_classify_improving_win_rate():
    assert _classify(6.0, 0.0) == "improving"


def test_classify_declining_win_rate():
    assert _classify(-6.0, 0.0) == "declining"


def test_classify_stable():
    assert _classify(2.0, 1.0) == "stable"


def test_classify_improving_pnl_secondary():
    assert _classify(3.0, 8.0) == "improving"


def test_classify_declining_pnl_secondary():
    assert _classify(-2.0, -7.0) == "declining"


# ── TrendAnalyzer.compute_trend tests ────────────────────────────


async def _seed_snapshots(db: Database, win_rates: list[float]) -> None:
    base = date(2026, 1, 6)
    for i, wr in enumerate(win_rates):
        ws = base + timedelta(weeks=i)
        we = ws + timedelta(days=7)
        await db.save_improvement_snapshot(
            tenant_id="default",
            week_start=ws,
            week_end=we,
            total_trades=10,
            win_rate_pct=wr,
            avg_pnl_pct=wr / 10,
            avg_alpha_vs_spy=None,
            total_cost_usd=1.0,
            strategy_mode="conservative",
            trailing_stop_multiplier=1.0,
            proposal_json=None,
            applied_changes=None,
            report_text=None,
        )


async def test_insufficient_data(db: Database):
    # Only 1 snapshot
    await _seed_snapshots(db, [50.0])

    analyzer = TrendAnalyzer()
    result = await analyzer.compute_trend(db)
    assert result.classification == "insufficient_data"
    assert result.weeks_analyzed == 1


async def test_stable_trend(db: Database):
    await _seed_snapshots(db, [50.0, 51.0, 50.0, 49.0, 50.0])

    analyzer = TrendAnalyzer()
    result = await analyzer.compute_trend(db)
    assert result.classification == "stable"
    assert len(result.data_points) == 5
    assert result.weeks_analyzed == 5


async def test_improving_trend(db: Database):
    await _seed_snapshots(db, [40.0, 50.0, 60.0, 70.0, 80.0])

    analyzer = TrendAnalyzer()
    result = await analyzer.compute_trend(db)
    assert result.classification == "improving"
    assert result.win_rate_slope > 5.0


async def test_declining_trend(db: Database):
    await _seed_snapshots(db, [80.0, 70.0, 60.0, 50.0, 40.0])

    analyzer = TrendAnalyzer()
    result = await analyzer.compute_trend(db)
    assert result.classification == "declining"
    assert result.win_rate_slope < -5.0


async def test_data_points_chronological(db: Database):
    await _seed_snapshots(db, [40.0, 50.0, 60.0])

    analyzer = TrendAnalyzer()
    result = await analyzer.compute_trend(db)

    # Data points should be in chronological order (oldest first)
    assert result.data_points[0].win_rate_pct == 40.0
    assert result.data_points[-1].win_rate_pct == 60.0


async def test_tenant_isolation(db: Database):
    await db.ensure_tenant("t1")
    await _seed_snapshots(db, [50.0, 60.0, 70.0, 80.0])

    # t1 has no snapshots
    analyzer = TrendAnalyzer()
    result = await analyzer.compute_trend(db, tenant_id="t1")
    assert result.classification == "insufficient_data"
