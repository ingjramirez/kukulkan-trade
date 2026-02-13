"""Tests for DecisionQualityTracker — forward return analysis."""

import json
from datetime import date, timedelta
from unittest.mock import AsyncMock

import pytest

from src.analysis.decision_quality import (
    DecisionQuality,
    DecisionQualityTracker,
)
from src.storage.database import Database
from src.storage.models import AgentDecisionRow


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


async def _seed_decision(
    db: Database,
    decision_date: date,
    ticker: str = "XLK",
    side: str = "BUY",
):
    trades_json = json.dumps([{"ticker": ticker, "side": side, "shares": 10, "price": 100}])
    async with db.session() as s:
        s.add(
            AgentDecisionRow(
                tenant_id="default",
                date=decision_date,
                prompt_summary="test",
                response_summary="test",
                proposed_trades=trades_json,
                reasoning="test",
                model_used="test",
                tokens_used=100,
            )
        )
        await s.commit()


@pytest.mark.asyncio
async def test_empty_db(db):
    tracker = DecisionQualityTracker(db)
    result = await tracker.analyze_recent(days=30, tenant_id="default")
    assert result == []


@pytest.mark.asyncio
async def test_favorable_buy(db):
    """A BUY with positive forward return is favorable."""
    decision_date = date.today() - timedelta(days=10)
    await _seed_decision(db, decision_date, "XLK", "BUY")

    tracker = DecisionQualityTracker(db)
    # Mock prices: XLK goes up
    tracker._fetch_prices = AsyncMock(
        return_value={
            "XLK": {
                decision_date: 100.0,
                decision_date + timedelta(days=1): 102.0,
                decision_date + timedelta(days=3): 105.0,
                decision_date + timedelta(days=5): 108.0,
            }
        }
    )

    qualities = await tracker.analyze_recent(days=30, tenant_id="default")
    assert len(qualities) == 1
    assert qualities[0].favorable_1d is True
    assert qualities[0].favorable_3d is True
    assert qualities[0].favorable_5d is True
    assert qualities[0].fwd_1d == 2.0  # (102-100)/100 * 100


@pytest.mark.asyncio
async def test_favorable_sell(db):
    """A SELL with negative forward return is favorable."""
    decision_date = date.today() - timedelta(days=10)
    await _seed_decision(db, decision_date, "XLE", "SELL")

    tracker = DecisionQualityTracker(db)
    tracker._fetch_prices = AsyncMock(
        return_value={
            "XLE": {
                decision_date: 100.0,
                decision_date + timedelta(days=1): 97.0,
                decision_date + timedelta(days=3): 95.0,
                decision_date + timedelta(days=5): 92.0,
            }
        }
    )

    qualities = await tracker.analyze_recent(days=30, tenant_id="default")
    assert len(qualities) == 1
    assert qualities[0].favorable_1d is True  # Price went down = favorable for SELL
    assert qualities[0].fwd_1d == -3.0


@pytest.mark.asyncio
async def test_summary_percentages(db):
    """Verify summary computes favorable percentages correctly."""
    qualities = [
        DecisionQuality(
            date=date.today(),
            ticker="A",
            side="BUY",
            fwd_1d=2.0,
            fwd_3d=3.0,
            fwd_5d=-1.0,
            favorable_1d=True,
            favorable_3d=True,
            favorable_5d=False,
        ),
        DecisionQuality(
            date=date.today(),
            ticker="B",
            side="BUY",
            fwd_1d=-1.0,
            fwd_3d=2.0,
            fwd_5d=5.0,
            favorable_1d=False,
            favorable_3d=True,
            favorable_5d=True,
        ),
    ]
    summary = DecisionQualityTracker.summarize(qualities)
    assert summary.total_decisions == 2
    assert summary.favorable_1d_pct == 50.0
    assert summary.favorable_3d_pct == 100.0
    assert summary.favorable_5d_pct == 50.0


def test_format_for_prompt():
    from src.analysis.decision_quality import DecisionQualitySummary

    summary = DecisionQualitySummary(
        total_decisions=10,
        favorable_1d_pct=60.0,
        favorable_3d_pct=70.0,
        favorable_5d_pct=55.0,
    )
    text = DecisionQualityTracker.format_for_prompt(summary)
    assert "10 trades" in text
    assert "60%" in text
    assert "70%" in text


def test_format_for_prompt_empty():
    from src.analysis.decision_quality import DecisionQualitySummary

    summary = DecisionQualitySummary(
        total_decisions=0,
        favorable_1d_pct=0.0,
        favorable_3d_pct=0.0,
        favorable_5d_pct=0.0,
    )
    text = DecisionQualityTracker.format_for_prompt(summary)
    assert "No decisions" in text


@pytest.mark.asyncio
async def test_partial_forward_data(db):
    """Handles cases where not all forward days have data."""
    decision_date = date.today() - timedelta(days=3)
    await _seed_decision(db, decision_date, "XLK", "BUY")

    tracker = DecisionQualityTracker(db)
    # Only 1d forward data available
    tracker._fetch_prices = AsyncMock(
        return_value={
            "XLK": {
                decision_date: 100.0,
                decision_date + timedelta(days=1): 103.0,
                # No 3d or 5d data
            }
        }
    )

    qualities = await tracker.analyze_recent(days=30, tenant_id="default")
    assert len(qualities) == 1
    assert qualities[0].fwd_1d == 3.0
    assert qualities[0].fwd_3d is None
    assert qualities[0].fwd_5d is None
