"""Tests for AutoApplyEngine — applies improvement proposals to tenant config."""

import json
from datetime import date

import pytest

from src.analysis.auto_apply import FLIP_FLOP_THRESHOLD, AutoApplyEngine
from src.analysis.weekly_improvement import ImprovementProposal, ProposedChange
from src.storage.database import Database


@pytest.fixture
async def db():
    d = Database(url="sqlite+aiosqlite:///:memory:")
    await d.init_db()
    yield d
    await d.close()


# ── Strategy Mode ─────────────────────────────────────────────────


async def test_apply_strategy_mode(db: Database):
    engine = AutoApplyEngine(db)
    proposal = ImprovementProposal(
        changes=[
            ProposedChange(
                category="strategy_mode",
                parameter="strategy_mode",
                old_value="conservative",
                new_value="standard",
                reason="Better win rate supports it",
            )
        ],
        summary="upgrade strategy",
    )

    results = await engine.apply("default", proposal)
    assert len(results) == 1
    assert results[0]["status"] == "applied"
    assert results[0]["new_value"] == "standard"

    tenant = await db.get_tenant("default")
    assert tenant.strategy_mode == "standard"


async def test_apply_strategy_mode_changelog(db: Database):
    engine = AutoApplyEngine(db)
    snap_id = await db.save_improvement_snapshot(
        tenant_id="default",
        week_start=date(2026, 2, 10),
        week_end=date(2026, 2, 17),
        total_trades=10,
        win_rate_pct=60.0,
        avg_pnl_pct=1.0,
        avg_alpha_vs_spy=0.5,
        total_cost_usd=1.0,
        strategy_mode="conservative",
        trailing_stop_multiplier=1.0,
        proposal_json=None,
        applied_changes=None,
        report_text=None,
    )

    proposal = ImprovementProposal(
        changes=[
            ProposedChange(
                category="strategy_mode",
                parameter="strategy_mode",
                old_value="conservative",
                new_value="standard",
                reason="test reason",
            )
        ],
    )

    await engine.apply("default", proposal, snapshot_id=snap_id)

    changelog = await db.get_parameter_changelog("default")
    assert len(changelog) == 1
    assert changelog[0].parameter == "strategy_mode"
    assert changelog[0].snapshot_id == snap_id


# ── Trailing Stop Multiplier ─────────────────────────────────────


async def test_apply_trailing_stop(db: Database):
    engine = AutoApplyEngine(db)
    proposal = ImprovementProposal(
        changes=[
            ProposedChange(
                category="trailing_stop",
                parameter="trailing_stop_multiplier",
                old_value="1.0",
                new_value="0.8",
                reason="Tighter stops for volatile market",
            )
        ],
    )

    results = await engine.apply("default", proposal)
    assert results[0]["status"] == "applied"

    tenant = await db.get_tenant("default")
    assert tenant.trailing_stop_multiplier == 0.8


# ── Universe Exclude ──────────────────────────────────────────────


async def test_apply_universe_exclude(db: Database):
    engine = AutoApplyEngine(db)
    proposal = ImprovementProposal(
        changes=[
            ProposedChange(
                category="universe_exclude",
                parameter="ticker_exclusion:TSLA",
                old_value=None,
                new_value="TSLA",
                reason="Consistent loser",
            )
        ],
    )

    results = await engine.apply("default", proposal)
    assert results[0]["status"] == "applied"

    tenant = await db.get_tenant("default")
    exclusions = json.loads(tenant.ticker_exclusions)
    assert "TSLA" in exclusions


async def test_apply_universe_exclude_appends(db: Database):
    """Verify new exclusions are appended to existing ones."""
    await db.update_tenant("default", {"ticker_exclusions": json.dumps(["META"])})

    engine = AutoApplyEngine(db)
    proposal = ImprovementProposal(
        changes=[
            ProposedChange(
                category="universe_exclude",
                parameter="ticker_exclusion:TSLA",
                old_value=None,
                new_value="TSLA",
                reason="test",
            )
        ],
    )

    await engine.apply("default", proposal)

    tenant = await db.get_tenant("default")
    exclusions = json.loads(tenant.ticker_exclusions)
    assert "META" in exclusions
    assert "TSLA" in exclusions


async def test_apply_universe_exclude_no_duplicate(db: Database):
    """If ticker already excluded, don't add again."""
    await db.update_tenant("default", {"ticker_exclusions": json.dumps(["TSLA"])})

    engine = AutoApplyEngine(db)
    proposal = ImprovementProposal(
        changes=[
            ProposedChange(
                category="universe_exclude",
                parameter="ticker_exclusion:TSLA",
                old_value=None,
                new_value="TSLA",
                reason="already there",
            )
        ],
    )

    await engine.apply("default", proposal)

    tenant = await db.get_tenant("default")
    exclusions = json.loads(tenant.ticker_exclusions)
    assert exclusions.count("TSLA") == 1


# ── Learning ──────────────────────────────────────────────────────


async def test_apply_learning(db: Database):
    engine = AutoApplyEngine(db)
    proposal = ImprovementProposal(
        changes=[
            ProposedChange(
                category="learning",
                parameter="sector_insight",
                old_value=None,
                new_value="Tech outperforms in bull regime",
                reason="observed pattern",
            )
        ],
    )

    results = await engine.apply("default", proposal)
    assert results[0]["status"] == "applied"

    # Check memory was saved
    memories = await db.get_agent_memories("agent_note", tenant_id="default")
    keys = [m.key for m in memories]
    assert "learning:sector_insight" in keys


# ── Flip-Flop Protection ─────────────────────────────────────────


async def test_flip_flop_blocks_strategy_change(db: Database):
    """After 3+ strategy changes in 4 weeks, further changes are blocked."""
    engine = AutoApplyEngine(db)

    # Pre-populate changelog with FLIP_FLOP_THRESHOLD changes
    for i in range(FLIP_FLOP_THRESHOLD):
        await db.insert_parameter_changelog(
            tenant_id="default",
            parameter="strategy_mode",
            old_value=f"mode_{i}",
            new_value=f"mode_{i + 1}",
        )

    proposal = ImprovementProposal(
        changes=[
            ProposedChange(
                category="strategy_mode",
                parameter="strategy_mode",
                old_value="conservative",
                new_value="standard",
                reason="should be blocked",
            )
        ],
    )

    results = await engine.apply("default", proposal)
    assert results[0]["status"] == "blocked_flipflop"

    # Verify tenant was NOT changed
    tenant = await db.get_tenant("default")
    assert tenant.strategy_mode == "conservative"


async def test_flip_flop_blocks_trailing_stop(db: Database):
    engine = AutoApplyEngine(db)

    for i in range(FLIP_FLOP_THRESHOLD):
        await db.insert_parameter_changelog(
            tenant_id="default",
            parameter="trailing_stop_multiplier",
            old_value=str(0.8 + i * 0.1),
            new_value=str(0.9 + i * 0.1),
        )

    proposal = ImprovementProposal(
        changes=[
            ProposedChange(
                category="trailing_stop",
                parameter="trailing_stop_multiplier",
                old_value="1.0",
                new_value="0.7",
                reason="should be blocked",
            )
        ],
    )

    results = await engine.apply("default", proposal)
    assert results[0]["status"] == "blocked_flipflop"


async def test_flip_flop_does_not_block_learnings(db: Database):
    """Learnings skip flip-flop detection."""
    engine = AutoApplyEngine(db)

    proposal = ImprovementProposal(
        changes=[
            ProposedChange(
                category="learning",
                parameter="insight_1",
                old_value=None,
                new_value="Some insight",
                reason="test",
            )
        ],
    )

    results = await engine.apply("default", proposal)
    assert results[0]["status"] == "applied"


async def test_flip_flop_does_not_block_exclusions(db: Database):
    """Universe exclusions skip flip-flop detection."""
    engine = AutoApplyEngine(db)

    proposal = ImprovementProposal(
        changes=[
            ProposedChange(
                category="universe_exclude",
                parameter="ticker_exclusion:BAD",
                old_value=None,
                new_value="BAD",
                reason="test",
            )
        ],
    )

    results = await engine.apply("default", proposal)
    assert results[0]["status"] == "applied"


# ── Unknown Category ─────────────────────────────────────────────


async def test_apply_unknown_category(db: Database):
    engine = AutoApplyEngine(db)
    proposal = ImprovementProposal(
        changes=[
            ProposedChange(
                category="unknown_cat",
                parameter="something",
                old_value=None,
                new_value="value",
                reason="test",
            )
        ],
    )

    results = await engine.apply("default", proposal)
    assert results[0]["status"] == "unknown_category"


# ── Multiple Changes ─────────────────────────────────────────────


async def test_apply_multiple_changes(db: Database):
    engine = AutoApplyEngine(db)
    proposal = ImprovementProposal(
        changes=[
            ProposedChange(
                category="strategy_mode",
                parameter="strategy_mode",
                old_value="conservative",
                new_value="standard",
                reason="higher WR",
            ),
            ProposedChange(
                category="trailing_stop",
                parameter="trailing_stop_multiplier",
                old_value="1.0",
                new_value="0.9",
                reason="slightly tighter",
            ),
            ProposedChange(
                category="learning",
                parameter="weekly_note",
                old_value=None,
                new_value="Market liked the data",
                reason="observation",
            ),
        ],
    )

    results = await engine.apply("default", proposal)
    applied = [r for r in results if r["status"] == "applied"]
    assert len(applied) == 3

    tenant = await db.get_tenant("default")
    assert tenant.strategy_mode == "standard"
    assert tenant.trailing_stop_multiplier == 0.9


# ── _get_trail_pct with multiplier ───────────────────────────────


def test_get_trail_pct_with_multiplier():
    # Import from orchestrator
    from src.orchestrator import _get_trail_pct
    from src.storage.models import TradeSchema

    trade = TradeSchema(
        portfolio="B",
        ticker="AAPL",
        side="BUY",
        shares=10,
        price=150.0,
        reason="high conviction buy",
    )

    # Default multiplier (1.0)
    pct = _get_trail_pct("conservative", trade, 1.0)
    assert pct == 0.05  # conservative + high conviction

    # Tighter (0.5x)
    pct = _get_trail_pct("conservative", trade, 0.5)
    assert pct == 0.025

    # Wider (2.0x)
    pct = _get_trail_pct("conservative", trade, 2.0)
    assert pct == 0.1


def test_get_trail_pct_default_multiplier():
    from src.orchestrator import _get_trail_pct
    from src.storage.models import TradeSchema

    trade = TradeSchema(
        portfolio="B",
        ticker="AAPL",
        side="BUY",
        shares=10,
        price=150.0,
        reason="medium conviction",
    )

    # No multiplier arg = 1.0 default
    pct = _get_trail_pct("conservative", trade)
    assert pct == 0.07  # conservative + medium
