"""Tests for PlaybookGenerator — regime x sector playbook matrix."""

from src.analysis.outcome_tracker import TradeOutcome
from src.analysis.playbook_generator import PlaybookCell, PlaybookGenerator


def _make_outcome(
    sector: str = "Technology",
    regime: str = "BULL",
    pnl_pct: float = 2.0,
    conviction: str = "medium",
) -> TradeOutcome:
    return TradeOutcome(
        ticker="AAPL",
        side="BUY",
        entry_price=100,
        current_price=102,
        exit_price=None,
        pnl_pct=pnl_pct,
        hold_days=5,
        sector=sector,
        sector_etf_pct=None,
        spy_pct=None,
        alpha_vs_sector=None,
        alpha_vs_spy=None,
        conviction=conviction,
        reasoning="test",
        regime_at_entry=regime,
    )


def test_empty_outcomes():
    gen = PlaybookGenerator()
    cells = gen.generate([])
    assert cells == []


def test_single_group_sweet_spot():
    """12 outcomes in BULL/Technology: 10 wins, 1 loss, 1 scratch -> sweet_spot."""
    outcomes = (
        [_make_outcome(pnl_pct=3.0) for _ in range(10)]  # wins (pnl > 0.5)
        + [_make_outcome(pnl_pct=-2.0)]  # loss (pnl < -0.5)
        + [_make_outcome(pnl_pct=0.1)]  # scratch (-0.5 <= pnl <= 0.5)
    )
    gen = PlaybookGenerator()
    cells = gen.generate(outcomes)

    assert len(cells) == 1
    cell = cells[0]
    assert cell.regime == "BULL"
    assert cell.sector == "Technology"
    assert cell.total == 12
    assert cell.wins == 10
    assert cell.losses == 1
    # win_rate = 10 / (10 + 1) * 100 = 90.9%
    assert cell.win_rate_pct == 90.9
    assert cell.recommendation == "sweet_spot"


def test_single_group_avoid():
    """12 outcomes: 3 wins, 8 losses, 1 scratch -> avoid (<45% WR)."""
    outcomes = (
        [_make_outcome(pnl_pct=3.0) for _ in range(3)]  # wins
        + [_make_outcome(pnl_pct=-2.0) for _ in range(8)]  # losses
        + [_make_outcome(pnl_pct=0.1)]  # scratch
    )
    gen = PlaybookGenerator()
    cells = gen.generate(outcomes)

    assert len(cells) == 1
    cell = cells[0]
    assert cell.total == 12
    assert cell.wins == 3
    assert cell.losses == 8
    # win_rate = 3 / (3 + 8) * 100 = 27.3%
    assert cell.win_rate_pct == 27.3
    assert cell.recommendation == "avoid"


def test_insufficient_data():
    """5 outcomes (< MIN_TRADES_PER_CELL=10) -> insufficient_data."""
    outcomes = [_make_outcome(pnl_pct=3.0) for _ in range(5)]
    gen = PlaybookGenerator()
    cells = gen.generate(outcomes)

    assert len(cells) == 1
    cell = cells[0]
    assert cell.total == 5
    assert cell.recommendation == "insufficient_data"


def test_multiple_groups():
    """Outcomes in BULL/Technology and BEAR/Healthcare -> 2 cells."""
    bull_tech = [_make_outcome(sector="Technology", regime="BULL", pnl_pct=3.0) for _ in range(12)]
    bear_health = [_make_outcome(sector="Healthcare", regime="BEAR", pnl_pct=-2.0) for _ in range(12)]
    outcomes = bull_tech + bear_health

    gen = PlaybookGenerator()
    cells = gen.generate(outcomes)

    assert len(cells) == 2
    regimes_sectors = {(c.regime, c.sector) for c in cells}
    assert ("BULL", "Technology") in regimes_sectors
    assert ("BEAR", "Healthcare") in regimes_sectors


def test_none_regime_becomes_unknown():
    """Outcomes with regime_at_entry=None are grouped under 'Unknown'."""
    outcomes = [_make_outcome(regime=None, pnl_pct=3.0) for _ in range(12)]  # type: ignore[arg-type]
    gen = PlaybookGenerator()
    cells = gen.generate(outcomes)

    assert len(cells) == 1
    assert cells[0].regime == "Unknown"


def test_format_for_prompt_shows_actionable():
    """Only sweet_spot, solid, and avoid cells appear in formatted output; neutral does not."""
    cells = [
        PlaybookCell(
            regime="BULL",
            sector="Technology",
            total=20,
            wins=15,
            losses=3,
            win_rate_pct=83.3,
            avg_pnl_pct=3.5,
            recommendation="sweet_spot",
        ),
        PlaybookCell(
            regime="BEAR",
            sector="Healthcare",
            total=15,
            wins=3,
            losses=10,
            win_rate_pct=23.1,
            avg_pnl_pct=-1.2,
            recommendation="avoid",
        ),
        PlaybookCell(
            regime="SIDEWAYS",
            sector="Energy",
            total=18,
            wins=9,
            losses=8,
            win_rate_pct=52.9,
            avg_pnl_pct=0.3,
            recommendation="neutral",
        ),
    ]
    gen = PlaybookGenerator()
    text = gen.format_for_prompt(cells)

    assert "## Empirical Playbook" in text
    assert "BULL/Technology" in text
    assert "sweet_spot" in text
    assert "BEAR/Healthcare" in text
    assert "avoid" in text
    # neutral should NOT appear
    assert "SIDEWAYS/Energy" not in text
    assert "neutral" not in text


def test_format_for_prompt_empty():
    """No actionable cells returns empty string."""
    gen = PlaybookGenerator()
    # Empty list
    assert gen.format_for_prompt([]) == ""
    # Only neutral cells
    cells = [
        PlaybookCell(
            regime="BULL",
            sector="Technology",
            total=15,
            wins=8,
            losses=7,
            win_rate_pct=53.3,
            avg_pnl_pct=0.5,
            recommendation="neutral",
        ),
    ]
    assert gen.format_for_prompt(cells) == ""
