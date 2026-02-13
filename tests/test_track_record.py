"""Tests for TrackRecord — win rate computation and prompt formatting."""

from src.analysis.outcome_tracker import TradeOutcome
from src.analysis.track_record import TrackRecord


def _make_outcome(
    ticker: str = "XLK",
    pnl_pct: float = 5.0,
    sector: str = "Technology",
    conviction: str = "high",
    alpha_vs_spy: float | None = 2.0,
    regime_at_entry: str | None = None,
    session_at_entry: str | None = None,
) -> TradeOutcome:
    return TradeOutcome(
        ticker=ticker,
        side="BUY",
        entry_price=100.0,
        current_price=100.0 + pnl_pct,
        exit_price=None,
        pnl_pct=pnl_pct,
        hold_days=5,
        sector=sector,
        sector_etf_pct=3.0,
        spy_pct=2.0,
        alpha_vs_sector=pnl_pct - 3.0,
        alpha_vs_spy=alpha_vs_spy,
        conviction=conviction,
        reasoning="test",
        regime_at_entry=regime_at_entry,
        session_at_entry=session_at_entry,
    )


def test_empty_outcomes():
    tr = TrackRecord()
    stats = tr.compute([])
    assert stats.total_trades == 0
    assert stats.win_rate_pct == 0.0
    assert stats.by_sector == []


def test_overall_stats():
    outcomes = [
        _make_outcome(pnl_pct=5.0),  # win
        _make_outcome(pnl_pct=-3.0),  # loss
        _make_outcome(pnl_pct=0.2),  # scratch
        _make_outcome(pnl_pct=8.0),  # win
    ]
    tr = TrackRecord()
    stats = tr.compute(outcomes)
    assert stats.total_trades == 4
    assert stats.wins == 2
    assert stats.losses == 1
    assert stats.scratches == 1
    # Win rate = 2/(2+1) = 66.7%
    assert stats.win_rate_pct == 66.7
    # Avg P&L = (5 + -3 + 0.2 + 8) / 4 = 2.55
    assert stats.avg_pnl_pct == 2.55


def test_by_sector():
    outcomes = [
        _make_outcome(ticker="XLK", sector="Technology", pnl_pct=5.0),
        _make_outcome(ticker="XLE", sector="Energy", pnl_pct=-2.0),
        _make_outcome(ticker="AAPL", sector="Technology", pnl_pct=3.0),
    ]
    tr = TrackRecord()
    stats = tr.compute(outcomes, min_trades=1)
    assert len(stats.by_sector) == 2
    tech = next(s for s in stats.by_sector if s.value == "Technology")
    assert tech.total == 2
    assert tech.wins == 2
    assert tech.losses == 0


def test_by_conviction():
    outcomes = [
        _make_outcome(conviction="high", pnl_pct=5.0),
        _make_outcome(conviction="high", pnl_pct=3.0),
        _make_outcome(conviction="low", pnl_pct=-2.0),
    ]
    tr = TrackRecord()
    stats = tr.compute(outcomes, min_trades=1)
    high = next(c for c in stats.by_conviction if c.value == "high")
    low = next(c for c in stats.by_conviction if c.value == "low")
    assert high.wins == 2
    assert low.losses == 1


def test_best_worst_sector():
    outcomes = [
        _make_outcome(sector="Technology", pnl_pct=10.0),
        _make_outcome(sector="Energy", pnl_pct=-5.0),
        _make_outcome(sector="Financials", pnl_pct=2.0),
    ]
    tr = TrackRecord()
    stats = tr.compute(outcomes)
    assert stats.best_sector == "Technology"
    assert stats.worst_sector == "Energy"


def test_format_for_prompt():
    outcomes = [
        _make_outcome(pnl_pct=5.0, conviction="high"),
        _make_outcome(pnl_pct=-3.0, conviction="low"),
        _make_outcome(pnl_pct=8.0, conviction="high"),
    ]
    tr = TrackRecord()
    stats = tr.compute(outcomes)
    text = tr.format_for_prompt(stats)
    assert "Win rate:" in text
    assert "Avg P&L:" in text
    assert "alpha" in text.lower()


def test_format_for_prompt_empty():
    tr = TrackRecord()
    stats = tr.compute([])
    text = tr.format_for_prompt(stats)
    assert "No completed trades" in text


def test_single_trade():
    outcomes = [_make_outcome(pnl_pct=2.0)]
    tr = TrackRecord()
    stats = tr.compute(outcomes)
    assert stats.total_trades == 1
    assert stats.wins == 1
    assert stats.win_rate_pct == 100.0


def test_by_regime():
    """Outcomes with regime_at_entry are grouped."""
    outcomes = [
        _make_outcome(pnl_pct=5.0, regime_at_entry="BULL"),
        _make_outcome(pnl_pct=-3.0, regime_at_entry="BEAR"),
        _make_outcome(pnl_pct=3.0, regime_at_entry="BULL"),
        _make_outcome(pnl_pct=2.0),  # None regime — excluded
    ]
    tr = TrackRecord()
    stats = tr.compute(outcomes, min_trades=1)
    assert len(stats.by_regime) == 2
    bull = next(r for r in stats.by_regime if r.value == "BULL")
    assert bull.total == 2
    assert bull.wins == 2


def test_by_session():
    """Outcomes with session_at_entry are grouped."""
    outcomes = [
        _make_outcome(pnl_pct=5.0, session_at_entry="Morning"),
        _make_outcome(pnl_pct=3.0, session_at_entry="Morning"),
        _make_outcome(pnl_pct=-2.0, session_at_entry="Closing"),
        _make_outcome(pnl_pct=1.0),  # None session — excluded
    ]
    tr = TrackRecord()
    stats = tr.compute(outcomes, min_trades=1)
    assert len(stats.by_session) == 2
    morning = next(s for s in stats.by_session if s.value == "Morning")
    assert morning.total == 2


def test_null_regime_session_excluded():
    """Outcomes with None regime/session are excluded from those groups."""
    outcomes = [_make_outcome(pnl_pct=5.0)]  # No regime/session
    tr = TrackRecord()
    stats = tr.compute(outcomes, min_trades=1)
    assert stats.by_regime == []
    assert stats.by_session == []


def test_min_trades_filters_small_groups():
    """Groups with fewer than min_trades are filtered out."""
    outcomes = [
        _make_outcome(ticker="A", pnl_pct=5.0, sector="Technology"),
        _make_outcome(ticker="B", pnl_pct=3.0, sector="Technology"),
        _make_outcome(ticker="C", pnl_pct=-2.0, sector="Energy"),  # Only 1 trade
    ]
    tr = TrackRecord()
    # min_trades=2: Energy has only 1 trade, should be filtered
    stats = tr.compute(outcomes, min_trades=2)
    assert len(stats.by_sector) == 1
    assert stats.by_sector[0].value == "Technology"
    # But best/worst sector still comes from unfiltered
    assert stats.best_sector == "Technology"
    assert stats.worst_sector == "Energy"


def test_default_min_trades_is_5():
    """Default min_trades=5 filters categories with fewer trades."""
    outcomes = [_make_outcome(pnl_pct=5.0, sector="Tech") for _ in range(3)]
    tr = TrackRecord()
    stats = tr.compute(outcomes)  # default min_trades=5
    assert stats.by_sector == []  # 3 < 5, filtered out
    assert stats.best_sector == "Tech"  # Still populated from unfiltered


def test_format_includes_n_equals():
    """Formatted output includes n= for sample size."""
    outcomes = [
        _make_outcome(pnl_pct=5.0, conviction="high"),
        _make_outcome(pnl_pct=-3.0, conviction="high"),
        _make_outcome(pnl_pct=8.0, conviction="high"),
        _make_outcome(pnl_pct=2.0, conviction="high"),
        _make_outcome(pnl_pct=1.0, conviction="high"),
    ]
    tr = TrackRecord()
    stats = tr.compute(outcomes, min_trades=1)
    text = tr.format_for_prompt(stats)
    assert "n=" in text


def test_format_includes_regime_session():
    """Formatted output includes regime and session breakdowns."""
    outcomes = [
        _make_outcome(pnl_pct=5.0, regime_at_entry="BULL", session_at_entry="Morning"),
    ] * 6
    tr = TrackRecord()
    stats = tr.compute(outcomes, min_trades=1)
    text = tr.format_for_prompt(stats)
    assert "By regime:" in text
    assert "By session:" in text
    assert "BULL" in text
    assert "Morning" in text
