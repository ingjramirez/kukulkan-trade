"""Tests for TrackRecord — win rate computation and prompt formatting."""

from src.analysis.outcome_tracker import TradeOutcome
from src.analysis.track_record import TrackRecord


def _make_outcome(
    ticker: str = "XLK",
    pnl_pct: float = 5.0,
    sector: str = "Technology",
    conviction: str = "high",
    alpha_vs_spy: float | None = 2.0,
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
    stats = tr.compute(outcomes)
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
    stats = tr.compute(outcomes)
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
