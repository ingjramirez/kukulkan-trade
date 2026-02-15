"""Tests for ConvictionCalibrator — accuracy per conviction level."""

from src.analysis.conviction_calibrator import ConvictionBucket, ConvictionCalibrator
from src.analysis.outcome_tracker import TradeOutcome


def _make_outcome(
    conviction: str = "high",
    pnl_pct: float = 2.0,
    sector: str = "Technology",
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
    )


def test_empty_outcomes():
    cal = ConvictionCalibrator()
    buckets = cal.calibrate([])
    assert buckets == []


def test_high_conviction_validated():
    """20 high-conviction outcomes: 15 wins, 3 losses, 2 scratches -> validated, 1.2x."""
    outcomes = (
        [_make_outcome(conviction="high", pnl_pct=3.0) for _ in range(15)]  # wins
        + [_make_outcome(conviction="high", pnl_pct=-2.0) for _ in range(3)]  # losses
        + [_make_outcome(conviction="high", pnl_pct=0.1) for _ in range(2)]  # scratches
    )
    cal = ConvictionCalibrator()
    buckets = cal.calibrate(outcomes)

    assert len(buckets) == 1
    b = buckets[0]
    assert b.conviction == "high"
    assert b.total == 20
    assert b.wins == 15
    assert b.losses == 3
    # win_rate = 15 / (15 + 3) * 100 = 83.3% (> 60%)
    assert b.win_rate_pct == 83.3
    # avg_pnl = (15*3.0 + 3*-2.0 + 2*0.1) / 20 = (45 - 6 + 0.2) / 20 = 1.96 (> 1.0)
    assert b.avg_pnl_pct == 1.96
    assert b.assessment == "validated"
    assert b.suggested_multiplier == 1.2


def test_high_conviction_overconfident():
    """20 high-conviction outcomes: 8 wins, 10 losses, 2 scratches -> overconfident, 0.7x."""
    outcomes = (
        [_make_outcome(conviction="high", pnl_pct=3.0) for _ in range(8)]  # wins
        + [_make_outcome(conviction="high", pnl_pct=-2.0) for _ in range(10)]  # losses
        + [_make_outcome(conviction="high", pnl_pct=0.1) for _ in range(2)]  # scratches
    )
    cal = ConvictionCalibrator()
    buckets = cal.calibrate(outcomes)

    assert len(buckets) == 1
    b = buckets[0]
    assert b.conviction == "high"
    assert b.total == 20
    assert b.wins == 8
    assert b.losses == 10
    # win_rate = 8 / (8 + 10) * 100 = 44.4% (< 50%)
    assert b.win_rate_pct == 44.4
    assert b.assessment == "overconfident"
    assert b.suggested_multiplier == 0.7


def test_low_conviction_underconfident():
    """20 low-conviction outcomes: 15 wins, 3 losses, avg_pnl > 1% -> underconfident, 1.3x."""
    outcomes = (
        [_make_outcome(conviction="low", pnl_pct=3.0) for _ in range(15)]  # wins
        + [_make_outcome(conviction="low", pnl_pct=-2.0) for _ in range(3)]  # losses
        + [_make_outcome(conviction="low", pnl_pct=0.1) for _ in range(2)]  # scratches
    )
    cal = ConvictionCalibrator()
    buckets = cal.calibrate(outcomes)

    assert len(buckets) == 1
    b = buckets[0]
    assert b.conviction == "low"
    assert b.total == 20
    assert b.wins == 15
    assert b.losses == 3
    # win_rate = 15 / (15 + 3) * 100 = 83.3% (> 60%)
    assert b.win_rate_pct == 83.3
    # avg_pnl = 1.96 (> 1.0)
    assert b.avg_pnl_pct == 1.96
    assert b.assessment == "underconfident"
    assert b.suggested_multiplier == 1.3


def test_medium_conviction_neutral():
    """20 medium-conviction outcomes with moderate win rate -> neutral, 1.0x."""
    outcomes = (
        [_make_outcome(conviction="medium", pnl_pct=1.0) for _ in range(10)]  # wins
        + [_make_outcome(conviction="medium", pnl_pct=-1.0) for _ in range(8)]  # losses
        + [_make_outcome(conviction="medium", pnl_pct=0.1) for _ in range(2)]  # scratches
    )
    cal = ConvictionCalibrator()
    buckets = cal.calibrate(outcomes)

    assert len(buckets) == 1
    b = buckets[0]
    assert b.conviction == "medium"
    assert b.total == 20
    assert b.wins == 10
    assert b.losses == 8
    # win_rate = 10 / (10 + 8) * 100 = 55.6% (not > 60%, not < 45%)
    assert b.win_rate_pct == 55.6
    assert b.assessment == "neutral"
    assert b.suggested_multiplier == 1.0


def test_insufficient_trades():
    """10 outcomes (< MIN_TRADES_PER_BUCKET=15) -> insufficient."""
    outcomes = [_make_outcome(conviction="high", pnl_pct=3.0) for _ in range(10)]
    cal = ConvictionCalibrator()
    buckets = cal.calibrate(outcomes)

    assert len(buckets) == 1
    b = buckets[0]
    assert b.total == 10
    assert b.assessment == "insufficient"
    assert b.suggested_multiplier == 1.0


def test_format_for_prompt():
    """Formatted output has header and shows assessment details."""
    buckets = [
        ConvictionBucket(
            conviction="high",
            total=20,
            wins=15,
            losses=3,
            win_rate_pct=83.3,
            avg_pnl_pct=1.96,
            assessment="validated",
            suggested_multiplier=1.2,
        ),
        ConvictionBucket(
            conviction="low",
            total=18,
            wins=5,
            losses=10,
            win_rate_pct=33.3,
            avg_pnl_pct=-0.5,
            assessment="neutral",
            suggested_multiplier=1.0,
        ),
    ]
    cal = ConvictionCalibrator()
    text = cal.format_for_prompt(buckets)

    assert "## Conviction Calibration" in text
    assert "high" in text
    assert "validated" in text
    assert "low" in text
    assert "neutral" in text


def test_format_for_prompt_empty():
    """Empty buckets returns empty string."""
    cal = ConvictionCalibrator()
    assert cal.format_for_prompt([]) == ""
