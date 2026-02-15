"""Tests for Phase 2 portfolio tools — upgraded + new tools.

Tests cover: get_portfolio_state, get_position_detail, get_portfolio_performance,
get_historical_trades, get_correlation_check, get_risk_assessment, and legacy aliases.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.agent.tools import ToolRegistry
from src.agent.tools.portfolio import (
    _get_correlation_check,
    _get_historical_trades,
    _get_portfolio_performance,
    _get_portfolio_state,
    _get_position_detail,
    _get_risk_assessment,
    register_portfolio_tools,
)
from src.storage.database import Database


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


@pytest.fixture
def closes():
    """Synthetic close price DataFrame (100 trading days, 5 tickers)."""
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    np.random.seed(42)
    data = {
        "XLK": 200 + np.cumsum(np.random.normal(0, 1, 100)),
        "XLE": 80 + np.cumsum(np.random.normal(0, 0.5, 100)),
        "NVDA": 120 + np.cumsum(np.random.normal(0, 2, 100)),
        "GLD": 180 + np.cumsum(np.random.normal(0, 0.3, 100)),
        "SPY": 450 + np.cumsum(np.random.normal(0, 1.5, 100)),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def current_prices(closes: pd.DataFrame) -> dict[str, float]:
    """Extract latest prices from closes fixture."""
    return {t: float(closes[t].iloc[-1]) for t in closes.columns}


async def _seed_portfolio(db: Database, cash: float = 50000.0) -> None:
    await db.upsert_portfolio("B", cash=cash, total_value=cash)


async def _seed_positions(db: Database) -> None:
    await db.upsert_position("B", "XLK", shares=100, avg_price=200.0)
    await db.upsert_position("B", "XLE", shares=50, avg_price=80.0)
    await db.upsert_position("B", "NVDA", shares=30, avg_price=110.0)


async def _seed_trailing_stop(db: Database, ticker: str = "XLK") -> None:
    await db.create_trailing_stop("default", "B", ticker, entry_price=200.0, trail_pct=0.07)


async def _seed_trades(db: Database, days_ago: int = 5) -> None:
    await db.log_trade("B", "XLK", "BUY", 100, 195.0, reason="Momentum", tenant_id="default")
    await db.log_trade("B", "XLE", "BUY", 50, 78.0, reason="Energy rotation", tenant_id="default")
    await db.log_trade("B", "NVDA", "BUY", 30, 110.0, reason="AI thesis", tenant_id="default")


async def _seed_snapshots(db: Database, count: int = 10) -> None:
    base_value = 66000.0
    for i in range(count):
        snap_date = date.today() - timedelta(days=count - i)
        daily_return = (-1) ** i * 0.5  # Alternating +/- 0.5%
        base_value *= 1 + daily_return / 100
        await db.save_snapshot(
            "B",
            snap_date,
            total_value=base_value,
            cash=20000.0,
            positions_value=base_value - 20000.0,
            daily_return_pct=daily_return,
            tenant_id="default",
        )


# ── get_portfolio_state ──────────────────────────────────────────────────────


async def test_portfolio_state_empty(db: Database):
    """get_portfolio_state returns empty positions when portfolio has no positions."""
    await _seed_portfolio(db, cash=66000.0)
    result = await _get_portfolio_state(db, "default", {})
    assert result["cash"] == 66000.0
    assert result["cash_pct"] == 100.0
    assert result["position_count"] == 0
    assert result["positions"] == []


async def test_portfolio_state_with_positions(db: Database, current_prices: dict):
    """get_portfolio_state returns positions with P&L and sector exposure."""
    await _seed_portfolio(db, cash=20000.0)
    await _seed_positions(db)

    result = await _get_portfolio_state(db, "default", current_prices)
    assert result["position_count"] == 3
    assert result["cash"] == 20000.0
    assert len(result["positions"]) == 3
    assert result["sector_exposure"]  # Non-empty

    # Check position details
    tickers = [p["ticker"] for p in result["positions"]]
    assert "XLK" in tickers
    assert "NVDA" in tickers

    # Each position has pnl
    for pos in result["positions"]:
        assert "pnl_pct" in pos
        assert "current_price" in pos
        assert "sector" in pos


async def test_portfolio_state_includes_trailing_stops(db: Database, current_prices: dict):
    """get_portfolio_state includes trailing stop info for positions that have stops."""
    await _seed_portfolio(db, cash=20000.0)
    await _seed_positions(db)
    await _seed_trailing_stop(db, "XLK")

    result = await _get_portfolio_state(db, "default", current_prices)
    xlk_pos = next(p for p in result["positions"] if p["ticker"] == "XLK")
    assert "trailing_stop" in xlk_pos
    assert xlk_pos["trailing_stop"]["trail_pct"] == 0.07

    # Position without stop should not have trailing_stop key
    xle_pos = next(p for p in result["positions"] if p["ticker"] == "XLE")
    assert "trailing_stop" not in xle_pos


# ── get_position_detail ──────────────────────────────────────────────────────


async def test_position_detail_not_found(db: Database):
    """get_position_detail returns error for non-existent position."""
    await _seed_portfolio(db)
    result = await _get_position_detail(db, "default", {}, "NONEXIST")
    assert "error" in result


async def test_position_detail_with_trades(db: Database, current_prices: dict):
    """get_position_detail includes recent trade history for the ticker."""
    await _seed_portfolio(db, cash=20000.0)
    await _seed_positions(db)
    await _seed_trades(db)

    result = await _get_position_detail(db, "default", current_prices, "XLK")
    assert result["ticker"] == "XLK"
    assert result["shares"] == 100
    assert "pnl_pct" in result
    assert "recent_trades" in result
    assert len(result["recent_trades"]) >= 1
    assert result["recent_trades"][0]["side"] == "BUY"


async def test_position_detail_with_stop(db: Database, current_prices: dict):
    """get_position_detail includes trailing stop info."""
    await _seed_portfolio(db, cash=20000.0)
    await _seed_positions(db)
    await _seed_trailing_stop(db, "XLK")

    result = await _get_position_detail(db, "default", current_prices, "XLK")
    assert result["trailing_stop"] is not None
    assert "stop_price" in result["trailing_stop"]
    assert "peak_price" in result["trailing_stop"]
    assert "pct_from_trigger" in result["trailing_stop"]


# ── get_portfolio_performance ────────────────────────────────────────────────


async def test_portfolio_performance_no_snapshots(db: Database, current_prices: dict):
    """get_portfolio_performance returns basic info when no snapshots exist."""
    await _seed_portfolio(db, cash=66000.0)
    result = await _get_portfolio_performance(db, "default", current_prices, period="30d")
    assert result["period"] == "30d"
    assert result["days"] == 30
    assert result["current_value"] == 66000.0
    assert "period_return_pct" not in result  # No snapshots → no return calc


async def test_portfolio_performance_with_snapshots(db: Database, current_prices: dict):
    """get_portfolio_performance computes return and drawdown from snapshots."""
    await _seed_portfolio(db, cash=20000.0)
    await _seed_positions(db)
    await _seed_snapshots(db, count=10)

    result = await _get_portfolio_performance(db, "default", current_prices, period="30d")
    assert "period_return_pct" in result
    assert "max_drawdown_pct" in result
    assert result["max_drawdown_pct"] <= 0  # Drawdown is always negative or zero
    assert "avg_daily_return_pct" in result


async def test_portfolio_performance_with_trades(db: Database, current_prices: dict):
    """get_portfolio_performance includes trade counts."""
    await _seed_portfolio(db, cash=20000.0)
    await _seed_positions(db)
    await _seed_trades(db)

    result = await _get_portfolio_performance(db, "default", current_prices, period="30d")
    assert result["total_trades"] == 3
    assert result["buys_count"] == 3


async def test_portfolio_performance_period_parsing(db: Database, current_prices: dict):
    """get_portfolio_performance accepts different period strings."""
    await _seed_portfolio(db, cash=66000.0)
    for period, expected_days in [("7d", 7), ("14d", 14), ("90d", 90)]:
        result = await _get_portfolio_performance(db, "default", current_prices, period=period)
        assert result["days"] == expected_days


# ── get_historical_trades ────────────────────────────────────────────────────


async def test_historical_trades_empty(db: Database):
    """get_historical_trades returns empty when no trades exist."""
    await _seed_portfolio(db)
    result = await _get_historical_trades(db, "default", days=30)
    assert result["total_trades"] == 0
    assert result["trades"] == []
    assert result["buy_total_usd"] == 0
    assert result["sell_total_usd"] == 0


async def test_historical_trades_with_data(db: Database):
    """get_historical_trades returns trade history with summary stats."""
    await _seed_portfolio(db, cash=20000.0)
    await _seed_positions(db)
    await _seed_trades(db)

    result = await _get_historical_trades(db, "default", days=30)
    assert result["total_trades"] == 3
    assert len(result["trades"]) == 3
    assert result["buy_total_usd"] > 0

    # Check trade format
    trade = result["trades"][0]
    assert "ticker" in trade
    assert "side" in trade
    assert "price" in trade
    assert "sector" in trade


async def test_historical_trades_days_clamping(db: Database):
    """get_historical_trades clamps days to 1-90 range."""
    await _seed_portfolio(db)
    result = await _get_historical_trades(db, "default", days=200)
    assert result["days"] == 90  # Clamped to max

    result = await _get_historical_trades(db, "default", days=-5)
    assert result["days"] == 1  # Clamped to min


# ── get_correlation_check ────────────────────────────────────────────────────


async def test_correlation_check_specific_tickers(closes: pd.DataFrame, current_prices: dict):
    """get_correlation_check returns correlation for specified tickers."""
    result = await _get_correlation_check(closes, current_prices, tickers=["XLK", "NVDA", "GLD"])
    assert "tickers" in result
    assert "high_correlations" in result
    assert "diversification_score" in result
    assert 0.0 <= result["diversification_score"] <= 1.0
    assert "avg_pairwise_correlation" in result


async def test_correlation_check_too_few_tickers(closes: pd.DataFrame, current_prices: dict):
    """get_correlation_check returns error with <2 tickers."""
    result = await _get_correlation_check(closes, current_prices, tickers=["XLK"])
    assert "error" in result


async def test_correlation_check_default_tickers(closes: pd.DataFrame, current_prices: dict):
    """get_correlation_check uses all available tickers when none specified."""
    result = await _get_correlation_check(closes, current_prices, tickers=None)
    assert len(result["tickers"]) == 5  # All 5 tickers from closes fixture


async def test_correlation_check_high_corr_detection(current_prices: dict):
    """get_correlation_check detects high correlations between similar tickers."""
    # Create tickers that are highly correlated (one is a noisy version of the other)
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    np.random.seed(42)
    base = np.cumsum(np.random.normal(0, 1, 100))
    correlated_closes = pd.DataFrame(
        {
            "A": 100 + base,
            "B": 100 + base + np.random.normal(0, 0.1, 100),  # Very similar
            "C": 100 + np.cumsum(np.random.normal(0, 1, 100)),  # Independent
        },
        index=dates,
    )

    result = await _get_correlation_check(correlated_closes, {"A": 100, "B": 100, "C": 100})
    # A and B should be highly correlated
    high = result["high_correlations"]
    assert any("A/B" in c["pair"] for c in high)


# ── get_risk_assessment ──────────────────────────────────────────────────────


async def test_risk_assessment_empty_portfolio(db: Database, closes: pd.DataFrame):
    """get_risk_assessment handles empty portfolio."""
    await _seed_portfolio(db, cash=66000.0)
    result = await _get_risk_assessment(db, "default", {}, closes)
    assert result["cash_pct"] == 100.0
    assert result["position_count"] == 0
    assert result["positions_without_stops"] == []
    assert result["largest_position"] is None


async def test_risk_assessment_with_positions(db: Database, current_prices: dict, closes: pd.DataFrame):
    """get_risk_assessment returns sector concentration and position weights."""
    await _seed_portfolio(db, cash=20000.0)
    await _seed_positions(db)

    result = await _get_risk_assessment(db, "default", current_prices, closes)
    assert result["position_count"] == 3
    assert result["equity_invested_pct"] > 0
    assert len(result["sector_concentration"]) > 0
    assert result["largest_position"] is not None
    assert len(result["position_risks"]) == 3

    # All positions should be unprotected (no stops seeded)
    assert len(result["positions_without_stops"]) == 3


async def test_risk_assessment_with_stops(db: Database, current_prices: dict, closes: pd.DataFrame):
    """get_risk_assessment shows stop distances and identifies unprotected positions."""
    await _seed_portfolio(db, cash=20000.0)
    await _seed_positions(db)
    await _seed_trailing_stop(db, "XLK")

    result = await _get_risk_assessment(db, "default", current_prices, closes)
    # XLK has a stop, XLE and NVDA don't
    assert len(result["positions_without_stops"]) == 2
    assert "XLE" in result["positions_without_stops"]
    assert "NVDA" in result["positions_without_stops"]
    assert "XLK" not in result["positions_without_stops"]

    # Check stop distance in position risks
    xlk_risk = next(r for r in result["position_risks"] if r["ticker"] == "XLK")
    assert xlk_risk["has_trailing_stop"] is True
    assert "pct_from_stop" in xlk_risk


async def test_risk_assessment_volatility_estimate(db: Database, current_prices: dict, closes: pd.DataFrame):
    """get_risk_assessment includes annualized volatility estimate."""
    await _seed_portfolio(db, cash=20000.0)
    await _seed_positions(db)

    result = await _get_risk_assessment(db, "default", current_prices, closes)
    # With 100 days of data and positions in XLK/XLE/NVDA, vol should be computed
    assert result["annualized_vol_pct"] is not None
    assert result["annualized_vol_pct"] > 0


# ── Registration and aliases ─────────────────────────────────────────────────


async def test_registration_all_tools(db: Database, closes: pd.DataFrame, current_prices: dict):
    """register_portfolio_tools registers all Phase 2 + legacy tools."""
    await _seed_portfolio(db)
    registry = ToolRegistry()
    register_portfolio_tools(registry, db, "default", current_prices, closes=closes)

    names = registry.tool_names
    # Phase 2 tools
    assert "get_portfolio_state" in names
    assert "get_position_detail" in names
    assert "get_portfolio_performance" in names
    assert "get_historical_trades" in names
    assert "get_correlation_check" in names
    assert "get_risk_assessment" in names
    # Phase 32 aliases
    assert "get_current_positions" in names
    assert "get_position_pnl" in names
    assert "get_portfolio_summary" in names


async def test_registration_without_closes(db: Database, current_prices: dict):
    """register_portfolio_tools without closes skips correlation + risk tools."""
    await _seed_portfolio(db)
    registry = ToolRegistry()
    register_portfolio_tools(registry, db, "default", current_prices, closes=None)

    names = registry.tool_names
    assert "get_portfolio_state" in names
    assert "get_position_detail" in names
    assert "get_correlation_check" not in names
    assert "get_risk_assessment" not in names


async def test_alias_get_portfolio_summary(db: Database, current_prices: dict):
    """Legacy get_portfolio_summary alias returns Phase 32 format (no positions list)."""
    from src.agent.tools.portfolio import _get_portfolio_summary

    await _seed_portfolio(db, cash=20000.0)
    await _seed_positions(db)

    result = await _get_portfolio_summary(db, "default", current_prices)
    assert "cash" in result
    assert "sector_exposure" in result
    assert "positions" not in result  # Legacy format has no positions list


async def test_alias_get_position_pnl_delegates(db: Database, current_prices: dict):
    """Legacy get_position_pnl alias delegates to get_position_detail."""
    from src.agent.tools.portfolio import _get_position_pnl

    await _seed_portfolio(db, cash=20000.0)
    await _seed_positions(db)
    await _seed_trades(db)

    result = await _get_position_pnl(db, "default", current_prices, "XLK")
    # Should include recent_trades from get_position_detail
    assert "recent_trades" in result
    assert result["ticker"] == "XLK"
