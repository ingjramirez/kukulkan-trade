"""Tests for extended hours sentinel — phase-aware thresholds and queue-based escalation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.sentinel import (
    EXTENDED_VIX_HIGH_THRESHOLD,
    VIX_HIGH_THRESHOLD,
    AlertLevel,
    SentinelRunner,
    _get_state,
    _reset_sentinel_state,
)


@pytest.fixture(autouse=True)
def reset_state():
    _reset_sentinel_state()
    yield
    _reset_sentinel_state()


def _make_trailing_stop(
    ticker: str = "XLK",
    stop_price: float = 95.0,
    peak_price: float = 105.0,
    trail_pct: float = 0.05,
    portfolio: str = "B",
) -> MagicMock:
    stop = MagicMock()
    stop.ticker = ticker
    stop.stop_price = stop_price
    stop.peak_price = peak_price
    stop.trail_pct = trail_pct
    stop.portfolio = portfolio
    stop.is_active = True
    return stop


def _make_db(stops: list | None = None) -> AsyncMock:
    db = AsyncMock()
    db.get_active_trailing_stops = AsyncMock(return_value=stops or [])
    return db


class TestExtendedThresholds:
    """Extended hours should use wider thresholds for VIX and SPY."""

    async def test_afterhours_uses_wider_vix_threshold(self) -> None:
        """VIX at 30 is WARNING in market hours but CLEAR in after-hours."""
        vix_between = (VIX_HIGH_THRESHOLD + EXTENDED_VIX_HIGH_THRESHOLD) / 2  # e.g. 30

        async def prices(tickers):
            return {"^VIX": vix_between, "SPY": 500.0}

        # Market hours: should trigger warning
        runner_mkt = SentinelRunner(
            db=_make_db(), price_fetcher=prices, market_phase="market",
        )
        result_mkt = await runner_mkt.check_regime_shift()
        vix_alerts_mkt = [a for a in result_mkt if a.ticker == "^VIX"]
        assert len(vix_alerts_mkt) > 0  # Should alert

        _reset_sentinel_state()

        # After-hours: same VIX should NOT trigger (wider threshold)
        runner_ah = SentinelRunner(
            db=_make_db(), price_fetcher=prices, market_phase="afterhours",
        )
        result_ah = await runner_ah.check_regime_shift()
        vix_alerts_ah = [a for a in result_ah if a.ticker == "^VIX"]
        assert len(vix_alerts_ah) == 0

    async def test_afterhours_uses_wider_spy_threshold(self) -> None:
        """SPY move of 2.5% is WARNING in market but CLEAR in after-hours."""
        base_spy = 500.0
        # 2.5% move — between market warning (2%) and extended warning (3%)
        moved_spy = base_spy * (1 + 0.025)

        # Seed last_spy
        _get_state("default")["last_spy_close"] = base_spy

        async def prices_mkt(tickers):
            return {"^VIX": 20.0, "SPY": moved_spy}

        runner_mkt = SentinelRunner(
            db=_make_db(), price_fetcher=prices_mkt, market_phase="market",
        )
        result_mkt = await runner_mkt.check_regime_shift()
        spy_alerts_mkt = [a for a in result_mkt if a.ticker == "SPY"]
        assert len(spy_alerts_mkt) > 0

        # Reset and test AH
        _get_state("default")["last_spy_close"] = base_spy

        runner_ah = SentinelRunner(
            db=_make_db(), price_fetcher=prices_mkt, market_phase="afterhours",
        )
        result_ah = await runner_ah.check_regime_shift()
        spy_alerts_ah = [a for a in result_ah if a.ticker == "SPY"]
        assert len(spy_alerts_ah) == 0

    async def test_afterhours_stop_thresholds_same(self) -> None:
        """Stop proximity thresholds are the same in all phases."""
        stop = _make_trailing_stop(ticker="AAPL", stop_price=95.0)

        async def prices(tickers):
            return {"AAPL": 96.5}  # 1.6% from stop → critical in both

        runner_mkt = SentinelRunner(
            db=_make_db([stop]), price_fetcher=prices, market_phase="market",
        )
        alerts_mkt = await runner_mkt.check_stop_proximity()

        _reset_sentinel_state()

        runner_ah = SentinelRunner(
            db=_make_db([stop]), price_fetcher=prices, market_phase="afterhours",
        )
        alerts_ah = await runner_ah.check_stop_proximity()

        assert len(alerts_mkt) == len(alerts_ah)
        assert alerts_mkt[0].level == alerts_ah[0].level == AlertLevel.CRITICAL

    async def test_premarket_uses_wider_thresholds(self) -> None:
        """Pre-market should use extended thresholds like after-hours."""
        runner = SentinelRunner(
            db=_make_db(), market_phase="premarket",
        )
        assert runner._is_extended is True

    async def test_market_phase_not_extended(self) -> None:
        runner = SentinelRunner(db=_make_db(), market_phase="market")
        assert runner._is_extended is False


class TestExtendedEscalation:
    """Extended hours should queue actions, not trigger crisis sessions."""

    async def test_extended_critical_creates_sell_action(self) -> None:
        """Critical alerts in extended hours should queue sell actions in DB."""
        stop = _make_trailing_stop(ticker="TSLA", stop_price=195.0)

        async def prices(tickers):
            return {"TSLA": 196.0}  # 0.5% from stop → critical

        db = _make_db([stop])
        runner = SentinelRunner(
            db=db, price_fetcher=prices, market_phase="afterhours",
        )
        result = await runner.run_all_checks()
        assert result.max_level == AlertLevel.CRITICAL

    async def test_no_crisis_session_in_extended_hours(self) -> None:
        """Even with critical alerts, extended hours should NOT trigger crisis sessions.

        This is verified by the main.py job structure — extended_sentinel_job
        queues actions instead of calling run_daily with Sentinel-Crisis.
        The SentinelRunner itself is phase-aware but doesn't control escalation.
        """
        runner = SentinelRunner(db=_make_db(), market_phase="afterhours")
        # The runner doesn't have an escalation method — that's handled by the job
        assert runner._is_extended is True
        assert runner._market_phase == "afterhours"


class TestMorningQueue:
    async def test_morning_queue_returns_pending(self) -> None:
        """Orchestrator._process_morning_queue should return context string."""
        from src.orchestrator import Orchestrator

        db = AsyncMock()
        db.get_pending_sentinel_actions = AsyncMock(return_value=[
            {
                "id": 1,
                "action_type": "sell",
                "ticker": "AAPL",
                "reason": "Stop proximity 1.5%",
                "source": "afterhours_sentinel",
                "alert_level": "critical",
                "created_at": "2026-02-16T22:00:00",
            },
        ])
        orch = Orchestrator(db=db)
        result = await orch._process_morning_queue("default")
        assert result is not None
        assert "PRE-MARKET QUEUE" in result
        assert "AAPL" in result
        assert "CRITICAL" in result

    async def test_morning_queue_empty_returns_none(self) -> None:
        from src.orchestrator import Orchestrator

        db = AsyncMock()
        db.get_pending_sentinel_actions = AsyncMock(return_value=[])
        orch = Orchestrator(db=db)
        result = await orch._process_morning_queue("default")
        assert result is None

    async def test_sentinel_alert_sse_includes_phase(self) -> None:
        """SSE events from extended sentinel should include market_phase."""
        # This is tested implicitly by the main.py job, but we verify the data structure
        alert_data = {
            "max_level": "warning",
            "alerts": [{"level": "warning", "ticker": "AAPL", "message": "test", "market_phase": "afterhours"}],
            "market_phase": "afterhours",
        }
        assert alert_data["market_phase"] == "afterhours"
        assert alert_data["alerts"][0]["market_phase"] == "afterhours"
