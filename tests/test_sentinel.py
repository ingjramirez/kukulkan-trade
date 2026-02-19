"""Tests for the Intraday Haiku Sentinel.

Validates stop proximity alerts, regime shift detection,
fill verification, and result aggregation.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.sentinel import (
    AlertLevel,
    SentinelAlert,
    SentinelResult,
    SentinelRunner,
    _get_state,
    _reset_sentinel_state,
    _sentinel_states,
    can_escalate,
    record_alert_sent,
    record_escalation,
    record_session_time,
    should_send_alert,
)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset module-level sentinel state between tests."""
    _reset_sentinel_state()
    yield
    _reset_sentinel_state()


# ── Helpers ──────────────────────────────────────────────────────────────────


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


def _make_executor(open_orders: list | None = None) -> AsyncMock:
    executor = AsyncMock()
    executor.get_open_orders = AsyncMock(return_value=open_orders or [])
    return executor


async def _mock_prices(tickers: list[str]) -> dict[str, float]:
    """Default mock price fetcher — returns nothing."""
    return {}


# ── Check 1: Stop Proximity ─────────────────────────────────────────────────


class TestStopProximity:
    async def test_no_stops_returns_empty(self) -> None:
        runner = SentinelRunner(db=_make_db(), price_fetcher=_mock_prices)
        alerts = await runner.check_stop_proximity()
        assert alerts == []

    async def test_clear_when_far_from_stop(self) -> None:
        stop = _make_trailing_stop(ticker="XLK", stop_price=90.0)
        db = _make_db(stops=[stop])

        async def prices(tickers):
            return {"XLK": 100.0}  # 10% above stop

        runner = SentinelRunner(db=db, price_fetcher=prices)
        alerts = await runner.check_stop_proximity()
        assert len(alerts) == 0

    async def test_warning_when_near_stop(self) -> None:
        stop = _make_trailing_stop(ticker="XLK", stop_price=97.5)
        db = _make_db(stops=[stop])

        async def prices(tickers):
            return {"XLK": 100.0}  # 2.5% above stop → warning

        runner = SentinelRunner(db=db, price_fetcher=prices)
        alerts = await runner.check_stop_proximity()
        assert len(alerts) == 1
        assert alerts[0].level == AlertLevel.WARNING
        assert alerts[0].check_type == "stop_proximity"
        assert alerts[0].ticker == "XLK"

    async def test_critical_when_very_near_stop(self) -> None:
        stop = _make_trailing_stop(ticker="XLK", stop_price=99.0)
        db = _make_db(stops=[stop])

        async def prices(tickers):
            return {"XLK": 100.0}  # 1% above stop → critical

        runner = SentinelRunner(db=db, price_fetcher=prices)
        alerts = await runner.check_stop_proximity()
        assert len(alerts) == 1
        assert alerts[0].level == AlertLevel.CRITICAL

    async def test_critical_when_below_stop(self) -> None:
        stop = _make_trailing_stop(ticker="XLK", stop_price=102.0)
        db = _make_db(stops=[stop])

        async def prices(tickers):
            return {"XLK": 100.0}  # Below stop → critical

        runner = SentinelRunner(db=db, price_fetcher=prices)
        alerts = await runner.check_stop_proximity()
        assert len(alerts) == 1
        assert alerts[0].level == AlertLevel.CRITICAL
        assert alerts[0].details["distance_pct"] < 0

    async def test_skips_missing_prices(self) -> None:
        stop = _make_trailing_stop(ticker="XLK", stop_price=99.0)
        db = _make_db(stops=[stop])

        runner = SentinelRunner(db=db, price_fetcher=_mock_prices)
        alerts = await runner.check_stop_proximity()
        assert len(alerts) == 0

    async def test_multiple_stops_mixed_levels(self) -> None:
        stops = [
            _make_trailing_stop(ticker="XLK", stop_price=90.0),  # 10% — clear
            _make_trailing_stop(ticker="AAPL", stop_price=97.5),  # 2.5% — warning
            _make_trailing_stop(ticker="MSFT", stop_price=99.0),  # 1% — critical
        ]
        db = _make_db(stops=stops)

        async def prices(tickers):
            return {"XLK": 100.0, "AAPL": 100.0, "MSFT": 100.0}

        runner = SentinelRunner(db=db, price_fetcher=prices)
        alerts = await runner.check_stop_proximity()
        assert len(alerts) == 2  # XLK is clear (no alert)
        levels = {a.ticker: a.level for a in alerts}
        assert levels["AAPL"] == AlertLevel.WARNING
        assert levels["MSFT"] == AlertLevel.CRITICAL

    async def test_details_include_price_info(self) -> None:
        stop = _make_trailing_stop(ticker="XLK", stop_price=98.0, trail_pct=0.05, portfolio="B")
        db = _make_db(stops=[stop])

        async def prices(tickers):
            return {"XLK": 100.0}

        runner = SentinelRunner(db=db, price_fetcher=prices)
        alerts = await runner.check_stop_proximity()
        assert len(alerts) == 1
        d = alerts[0].details
        assert d["price"] == 100.0
        assert d["stop_price"] == 98.0
        assert d["portfolio"] == "B"
        assert d["trail_pct"] == 0.05
        assert d["distance_pct"] == 2.0

    async def test_skips_zero_price(self) -> None:
        stop = _make_trailing_stop(ticker="XLK", stop_price=98.0)
        db = _make_db(stops=[stop])

        async def prices(tickers):
            return {"XLK": 0.0}

        runner = SentinelRunner(db=db, price_fetcher=prices)
        alerts = await runner.check_stop_proximity()
        assert len(alerts) == 0


# ── Check 2: Regime Shift ───────────────────────────────────────────────────


class TestRegimeShift:
    async def test_vix_critical_extreme(self) -> None:
        async def prices(tickers):
            return {"^VIX": 40.0, "SPY": 400.0}

        runner = SentinelRunner(db=_make_db(), price_fetcher=prices)
        alerts = await runner.check_regime_shift()

        vix_alerts = [a for a in alerts if a.ticker == "^VIX"]
        assert len(vix_alerts) == 1
        assert vix_alerts[0].level == AlertLevel.CRITICAL
        assert "extreme fear" in vix_alerts[0].message

    async def test_vix_crossing_up_warning(self) -> None:
        _get_state("default")["last_vix"] = 25.0  # Was below threshold

        async def prices(tickers):
            return {"^VIX": 30.0, "SPY": 400.0}

        runner = SentinelRunner(db=_make_db(), price_fetcher=prices)
        alerts = await runner.check_regime_shift()

        vix_alerts = [a for a in alerts if a.ticker == "^VIX"]
        assert len(vix_alerts) == 1
        assert vix_alerts[0].level == AlertLevel.WARNING
        assert "crossed above" in vix_alerts[0].message

    async def test_vix_elevated_first_check(self) -> None:
        async def prices(tickers):
            return {"^VIX": 30.0, "SPY": 400.0}

        runner = SentinelRunner(db=_make_db(), price_fetcher=prices)
        alerts = await runner.check_regime_shift()

        vix_alerts = [a for a in alerts if a.ticker == "^VIX"]
        assert len(vix_alerts) == 1
        assert vix_alerts[0].level == AlertLevel.WARNING
        assert "elevated" in vix_alerts[0].message

    async def test_vix_staying_elevated_no_alert(self) -> None:
        _get_state("default")["last_vix"] = 30.0  # Already elevated

        async def prices(tickers):
            return {"^VIX": 31.0, "SPY": 400.0}

        runner = SentinelRunner(db=_make_db(), price_fetcher=prices)
        alerts = await runner.check_regime_shift()

        vix_alerts = [a for a in alerts if a.ticker == "^VIX"]
        assert len(vix_alerts) == 0

    async def test_vix_calming_signal(self) -> None:
        _get_state("default")["last_vix"] = 30.0

        async def prices(tickers):
            return {"^VIX": 18.0, "SPY": 400.0}

        runner = SentinelRunner(db=_make_db(), price_fetcher=prices)
        alerts = await runner.check_regime_shift()

        vix_alerts = [a for a in alerts if a.ticker == "^VIX"]
        assert len(vix_alerts) == 1
        assert vix_alerts[0].level == AlertLevel.CLEAR
        assert "calming" in vix_alerts[0].message

    async def test_spy_warning_big_move(self) -> None:
        _get_state("default")["last_spy_close"] = 400.0

        async def prices(tickers):
            return {"^VIX": 20.0, "SPY": 390.0}  # 2.5% move

        runner = SentinelRunner(db=_make_db(), price_fetcher=prices)
        alerts = await runner.check_regime_shift()

        spy_alerts = [a for a in alerts if a.ticker == "SPY"]
        assert len(spy_alerts) == 1
        assert spy_alerts[0].level == AlertLevel.WARNING

    async def test_spy_critical_huge_move(self) -> None:
        _get_state("default")["last_spy_close"] = 400.0

        async def prices(tickers):
            return {"^VIX": 20.0, "SPY": 385.0}  # 3.75% move

        runner = SentinelRunner(db=_make_db(), price_fetcher=prices)
        alerts = await runner.check_regime_shift()

        spy_alerts = [a for a in alerts if a.ticker == "SPY"]
        assert len(spy_alerts) == 1
        assert spy_alerts[0].level == AlertLevel.CRITICAL

    async def test_spy_no_alert_small_move(self) -> None:
        _get_state("default")["last_spy_close"] = 400.0

        async def prices(tickers):
            return {"^VIX": 20.0, "SPY": 396.0}  # 1% move

        runner = SentinelRunner(db=_make_db(), price_fetcher=prices)
        alerts = await runner.check_regime_shift()

        spy_alerts = [a for a in alerts if a.ticker == "SPY"]
        assert len(spy_alerts) == 0

    async def test_spy_first_check_no_alert(self) -> None:
        """First check with no previous SPY — no alert (need baseline)."""

        async def prices(tickers):
            return {"^VIX": 18.0, "SPY": 400.0}

        runner = SentinelRunner(db=_make_db(), price_fetcher=prices)
        alerts = await runner.check_regime_shift()

        spy_alerts = [a for a in alerts if a.ticker == "SPY"]
        assert len(spy_alerts) == 0

    async def test_updates_sentinel_state(self) -> None:
        async def prices(tickers):
            return {"^VIX": 22.0, "SPY": 410.0}

        runner = SentinelRunner(db=_make_db(), price_fetcher=prices)
        await runner.check_regime_shift()

        state = _get_state("default")
        assert state["last_vix"] == 22.0
        assert state["last_spy_close"] == 410.0

    async def test_no_prices_returns_empty(self) -> None:
        runner = SentinelRunner(db=_make_db(), price_fetcher=_mock_prices)
        alerts = await runner.check_regime_shift()
        assert alerts == []


# ── Check 3: Fill Verification ───────────────────────────────────────────────


class TestFillCheck:
    async def test_no_executor_returns_empty(self) -> None:
        runner = SentinelRunner(db=_make_db(), executor=None, price_fetcher=_mock_prices)
        alerts = await runner.check_fills()
        assert alerts == []

    async def test_no_get_open_orders_method_returns_empty(self) -> None:
        executor = MagicMock(spec=[])
        runner = SentinelRunner(db=_make_db(), executor=executor, price_fetcher=_mock_prices)
        alerts = await runner.check_fills()
        assert alerts == []

    async def test_no_open_orders_returns_empty(self) -> None:
        executor = _make_executor(open_orders=[])
        runner = SentinelRunner(db=_make_db(), executor=executor, price_fetcher=_mock_prices)
        alerts = await runner.check_fills()
        assert alerts == []

    async def test_partial_fill_warning(self) -> None:
        now = datetime.now(timezone.utc)
        orders = [
            {
                "order_id": "abc12345678",
                "ticker": "AAPL",
                "status": "partially_filled",
                "qty": 10,
                "filled_qty": 5,
                "created_at": now - timedelta(minutes=20),
            }
        ]
        executor = _make_executor(open_orders=orders)
        runner = SentinelRunner(db=_make_db(), executor=executor, price_fetcher=_mock_prices)
        alerts = await runner.check_fills()
        assert len(alerts) == 1
        assert alerts[0].level == AlertLevel.WARNING
        assert alerts[0].check_type == "fill_check"
        assert "partially filled" in alerts[0].message

    async def test_stale_new_order_warning(self) -> None:
        now = datetime.now(timezone.utc)
        orders = [
            {
                "order_id": "abc12345678",
                "ticker": "MSFT",
                "status": "new",
                "qty": 5,
                "filled_qty": 0,
                "created_at": now - timedelta(minutes=45),
            }
        ]
        executor = _make_executor(open_orders=orders)
        runner = SentinelRunner(db=_make_db(), executor=executor, price_fetcher=_mock_prices)
        alerts = await runner.check_fills()
        assert len(alerts) == 1
        assert alerts[0].level == AlertLevel.WARNING
        assert "45" in alerts[0].message

    async def test_very_stale_order_critical(self) -> None:
        now = datetime.now(timezone.utc)
        orders = [
            {
                "order_id": "abc12345678",
                "ticker": "GOOG",
                "status": "accepted",
                "qty": 3,
                "filled_qty": 0,
                "created_at": now - timedelta(minutes=90),
            }
        ]
        executor = _make_executor(open_orders=orders)
        runner = SentinelRunner(db=_make_db(), executor=executor, price_fetcher=_mock_prices)
        alerts = await runner.check_fills()
        assert len(alerts) == 1
        assert alerts[0].level == AlertLevel.CRITICAL

    async def test_recent_order_no_alert(self) -> None:
        now = datetime.now(timezone.utc)
        orders = [
            {
                "order_id": "abc12345678",
                "ticker": "AAPL",
                "status": "new",
                "qty": 10,
                "filled_qty": 0,
                "created_at": now - timedelta(minutes=5),
            }
        ]
        executor = _make_executor(open_orders=orders)
        runner = SentinelRunner(db=_make_db(), executor=executor, price_fetcher=_mock_prices)
        alerts = await runner.check_fills()
        assert len(alerts) == 0

    async def test_executor_error_returns_empty(self) -> None:
        executor = AsyncMock()
        executor.get_open_orders = AsyncMock(side_effect=Exception("connection failed"))
        runner = SentinelRunner(db=_make_db(), executor=executor, price_fetcher=_mock_prices)
        alerts = await runner.check_fills()
        assert alerts == []

    async def test_order_missing_created_at_skipped(self) -> None:
        orders = [
            {
                "order_id": "abc12345678",
                "ticker": "AAPL",
                "status": "new",
                "qty": 10,
                "filled_qty": 0,
                "created_at": None,
            }
        ]
        executor = _make_executor(open_orders=orders)
        runner = SentinelRunner(db=_make_db(), executor=executor, price_fetcher=_mock_prices)
        alerts = await runner.check_fills()
        assert len(alerts) == 0


# ── run_all_checks ───────────────────────────────────────────────────────────


class TestRunAllChecks:
    async def test_runs_all_three_checks(self) -> None:
        runner = SentinelRunner(db=_make_db(), price_fetcher=_mock_prices)
        result = await runner.run_all_checks()
        assert result.checks_run == 3

    async def test_aggregates_alerts(self) -> None:
        stop = _make_trailing_stop(ticker="XLK", stop_price=99.0)
        db = _make_db(stops=[stop])

        async def prices(tickers):
            return {"XLK": 100.0, "^VIX": 40.0, "SPY": 400.0}

        runner = SentinelRunner(db=db, price_fetcher=prices)
        result = await runner.run_all_checks()
        assert len(result.alerts) >= 2  # At least stop critical + VIX critical

    async def test_max_level_critical(self) -> None:
        stop = _make_trailing_stop(ticker="XLK", stop_price=99.5)
        db = _make_db(stops=[stop])

        async def prices(tickers):
            return {"XLK": 100.0, "^VIX": 20.0, "SPY": 400.0}

        runner = SentinelRunner(db=db, price_fetcher=prices)
        result = await runner.run_all_checks()
        assert result.max_level == AlertLevel.CRITICAL

    async def test_needs_escalation_true_on_critical(self) -> None:
        stop = _make_trailing_stop(ticker="XLK", stop_price=99.5)
        db = _make_db(stops=[stop])

        async def prices(tickers):
            return {"XLK": 100.0, "^VIX": 20.0, "SPY": 400.0}

        runner = SentinelRunner(db=db, price_fetcher=prices)
        result = await runner.run_all_checks()
        assert result.needs_escalation is True

    async def test_needs_escalation_false_on_clear(self) -> None:
        runner = SentinelRunner(db=_make_db(), price_fetcher=_mock_prices)
        result = await runner.run_all_checks()
        assert result.needs_escalation is False

    async def test_check_failure_doesnt_stop_others(self) -> None:
        db = AsyncMock()
        db.get_active_trailing_stops = AsyncMock(side_effect=Exception("DB error"))

        async def prices(tickers):
            return {"^VIX": 20.0, "SPY": 400.0}

        runner = SentinelRunner(db=db, price_fetcher=prices)
        result = await runner.run_all_checks()
        assert result.checks_run == 3  # All attempted
        error_alerts = [a for a in result.alerts if "failed" in a.message.lower()]
        assert len(error_alerts) == 1

    async def test_crypto_only_skips_regime_and_fills(self) -> None:
        """crypto_only=True should only run stop proximity, not regime/fills."""
        stop = _make_trailing_stop(ticker="BTC-USD", stop_price=92000.0)
        db = _make_db(stops=[stop])

        async def prices(tickers):
            return {"BTC-USD": 93000.0}  # ~1.1% from stop → critical

        runner = SentinelRunner(db=db, price_fetcher=prices)
        result = await runner.run_all_checks(crypto_only=True)
        assert result.checks_run == 1  # Only stop proximity
        assert len(result.alerts) == 1
        assert result.alerts[0].ticker == "BTC-USD"

    async def test_crypto_only_false_runs_all(self) -> None:
        """Default crypto_only=False should still run all 3 checks."""
        runner = SentinelRunner(db=_make_db(), price_fetcher=_mock_prices)
        result = await runner.run_all_checks(crypto_only=False)
        assert result.checks_run == 3

    async def test_has_timestamp(self) -> None:
        runner = SentinelRunner(db=_make_db(), price_fetcher=_mock_prices)
        result = await runner.run_all_checks()
        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)

    async def test_tenant_id_passed_to_db(self) -> None:
        db = _make_db()
        runner = SentinelRunner(db=db, tenant_id="tenant-42", price_fetcher=_mock_prices)
        await runner.run_all_checks()
        db.get_active_trailing_stops.assert_called_once_with("tenant-42")


# ── SentinelResult Dataclass ─────────────────────────────────────────────────


class TestSentinelResult:
    def test_max_level_empty(self) -> None:
        result = SentinelResult()
        assert result.max_level == AlertLevel.CLEAR

    def test_max_level_warning(self) -> None:
        result = SentinelResult(
            alerts=[SentinelAlert(level=AlertLevel.WARNING, check_type="test", ticker="X", message="test")]
        )
        assert result.max_level == AlertLevel.WARNING

    def test_max_level_critical_wins(self) -> None:
        result = SentinelResult(
            alerts=[
                SentinelAlert(level=AlertLevel.WARNING, check_type="test", ticker="X", message="test"),
                SentinelAlert(level=AlertLevel.CRITICAL, check_type="test", ticker="Y", message="test"),
            ]
        )
        assert result.max_level == AlertLevel.CRITICAL

    def test_needs_escalation_false_for_warning(self) -> None:
        result = SentinelResult(
            alerts=[SentinelAlert(level=AlertLevel.WARNING, check_type="test", ticker="X", message="test")]
        )
        assert result.needs_escalation is False

    def test_needs_escalation_true_for_critical(self) -> None:
        result = SentinelResult(
            alerts=[SentinelAlert(level=AlertLevel.CRITICAL, check_type="test", ticker="X", message="test")]
        )
        assert result.needs_escalation is True


# ── _reset_sentinel_state ────────────────────────────────────────────────────


class TestSentinelState:
    def test_reset_clears_all_tenants(self) -> None:
        state = _get_state("default")
        state["last_vix"] = 30.0
        state["last_spy_close"] = 400.0
        state["escalations_today"] = 5
        _get_state("tenant-2")["last_vix"] = 25.0

        _reset_sentinel_state()

        assert _sentinel_states == {}
        # New state after reset is fresh
        fresh = _get_state("default")
        assert fresh["last_vix"] is None
        assert fresh["last_spy_close"] is None
        assert fresh["escalations_today"] == 0
        assert fresh["last_escalation_date"] is None
        assert fresh["last_session_time"] is None


# ── Escalation Guards ────────────────────────────────────────────────────────


class TestCanEscalate:
    def test_allowed_when_no_prior_escalations(self) -> None:
        assert can_escalate(max_per_day=2) is True

    def test_blocked_when_daily_limit_reached(self) -> None:
        record_escalation()
        record_escalation()
        assert can_escalate(max_per_day=2) is False

    def test_allowed_under_daily_limit(self) -> None:
        record_escalation()
        assert can_escalate(max_per_day=2) is True

    def test_blocked_during_cooldown(self) -> None:
        # Session happened 10 minutes ago
        _get_state("default")["last_session_time"] = datetime.now(timezone.utc) - timedelta(minutes=10)
        assert can_escalate(max_per_day=2) is False

    def test_allowed_after_cooldown(self) -> None:
        # Session happened 45 minutes ago (> 30 min cooldown)
        _get_state("default")["last_session_time"] = datetime.now(timezone.utc) - timedelta(minutes=45)
        assert can_escalate(max_per_day=2) is True

    def test_daily_counter_resets_next_day(self) -> None:
        from datetime import date

        state = _get_state("default")
        state["escalations_today"] = 5
        state["last_escalation_date"] = date(2020, 1, 1)  # Old date
        assert can_escalate(max_per_day=2) is True
        # Counter should have been reset
        assert _get_state("default")["escalations_today"] == 0


class TestRecordEscalation:
    def test_increments_counter(self) -> None:
        assert _get_state("default")["escalations_today"] == 0
        record_escalation()
        assert _get_state("default")["escalations_today"] == 1
        record_escalation()
        assert _get_state("default")["escalations_today"] == 2

    def test_sets_date(self) -> None:
        from datetime import date

        record_escalation()
        assert _get_state("default")["last_escalation_date"] == date.today()


class TestRecordSessionTime:
    def test_sets_timestamp(self) -> None:
        assert _get_state("default")["last_session_time"] is None
        record_session_time()
        assert _get_state("default")["last_session_time"] is not None
        elapsed = (datetime.now(timezone.utc) - _get_state("default")["last_session_time"]).total_seconds()
        assert elapsed < 2  # Just set it


# ── Multi-Tenant State Isolation ─────────────────────────────────────────────


class TestMultiTenantIsolation:
    async def test_regime_state_isolated_per_tenant(self) -> None:
        """VIX/SPY tracking for tenant A should not affect tenant B."""

        async def prices(tickers):
            return {"^VIX": 22.0, "SPY": 410.0}

        runner_a = SentinelRunner(db=_make_db(), tenant_id="tenant-a", price_fetcher=prices)
        await runner_a.check_regime_shift()

        runner_b = SentinelRunner(db=_make_db(), tenant_id="tenant-b", price_fetcher=prices)
        await runner_b.check_regime_shift()

        assert _get_state("tenant-a")["last_vix"] == 22.0
        assert _get_state("tenant-b")["last_vix"] == 22.0
        # Modify one — should not affect the other
        _get_state("tenant-a")["last_vix"] = 40.0
        assert _get_state("tenant-b")["last_vix"] == 22.0

    def test_escalation_counter_isolated_per_tenant(self) -> None:
        """Hitting escalation limit on tenant A should not block tenant B."""
        record_escalation(tenant_id="tenant-a")
        record_escalation(tenant_id="tenant-a")

        assert can_escalate(max_per_day=2, tenant_id="tenant-a") is False
        assert can_escalate(max_per_day=2, tenant_id="tenant-b") is True

    def test_session_time_isolated_per_tenant(self) -> None:
        """Recording session on tenant A should not cooldown tenant B."""
        record_session_time(tenant_id="tenant-a")

        state_a = _get_state("tenant-a")
        state_b = _get_state("tenant-b")
        assert state_a["last_session_time"] is not None
        assert state_b["last_session_time"] is None

    def test_alert_throttle_isolated_per_tenant(self) -> None:
        """Alert throttle for tenant A should not suppress alerts for tenant B."""
        record_alert_sent("AAPL", tenant_id="tenant-a")

        assert should_send_alert("AAPL", tenant_id="tenant-a") is False
        assert should_send_alert("AAPL", tenant_id="tenant-b") is True


# ── Alert Throttling ─────────────────────────────────────────────────────────


class TestAlertThrottling:
    def test_first_alert_allowed(self) -> None:
        assert should_send_alert("AAPL") is True

    def test_repeat_alert_blocked(self) -> None:
        record_alert_sent("AAPL")
        assert should_send_alert("AAPL") is False

    def test_different_ticker_allowed(self) -> None:
        record_alert_sent("AAPL")
        assert should_send_alert("MSFT") is True

    def test_alert_allowed_after_cooldown(self) -> None:
        state = _get_state("default")
        state["last_alert_by_ticker"]["AAPL"] = datetime.now(timezone.utc) - timedelta(hours=2)
        assert should_send_alert("AAPL") is True
