"""Intraday Haiku Sentinel — lightweight checks between scheduled sessions.

Runs every 30 minutes during market hours. Three check types:
1. Stop proximity: Are any positions near their trailing stop?
2. Regime shift: Has VIX or SPY moved significantly?
3. Fill verification: Are any orders still pending from recent sessions?

Escalation levels:
- CLEAR: No issues
- WARNING: Telegram alert only
- CRITICAL: Trigger crisis Sonnet session
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine

import structlog

log = structlog.get_logger()

# ── Thresholds ───────────────────────────────────────────────────────────────

STOP_PROXIMITY_WARNING_PCT = 3.0  # <3% from stop → warning
STOP_PROXIMITY_CRITICAL_PCT = 2.0  # <2% from stop → critical

VIX_HIGH_THRESHOLD = 28.0  # VIX crossed above → warning
VIX_CRITICAL_THRESHOLD = 35.0  # VIX above → critical
VIX_LOW_THRESHOLD = 20.0  # VIX dropped below after being high → calming

SPY_MOVE_WARNING_PCT = 2.0  # SPY intraday >2% → warning
SPY_MOVE_CRITICAL_PCT = 3.0  # SPY intraday >3% → critical

FILL_STALE_WARNING_MINUTES = 30  # Open order >30min → warning
FILL_STALE_CRITICAL_MINUTES = 60  # Open order >60min → critical

# ── Types ────────────────────────────────────────────────────────────────────

PriceFetcher = Callable[[list[str]], Coroutine[Any, Any, dict[str, float]]]


class AlertLevel(str, Enum):
    CLEAR = "clear"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class SentinelAlert:
    """Single alert from a sentinel check."""

    level: AlertLevel
    check_type: str
    ticker: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SentinelResult:
    """Aggregated result from all sentinel checks."""

    alerts: list[SentinelAlert] = field(default_factory=list)
    checks_run: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def max_level(self) -> AlertLevel:
        if not self.alerts:
            return AlertLevel.CLEAR
        levels = [a.level for a in self.alerts]
        if AlertLevel.CRITICAL in levels:
            return AlertLevel.CRITICAL
        if AlertLevel.WARNING in levels:
            return AlertLevel.WARNING
        return AlertLevel.CLEAR

    @property
    def needs_escalation(self) -> bool:
        return self.max_level == AlertLevel.CRITICAL


# ── Per-tenant state for cross-check tracking ────────────────────────────────

_DEFAULT_STATE: dict[str, Any] = {
    "last_vix": None,
    "last_spy_close": None,
    "escalations_today": 0,
    "last_escalation_date": None,
    "last_session_time": None,
    "last_alert_by_ticker": {},  # ticker → datetime (for Telegram throttling)
}

# Keyed by tenant_id → state dict
_sentinel_states: dict[str, dict[str, Any]] = {}

ALERT_THROTTLE_SECONDS = 3600  # Don't re-send same ticker alert within 1 hour


def _get_state(tenant_id: str = "default") -> dict[str, Any]:
    """Get or create per-tenant sentinel state."""
    if tenant_id not in _sentinel_states:
        _sentinel_states[tenant_id] = {k: (v.copy() if isinstance(v, dict) else v) for k, v in _DEFAULT_STATE.items()}
    return _sentinel_states[tenant_id]


def _reset_sentinel_state() -> None:
    """Reset all tenant states (for testing)."""
    _sentinel_states.clear()


def can_escalate(max_per_day: int = 2, tenant_id: str = "default") -> bool:
    """Check if a crisis escalation is allowed for this tenant.

    Guards:
    1. Daily limit (default 2 per day, per tenant)
    2. Cooldown: no escalation within cooldown window of a scheduled session
    """
    from datetime import date

    from config.settings import settings

    cooldown_s = settings.sentinel_escalation_cooldown_s
    state = _get_state(tenant_id)
    today = date.today()
    if state["last_escalation_date"] != today:
        state["escalations_today"] = 0
        state["last_escalation_date"] = today

    if state["escalations_today"] >= max_per_day:
        log.info(
            "sentinel_escalation_blocked_daily_limit",
            tenant_id=tenant_id,
            used=state["escalations_today"],
        )
        return False

    last_session = state.get("last_session_time")
    if last_session is not None:
        elapsed = (datetime.now(timezone.utc) - last_session).total_seconds()
        if elapsed < cooldown_s:
            log.info("sentinel_escalation_blocked_cooldown", tenant_id=tenant_id, elapsed_s=int(elapsed))
            return False

    return True


def record_escalation(tenant_id: str = "default") -> None:
    """Record that an escalation was triggered (increments daily counter)."""
    from datetime import date

    state = _get_state(tenant_id)
    today = date.today()
    if state["last_escalation_date"] != today:
        state["escalations_today"] = 0
        state["last_escalation_date"] = today
    state["escalations_today"] += 1
    log.info("sentinel_escalation_recorded", tenant_id=tenant_id, count=state["escalations_today"])


def record_session_time(tenant_id: str = "default") -> None:
    """Record when a scheduled session completed (for cooldown tracking)."""
    _get_state(tenant_id)["last_session_time"] = datetime.now(timezone.utc)


def should_send_alert(ticker: str, tenant_id: str = "default") -> bool:
    """Check if a Telegram alert should be sent for this ticker (throttle dedup)."""
    state = _get_state(tenant_id)
    last_sent = state["last_alert_by_ticker"].get(ticker)
    if last_sent is not None:
        elapsed = (datetime.now(timezone.utc) - last_sent).total_seconds()
        if elapsed < ALERT_THROTTLE_SECONDS:
            return False
    return True


def record_alert_sent(ticker: str, tenant_id: str = "default") -> None:
    """Record that an alert was sent for this ticker."""
    _get_state(tenant_id)["last_alert_by_ticker"][ticker] = datetime.now(timezone.utc)


# ── Default price fetcher ────────────────────────────────────────────────────


async def _fetch_latest_prices(tickers: list[str]) -> dict[str, float]:
    """Fetch latest prices for multiple tickers via yfinance (sync → thread)."""
    import yfinance as yf

    def _fetch() -> dict[str, float]:
        prices: dict[str, float] = {}
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period="1d")
                if not hist.empty:
                    prices[ticker] = float(hist["Close"].iloc[-1])
            except Exception:
                log.warning("sentinel_price_fetch_failed", ticker=ticker)
        return prices

    return await asyncio.to_thread(_fetch)


# ── SentinelRunner ───────────────────────────────────────────────────────────


class SentinelRunner:
    """Lightweight intraday monitor that runs between scheduled sessions.

    Constructor accepts injectable dependencies for testability.
    """

    def __init__(
        self,
        db: Any,
        executor: Any | None = None,
        tenant_id: str = "default",
        price_fetcher: PriceFetcher | None = None,
    ) -> None:
        self._db = db
        self._executor = executor
        self._tenant_id = tenant_id
        self._price_fetcher = price_fetcher or _fetch_latest_prices

    async def run_all_checks(self) -> SentinelResult:
        """Run all sentinel checks and aggregate results."""
        result = SentinelResult()

        checks = [
            self.check_stop_proximity,
            self.check_regime_shift,
            self.check_fills,
        ]

        for check in checks:
            try:
                alerts = await check()
                result.alerts.extend(alerts)
                result.checks_run += 1
            except Exception as e:
                log.error("sentinel_check_failed", check=check.__name__, error=str(e))
                result.alerts.append(
                    SentinelAlert(
                        level=AlertLevel.WARNING,
                        check_type=check.__name__,
                        ticker="",
                        message=f"Check failed: {e}",
                    )
                )
                result.checks_run += 1

        log.info(
            "sentinel_run_complete",
            checks=result.checks_run,
            alerts=len(result.alerts),
            max_level=result.max_level.value,
            needs_escalation=result.needs_escalation,
        )
        return result

    # ── Check 1: Stop Proximity ──────────────────────────────────────────

    async def check_stop_proximity(self) -> list[SentinelAlert]:
        """Check how close current prices are to trailing stop levels."""
        alerts: list[SentinelAlert] = []

        stops = await self._db.get_active_trailing_stops(self._tenant_id)
        if not stops:
            return alerts

        tickers = [s.ticker for s in stops]
        prices = await self._price_fetcher(tickers)

        for stop in stops:
            price = prices.get(stop.ticker)
            if price is None or price <= 0:
                continue

            distance_pct = ((price - stop.stop_price) / price) * 100

            if distance_pct < STOP_PROXIMITY_CRITICAL_PCT:
                level = AlertLevel.CRITICAL
            elif distance_pct < STOP_PROXIMITY_WARNING_PCT:
                level = AlertLevel.WARNING
            else:
                continue  # Far enough — no alert

            alerts.append(
                SentinelAlert(
                    level=level,
                    check_type="stop_proximity",
                    ticker=stop.ticker,
                    message=f"{stop.ticker} is {distance_pct:.1f}% from trailing stop (${stop.stop_price:.2f})",
                    details={
                        "price": price,
                        "stop_price": stop.stop_price,
                        "distance_pct": round(distance_pct, 2),
                        "portfolio": stop.portfolio,
                        "trail_pct": stop.trail_pct,
                    },
                )
            )

        return alerts

    # ── Check 2: Regime Shift ────────────────────────────────────────────

    async def check_regime_shift(self) -> list[SentinelAlert]:
        """Check VIX and SPY for significant intraday moves."""
        alerts: list[SentinelAlert] = []
        state = _get_state(self._tenant_id)

        prices = await self._price_fetcher(["SPY", "^VIX"])

        vix = prices.get("^VIX")
        spy = prices.get("SPY")

        if vix is not None:
            alerts.extend(self._evaluate_vix(vix))
            state["last_vix"] = vix

        if spy is not None:
            alerts.extend(self._evaluate_spy(spy))
            state["last_spy_close"] = spy

        return alerts

    def _evaluate_vix(self, vix: float) -> list[SentinelAlert]:
        """Evaluate VIX level and crossing."""
        alerts: list[SentinelAlert] = []
        last_vix = _get_state(self._tenant_id)["last_vix"]

        if vix >= VIX_CRITICAL_THRESHOLD:
            alerts.append(
                SentinelAlert(
                    level=AlertLevel.CRITICAL,
                    check_type="regime_shift",
                    ticker="^VIX",
                    message=f"VIX at {vix:.1f} — extreme fear",
                    details={"vix": vix, "threshold": VIX_CRITICAL_THRESHOLD, "previous_vix": last_vix},
                )
            )
        elif vix >= VIX_HIGH_THRESHOLD:
            if last_vix is not None and last_vix < VIX_HIGH_THRESHOLD:
                # Crossed up
                alerts.append(
                    SentinelAlert(
                        level=AlertLevel.WARNING,
                        check_type="regime_shift",
                        ticker="^VIX",
                        message=f"VIX crossed above {VIX_HIGH_THRESHOLD} → now {vix:.1f}",
                        details={"vix": vix, "threshold": VIX_HIGH_THRESHOLD, "previous_vix": last_vix},
                    )
                )
            elif last_vix is None:
                # First check — already elevated
                alerts.append(
                    SentinelAlert(
                        level=AlertLevel.WARNING,
                        check_type="regime_shift",
                        ticker="^VIX",
                        message=f"VIX elevated at {vix:.1f} (above {VIX_HIGH_THRESHOLD})",
                        details={"vix": vix, "threshold": VIX_HIGH_THRESHOLD, "previous_vix": None},
                    )
                )
            # else: staying elevated, no new alert
        elif last_vix is not None and last_vix >= VIX_HIGH_THRESHOLD and vix < VIX_LOW_THRESHOLD:
            # Significant calming
            alerts.append(
                SentinelAlert(
                    level=AlertLevel.CLEAR,
                    check_type="regime_shift",
                    ticker="^VIX",
                    message=f"VIX dropped to {vix:.1f} (below {VIX_LOW_THRESHOLD}) — regime calming",
                    details={"vix": vix, "threshold": VIX_LOW_THRESHOLD, "previous_vix": last_vix},
                )
            )

        return alerts

    def _evaluate_spy(self, spy: float) -> list[SentinelAlert]:
        """Evaluate SPY intraday move."""
        alerts: list[SentinelAlert] = []
        last_spy = _get_state(self._tenant_id)["last_spy_close"]

        if last_spy is None or last_spy <= 0:
            return alerts

        move_pct = abs((spy - last_spy) / last_spy) * 100
        direction = "up" if spy > last_spy else "down"

        if move_pct >= SPY_MOVE_CRITICAL_PCT:
            alerts.append(
                SentinelAlert(
                    level=AlertLevel.CRITICAL,
                    check_type="regime_shift",
                    ticker="SPY",
                    message=f"SPY moved {move_pct:.1f}% {direction} since last check",
                    details={"spy": spy, "previous_spy": last_spy, "move_pct": round(move_pct, 2)},
                )
            )
        elif move_pct >= SPY_MOVE_WARNING_PCT:
            alerts.append(
                SentinelAlert(
                    level=AlertLevel.WARNING,
                    check_type="regime_shift",
                    ticker="SPY",
                    message=f"SPY moved {move_pct:.1f}% {direction} since last check",
                    details={"spy": spy, "previous_spy": last_spy, "move_pct": round(move_pct, 2)},
                )
            )

        return alerts

    # ── Check 3: Fill Verification ───────────────────────────────────────

    async def check_fills(self) -> list[SentinelAlert]:
        """Check for stale or problematic open orders."""
        alerts: list[SentinelAlert] = []

        if self._executor is None:
            return alerts

        if not hasattr(self._executor, "get_open_orders"):
            return alerts

        try:
            open_orders = await self._executor.get_open_orders()
        except Exception as e:
            log.warning("sentinel_fill_check_failed", error=str(e))
            return alerts

        now = datetime.now(timezone.utc)

        for order in open_orders:
            created_at = order.get("created_at")
            if created_at is None:
                continue

            age_minutes = (now - created_at).total_seconds() / 60
            status = order.get("status", "unknown")
            ticker = order.get("ticker", "???")
            order_id = order.get("order_id", "???")

            if status == "partially_filled":
                alerts.append(
                    SentinelAlert(
                        level=AlertLevel.WARNING,
                        check_type="fill_check",
                        ticker=ticker,
                        message=(f"{ticker} order {order_id[:8]} partially filled ({age_minutes:.0f}min old)"),
                        details={
                            "order_id": order_id,
                            "status": status,
                            "age_minutes": round(age_minutes, 1),
                            "filled_qty": order.get("filled_qty", 0),
                            "total_qty": order.get("qty", 0),
                        },
                    )
                )
            elif age_minutes > FILL_STALE_WARNING_MINUTES and status in ("new", "accepted", "pending_new"):
                level = AlertLevel.CRITICAL if age_minutes > FILL_STALE_CRITICAL_MINUTES else AlertLevel.WARNING
                alerts.append(
                    SentinelAlert(
                        level=level,
                        check_type="fill_check",
                        ticker=ticker,
                        message=f"{ticker} order {order_id[:8]} still {status} after {age_minutes:.0f}min",
                        details={
                            "order_id": order_id,
                            "status": status,
                            "age_minutes": round(age_minutes, 1),
                        },
                    )
                )

        return alerts
