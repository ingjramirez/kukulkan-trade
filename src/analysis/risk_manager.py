"""Risk manager enforcing limits from config/risk_rules.py.

Two enforcement points:
1. Circuit breakers: halt trading if daily/weekly loss exceeds threshold.
2. Pre-trade filter: block positions that violate concentration limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import structlog

from config.risk_rules import RISK_RULES, RiskRules
from config.universe import INVERSE_ETF_META, SECTOR_MAP, is_equity_hedge
from src.storage.database import Database
from src.storage.models import TradeSchema

if TYPE_CHECKING:
    from src.agent.posture import PostureLimits

log = structlog.get_logger()


@dataclass
class RiskVerdict:
    """Result of pre-trade risk check."""

    allowed: list[TradeSchema] = field(default_factory=list)
    blocked: list[tuple[TradeSchema, str]] = field(default_factory=list)
    requires_approval: list[TradeSchema] = field(default_factory=list)
    requires_trade_approval: list[tuple[TradeSchema, str]] = field(default_factory=list)


# Inverse ETF risk constants
MAX_SINGLE_INVERSE_PCT = 0.10  # 10% max single inverse position
MAX_TOTAL_INVERSE_PCT = 0.15  # 15% max total inverse exposure
MAX_INVERSE_POSITIONS = 2  # Max 2 inverse positions at once

# Regimes and postures where equity hedges are allowed
HEDGE_ALLOWED_REGIMES = {"CORRECTION", "CRISIS"}
HEDGE_ALLOWED_POSTURES = {"defensive", "crisis"}


class RiskManager:
    """Enforces position-level and portfolio-level risk limits."""

    def __init__(self, rules: RiskRules | None = None) -> None:
        self._rules = rules or RISK_RULES

    async def check_circuit_breakers(
        self,
        portfolio_name: str,
        db: Database,
        today: date,
        tenant_id: str = "default",
    ) -> tuple[bool, str]:
        """Check if trading should be halted due to drawdown.

        Args:
            portfolio_name: Portfolio to check (A or B).
            db: Database instance.
            today: Current date.
            tenant_id: Tenant UUID for data isolation.

        Returns:
            (should_halt, reason) — True means skip trading for this portfolio.
        """
        snapshots = await db.get_snapshots(portfolio_name, tenant_id=tenant_id)
        if not snapshots:
            return False, ""

        # Daily loss check — compare latest snapshot to the one before it
        if len(snapshots) >= 2:
            prev = snapshots[-2]
            latest = snapshots[-1]
            if prev.total_value > 0:
                daily_pct = (latest.total_value - prev.total_value) / prev.total_value
                if daily_pct <= -self._rules.daily_loss_limit_pct:
                    reason = f"Daily loss {daily_pct:.1%} exceeds -{self._rules.daily_loss_limit_pct:.0%} limit"
                    log.warning("circuit_breaker_daily", portfolio=portfolio_name, reason=reason)
                    return True, reason

        # Weekly loss check — compare to snapshot from ~5 trading days ago
        week_ago = today - timedelta(days=7)
        week_snapshots = [s for s in snapshots if s.date >= week_ago]
        if len(week_snapshots) >= 2:
            week_start = week_snapshots[0]
            week_end = week_snapshots[-1]
            if week_start.total_value > 0:
                weekly_pct = (week_end.total_value - week_start.total_value) / week_start.total_value
                if weekly_pct <= -self._rules.weekly_loss_limit_pct:
                    reason = f"Weekly loss {weekly_pct:.1%} exceeds -{self._rules.weekly_loss_limit_pct:.0%} limit"
                    log.warning("circuit_breaker_weekly", portfolio=portfolio_name, reason=reason)
                    return True, reason

        return False, ""

    def check_pre_trade(
        self,
        trades: list[TradeSchema],
        portfolio_name: str,
        current_positions: dict[str, float],
        latest_prices: dict[str, float],
        portfolio_value: float,
        cash: float,
        posture_limits: "PostureLimits | None" = None,
        regime: str | None = None,
        current_posture: str | None = None,
    ) -> RiskVerdict:
        """Filter trades that violate risk limits.

        SELLs always pass (reducing risk). BUYs checked against:
        - Inverse-specific rules (regime gate, posture gate, exposure limits)
        - Max single position concentration (35%)
        - Max sector concentration (50%)
        - Max tech weight (40%) for Portfolio B

        Trades are evaluated sequentially so later trades account for earlier ones.

        Args:
            trades: Proposed trades.
            portfolio_name: A or B.
            current_positions: Dict ticker -> shares.
            latest_prices: Dict ticker -> price.
            portfolio_value: Current total portfolio value.
            cash: Available cash.
            posture_limits: Optional posture limits (can only tighten).
            regime: Current market regime string (e.g. "BULL", "CORRECTION").
            current_posture: Current agent posture string (e.g. "defensive", "crisis").

        Returns:
            RiskVerdict with allowed, blocked, and requires_approval lists.
        """
        verdict = RiskVerdict()

        # Resolve effective limits: posture can only tighten, never loosen
        eff_single = self._rules.max_single_position_pct
        eff_sector = self._rules.max_sector_concentration
        if posture_limits is not None:
            eff_single = min(eff_single, posture_limits.max_single_position_pct)
            eff_sector = min(eff_sector, posture_limits.max_sector_concentration)

        # Build projected position values from current state
        projected_values: dict[str, float] = {}
        for ticker, shares in current_positions.items():
            price = latest_prices.get(ticker, 0)
            projected_values[ticker] = shares * price

        for trade in trades:
            # SELLs always pass
            if trade.side.value == "SELL":
                # Update projected state
                sell_value = trade.shares * latest_prices.get(trade.ticker, trade.price)
                projected_values[trade.ticker] = projected_values.get(trade.ticker, 0) - sell_value
                if projected_values[trade.ticker] <= 0:
                    projected_values.pop(trade.ticker, None)
                verdict.allowed.append(trade)
                continue

            # BUY — compute projected position value after this trade
            buy_value = trade.shares * latest_prices.get(trade.ticker, trade.price)
            new_position_value = projected_values.get(trade.ticker, 0) + buy_value
            # Use portfolio_value as denominator (accounts for cash + all positions)
            total_denominator = max(portfolio_value, sum(projected_values.values()) + buy_value)

            # ── Inverse ETF-specific rules (before standard rules) ────────
            is_inverse_buy = trade.ticker in INVERSE_ETF_META
            if is_inverse_buy:
                ticker_is_equity_hedge = is_equity_hedge(trade.ticker)

                # Rule 0a: Regime gate — equity hedges blocked unless CORRECTION/CRISIS
                if ticker_is_equity_hedge:
                    if regime is None or regime.upper() not in HEDGE_ALLOWED_REGIMES:
                        reason = (
                            f"{trade.ticker} equity hedge blocked: "
                            f"regime={regime or 'unknown'} (requires CORRECTION or CRISIS)"
                        )
                        log.warning("risk_blocked_inverse_regime", trade=trade.ticker, reason=reason)
                        verdict.blocked.append((trade, reason))
                        continue

                # Rule 0b: Posture gate — equity hedges blocked unless defensive/crisis
                if ticker_is_equity_hedge:
                    if current_posture is None or current_posture.lower() not in HEDGE_ALLOWED_POSTURES:
                        reason = (
                            f"{trade.ticker} equity hedge blocked: "
                            f"posture={current_posture or 'unknown'} (requires defensive or crisis)"
                        )
                        log.warning("risk_blocked_inverse_posture", trade=trade.ticker, reason=reason)
                        verdict.blocked.append((trade, reason))
                        continue

                # Rule 0c: Max 10% single inverse position
                if total_denominator > 0:
                    inv_position_pct = new_position_value / total_denominator
                    if inv_position_pct > MAX_SINGLE_INVERSE_PCT:
                        reason = (
                            f"{trade.ticker} inverse position would be {inv_position_pct:.0%} "
                            f"(limit {MAX_SINGLE_INVERSE_PCT:.0%})"
                        )
                        log.warning("risk_blocked_inverse_single", trade=trade.ticker, reason=reason)
                        verdict.blocked.append((trade, reason))
                        continue

                # Rule 0d: Max 15% total inverse exposure
                projected_inverse_total = (
                    sum(v for t, v in projected_values.items() if t in INVERSE_ETF_META) + buy_value
                )
                if total_denominator > 0:
                    inv_total_pct = projected_inverse_total / total_denominator
                    if inv_total_pct > MAX_TOTAL_INVERSE_PCT:
                        reason = (
                            f"Total inverse exposure would be {inv_total_pct:.0%} (limit {MAX_TOTAL_INVERSE_PCT:.0%})"
                        )
                        log.warning("risk_blocked_inverse_total", trade=trade.ticker, reason=reason)
                        verdict.blocked.append((trade, reason))
                        continue

                # Rule 0e: Max 2 inverse positions
                current_inverse_count = sum(1 for t in projected_values if t in INVERSE_ETF_META)
                # If this is a new inverse position (not adding to existing)
                if trade.ticker not in projected_values:
                    if current_inverse_count >= MAX_INVERSE_POSITIONS:
                        reason = (
                            f"Max {MAX_INVERSE_POSITIONS} inverse positions reached ({current_inverse_count} existing)"
                        )
                        log.warning("risk_blocked_inverse_count", trade=trade.ticker, reason=reason)
                        verdict.blocked.append((trade, reason))
                        continue

                # Inverse BUY passed all inverse-specific checks → flag for approval
                verdict.requires_approval.append(trade)

            # Rule 1: Single position concentration
            if total_denominator > 0:
                position_pct = new_position_value / total_denominator
                if position_pct > eff_single:
                    reason = f"{trade.ticker} would be {position_pct:.0%} of portfolio (limit {eff_single:.0%})"
                    log.warning("risk_blocked_position", trade=trade.ticker, reason=reason)
                    verdict.blocked.append((trade, reason))
                    continue

            # Rule 2: Sector concentration
            sector = SECTOR_MAP.get(trade.ticker, "Unknown")
            sector_value = new_position_value  # this trade's ticker
            for ticker, val in projected_values.items():
                if ticker != trade.ticker and SECTOR_MAP.get(ticker) == sector:
                    sector_value += val
            if total_denominator > 0:
                sector_pct = sector_value / total_denominator
                sector_limit = min(
                    self._rules.sector_concentration_overrides.get(
                        sector,
                        self._rules.max_sector_concentration,
                    ),
                    eff_sector,
                )
                if sector_pct > sector_limit:
                    reason = f"{sector} sector would be {sector_pct:.0%} (limit {sector_limit:.0%})"
                    log.warning("risk_blocked_sector", trade=trade.ticker, reason=reason)
                    verdict.blocked.append((trade, reason))
                    continue

            # Rule 3: Tech weight cap (Portfolio B only)
            if portfolio_name == "B":
                tech_tickers = set(self._rules.tech_etfs)
                tech_value = 0.0
                for ticker, val in projected_values.items():
                    if ticker in tech_tickers:
                        tech_value += val
                if trade.ticker in tech_tickers:
                    tech_value += buy_value
                else:
                    # If buying non-tech, tech weight isn't increased
                    pass
                tech_over = total_denominator > 0 and tech_value / total_denominator > self._rules.max_tech_weight
                if tech_over:
                    reason = (
                        f"Tech weight would be {tech_value / total_denominator:.0%} "
                        f"(limit {self._rules.max_tech_weight:.0%})"
                    )
                    log.warning("risk_blocked_tech", trade=trade.ticker, reason=reason)
                    verdict.blocked.append((trade, reason))
                    continue

            # Rule 4: Large trade approval (non-inverse BUYs > threshold % of portfolio)
            if not is_inverse_buy and total_denominator > 0:
                from config.settings import settings

                trade_pct = (buy_value / total_denominator) * 100
                if trade_pct > settings.trade_approval_threshold_pct:
                    reason = (
                        f"{trade.ticker} trade value ${buy_value:,.0f} is {trade_pct:.1f}% of portfolio "
                        f"(threshold: {settings.trade_approval_threshold_pct}%)"
                    )
                    log.info("trade_requires_approval", trade=trade.ticker, reason=reason)
                    verdict.requires_trade_approval.append((trade, reason))

            # Passed all checks — update projected state
            projected_values[trade.ticker] = new_position_value
            verdict.allowed.append(trade)

        if verdict.blocked:
            log.info(
                "risk_pre_trade_summary",
                portfolio=portfolio_name,
                allowed=len(verdict.allowed),
                blocked=len(verdict.blocked),
            )

        return verdict

    def compute_portfolio_correlation(
        self,
        closes: pd.DataFrame,
        held_tickers: list[str],
        lookback_days: int = 60,
        threshold: float = 0.7,
    ) -> dict:
        """Compute pairwise correlation for held positions.

        Args:
            closes: Full closes DataFrame.
            held_tickers: Tickers currently held in the portfolio.
            lookback_days: Number of trading days for correlation window.
            threshold: Correlation level to flag as high.

        Returns:
            Dict with avg_correlation, high_pairs list, matrix_size.
        """
        valid = [t for t in held_tickers if t in closes.columns]
        if len(valid) < 2:
            return {"avg_correlation": 0.0, "high_pairs": [], "matrix_size": len(valid)}

        returns = closes[valid].tail(lookback_days).pct_change().dropna()
        if len(returns) < 5:
            return {"avg_correlation": 0.0, "high_pairs": [], "matrix_size": len(valid)}

        corr_matrix = returns.corr()

        # Extract upper triangle (exclude diagonal)
        n = len(valid)
        correlations: list[float] = []
        high_pairs: list[tuple[str, str, float]] = []

        for i in range(n):
            for j in range(i + 1, n):
                val = float(corr_matrix.iloc[i, j])
                if np.isnan(val):
                    continue
                correlations.append(val)
                if val >= threshold:
                    high_pairs.append((valid[i], valid[j], round(val, 3)))

        avg_corr = float(np.mean(correlations)) if correlations else 0.0
        # Sort by correlation descending, keep top 5
        high_pairs.sort(key=lambda x: x[2], reverse=True)

        return {
            "avg_correlation": round(avg_corr, 3),
            "high_pairs": high_pairs[:5],
            "matrix_size": n,
        }

    async def check_inverse_hold_times(
        self,
        db: Database,
        portfolio_name: str,
        tenant_id: str = "default",
    ) -> list[dict]:
        """Check hold times for inverse ETF positions.

        Inverse ETFs decay over time, so prolonged holding is risky.
        Returns alerts for positions held too long.

        Args:
            db: Database instance.
            portfolio_name: Portfolio name (typically "B").
            tenant_id: Tenant UUID.

        Returns:
            List of alert dicts with ticker, days_held, alert_level, message.
        """
        from config.universe import INVERSE_ETF_META

        positions = await db.get_positions(portfolio_name, tenant_id=tenant_id)
        inverse_positions = [p for p in positions if p.ticker in INVERSE_ETF_META]

        if not inverse_positions:
            return []

        alerts: list[dict] = []
        for pos in inverse_positions:
            # Find most recent BUY for this ticker
            trades = await db.get_trades(portfolio_name, tenant_id=tenant_id)
            buy_trades = [t for t in trades if t.ticker == pos.ticker and t.side == "BUY"]
            if not buy_trades:
                continue

            # Most recent BUY
            latest_buy = buy_trades[0]  # trades are returned most recent first
            if not latest_buy.executed_at:
                continue

            from datetime import datetime, timezone

            executed = latest_buy.executed_at
            # SQLite returns naive datetimes — handle both
            now = datetime.now(timezone.utc)
            if executed.tzinfo is None:
                now = now.replace(tzinfo=None)
            days_held = (now - executed).days

            meta = INVERSE_ETF_META.get(pos.ticker, {})
            description = meta.get("description", pos.ticker)

            if days_held >= 5:
                alerts.append(
                    {
                        "ticker": pos.ticker,
                        "days_held": days_held,
                        "alert_level": "review",
                        "message": f"{description} held {days_held}d — review for time decay risk",
                    }
                )
            elif days_held >= 3:
                alerts.append(
                    {
                        "ticker": pos.ticker,
                        "days_held": days_held,
                        "alert_level": "warning",
                        "message": f"{description} held {days_held}d — approaching decay threshold",
                    }
                )

        return alerts
