"""Risk manager enforcing limits from config/risk_rules.py.

Two enforcement points:
1. Circuit breakers: halt trading if daily/weekly loss exceeds threshold.
2. Pre-trade filter: block positions that violate concentration limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
import structlog

from config.risk_rules import RISK_RULES, RiskRules
from config.universe import SECTOR_MAP
from src.storage.database import Database
from src.storage.models import TradeSchema

log = structlog.get_logger()


@dataclass
class RiskVerdict:
    """Result of pre-trade risk check."""

    allowed: list[TradeSchema] = field(default_factory=list)
    blocked: list[tuple[TradeSchema, str]] = field(default_factory=list)


class RiskManager:
    """Enforces position-level and portfolio-level risk limits."""

    def __init__(self, rules: RiskRules | None = None) -> None:
        self._rules = rules or RISK_RULES

    async def check_circuit_breakers(
        self, portfolio_name: str, db: Database, today: date,
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
                    reason = (
                        f"Daily loss {daily_pct:.1%} exceeds "
                        f"-{self._rules.daily_loss_limit_pct:.0%} limit"
                    )
                    log.warning("circuit_breaker_daily", portfolio=portfolio_name, reason=reason)
                    return True, reason

        # Weekly loss check — compare to snapshot from ~5 trading days ago
        week_ago = today - timedelta(days=7)
        week_snapshots = [s for s in snapshots if s.date >= week_ago]
        if len(week_snapshots) >= 2:
            week_start = week_snapshots[0]
            week_end = week_snapshots[-1]
            if week_start.total_value > 0:
                weekly_pct = (
                    (week_end.total_value - week_start.total_value) / week_start.total_value
                )
                if weekly_pct <= -self._rules.weekly_loss_limit_pct:
                    reason = (
                        f"Weekly loss {weekly_pct:.1%} exceeds "
                        f"-{self._rules.weekly_loss_limit_pct:.0%} limit"
                    )
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
    ) -> RiskVerdict:
        """Filter trades that violate risk limits.

        SELLs always pass (reducing risk). BUYs checked against:
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

        Returns:
            RiskVerdict with allowed and blocked lists.
        """
        verdict = RiskVerdict()

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

            # Rule 1: Single position concentration
            if total_denominator > 0:
                position_pct = new_position_value / total_denominator
                if position_pct > self._rules.max_single_position_pct:
                    reason = (
                        f"{trade.ticker} would be {position_pct:.0%} of portfolio "
                        f"(limit {self._rules.max_single_position_pct:.0%})"
                    )
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
                sector_limit = self._rules.sector_concentration_overrides.get(
                    sector, self._rules.max_sector_concentration,
                )
                if sector_pct > sector_limit:
                    reason = (
                        f"{sector} sector would be {sector_pct:.0%} "
                        f"(limit {sector_limit:.0%})"
                    )
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
                tech_over = (
                    total_denominator > 0
                    and tech_value / total_denominator > self._rules.max_tech_weight
                )
                if tech_over:
                    reason = (
                        f"Tech weight would be {tech_value / total_denominator:.0%} "
                        f"(limit {self._rules.max_tech_weight:.0%})"
                    )
                    log.warning("risk_blocked_tech", trade=trade.ticker, reason=reason)
                    verdict.blocked.append((trade, reason))
                    continue

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
