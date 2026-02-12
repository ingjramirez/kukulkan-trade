"""Centralised portfolio allocation resolution.

Replaces all hardcoded 33K/66K references with a single source of truth.
Resolution priority: initial_equity * pct → explicit cash → global defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.strategies import PORTFOLIO_A, PORTFOLIO_B
from src.storage.models import TenantRow

# Minimum delta (USD) to treat as a deposit rather than rounding noise.
DEPOSIT_THRESHOLD = 50.0

# Minimum drift (USD) to trigger equity reconciliation against broker.
RECONCILE_THRESHOLD = 10.0


@dataclass(frozen=True)
class TenantAllocations:
    """Resolved dollar allocations for a tenant's portfolios."""

    initial_equity: float
    portfolio_a_pct: float
    portfolio_b_pct: float
    portfolio_a_cash: float  # = initial_equity * a_pct / 100
    portfolio_b_cash: float  # = initial_equity * b_pct / 100

    def for_portfolio(self, name: str) -> float:
        """Return the dollar allocation for the given portfolio name."""
        if name == "A":
            return self.portfolio_a_cash
        if name == "B":
            return self.portfolio_b_cash
        raise ValueError(f"Unknown portfolio: {name!r}")


_DEFAULT_A_PCT = 33.33
_DEFAULT_B_PCT = 66.67


def resolve_allocations(
    initial_equity: float | None = None,
    portfolio_a_pct: float | None = None,
    portfolio_b_pct: float | None = None,
    portfolio_a_cash: float | None = None,
    portfolio_b_cash: float | None = None,
) -> TenantAllocations:
    """Build a TenantAllocations from the best available data.

    Priority:
        1. initial_equity * pct  (percentage-based, most accurate)
        2. Explicit cash values   (legacy / manual override)
        3. Global defaults from config/strategies.py
    """
    a_pct = portfolio_a_pct if portfolio_a_pct is not None else _DEFAULT_A_PCT
    b_pct = portfolio_b_pct if portfolio_b_pct is not None else _DEFAULT_B_PCT

    if initial_equity is not None and initial_equity > 0:
        return TenantAllocations(
            initial_equity=initial_equity,
            portfolio_a_pct=a_pct,
            portfolio_b_pct=b_pct,
            portfolio_a_cash=initial_equity * a_pct / 100,
            portfolio_b_cash=initial_equity * b_pct / 100,
        )

    a_cash = portfolio_a_cash if portfolio_a_cash is not None else PORTFOLIO_A.allocation_usd
    b_cash = portfolio_b_cash if portfolio_b_cash is not None else PORTFOLIO_B.allocation_usd
    equity = a_cash + b_cash

    return TenantAllocations(
        initial_equity=equity,
        portfolio_a_pct=a_pct,
        portfolio_b_pct=b_pct,
        portfolio_a_cash=a_cash,
        portfolio_b_cash=b_cash,
    )


def resolve_from_tenant(tenant: TenantRow) -> TenantAllocations:
    """Convenience wrapper: resolve allocations from a TenantRow."""
    return resolve_allocations(
        initial_equity=tenant.initial_equity,
        portfolio_a_pct=tenant.portfolio_a_pct,
        portfolio_b_pct=tenant.portfolio_b_pct,
        portfolio_a_cash=tenant.portfolio_a_cash,
        portfolio_b_cash=tenant.portfolio_b_cash,
    )


# Module-level constant for backward compatibility (default tenant, no TenantRow).
DEFAULT_ALLOCATIONS = resolve_allocations()
