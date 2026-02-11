"""Tests for src/utils/allocations.py — allocation resolution logic."""

import pytest

from config.strategies import PORTFOLIO_A, PORTFOLIO_B
from src.storage.models import TenantRow
from src.utils.allocations import (
    DEFAULT_ALLOCATIONS,
    TenantAllocations,
    resolve_allocations,
    resolve_from_tenant,
)


class TestTenantAllocations:
    def test_for_portfolio_a(self) -> None:
        alloc = TenantAllocations(
            initial_equity=100_000,
            portfolio_a_pct=33.33,
            portfolio_b_pct=66.67,
            portfolio_a_cash=33_330,
            portfolio_b_cash=66_670,
        )
        assert alloc.for_portfolio("A") == 33_330

    def test_for_portfolio_b(self) -> None:
        alloc = TenantAllocations(
            initial_equity=100_000,
            portfolio_a_pct=33.33,
            portfolio_b_pct=66.67,
            portfolio_a_cash=33_330,
            portfolio_b_cash=66_670,
        )
        assert alloc.for_portfolio("B") == 66_670

    def test_for_portfolio_invalid(self) -> None:
        alloc = TenantAllocations(
            initial_equity=100_000,
            portfolio_a_pct=33.33,
            portfolio_b_pct=66.67,
            portfolio_a_cash=33_330,
            portfolio_b_cash=66_670,
        )
        with pytest.raises(ValueError, match="Unknown portfolio"):
            alloc.for_portfolio("C")

    def test_frozen_dataclass(self) -> None:
        alloc = TenantAllocations(
            initial_equity=100_000,
            portfolio_a_pct=33.33,
            portfolio_b_pct=66.67,
            portfolio_a_cash=33_330,
            portfolio_b_cash=66_670,
        )
        with pytest.raises(AttributeError):
            alloc.initial_equity = 200_000  # type: ignore[misc]


class TestResolveAllocations:
    def test_equity_and_pct_takes_priority(self) -> None:
        """When initial_equity is set, cash is computed from pct."""
        alloc = resolve_allocations(
            initial_equity=100_000,
            portfolio_a_pct=40.0,
            portfolio_b_pct=60.0,
            portfolio_a_cash=1.0,  # should be ignored
            portfolio_b_cash=2.0,  # should be ignored
        )
        assert alloc.initial_equity == 100_000
        assert alloc.portfolio_a_cash == 40_000.0
        assert alloc.portfolio_b_cash == 60_000.0

    def test_fallback_to_explicit_cash(self) -> None:
        """When initial_equity is None, uses explicit cash values."""
        alloc = resolve_allocations(
            initial_equity=None,
            portfolio_a_cash=25_000.0,
            portfolio_b_cash=75_000.0,
        )
        assert alloc.portfolio_a_cash == 25_000.0
        assert alloc.portfolio_b_cash == 75_000.0
        assert alloc.initial_equity == 100_000.0  # sum of cash

    def test_fallback_to_global_defaults(self) -> None:
        """When nothing is provided, uses config/strategies.py defaults."""
        alloc = resolve_allocations()
        assert alloc.portfolio_a_cash == PORTFOLIO_A.allocation_usd
        assert alloc.portfolio_b_cash == PORTFOLIO_B.allocation_usd

    def test_zero_equity_falls_back(self) -> None:
        """Zero equity should not be treated as a valid equity."""
        alloc = resolve_allocations(initial_equity=0.0)
        assert alloc.portfolio_a_cash == PORTFOLIO_A.allocation_usd

    def test_negative_equity_falls_back(self) -> None:
        alloc = resolve_allocations(initial_equity=-1000.0)
        assert alloc.portfolio_a_cash == PORTFOLIO_A.allocation_usd

    def test_custom_pct_split(self) -> None:
        alloc = resolve_allocations(
            initial_equity=200_000,
            portfolio_a_pct=50.0,
            portfolio_b_pct=50.0,
        )
        assert alloc.portfolio_a_cash == 100_000.0
        assert alloc.portfolio_b_cash == 100_000.0


class TestResolveFromTenant:
    def test_tenant_with_equity(self) -> None:
        tenant = TenantRow(
            id="t1",
            name="Test",
            initial_equity=100_000.0,
            portfolio_a_pct=30.0,
            portfolio_b_pct=70.0,
            portfolio_a_cash=33_000.0,
            portfolio_b_cash=66_000.0,
        )
        alloc = resolve_from_tenant(tenant)
        assert alloc.initial_equity == 100_000.0
        assert alloc.portfolio_a_cash == 30_000.0
        assert alloc.portfolio_b_cash == 70_000.0

    def test_tenant_without_equity(self) -> None:
        """Tenant with no initial_equity uses explicit cash."""
        tenant = TenantRow(
            id="t2",
            name="Test2",
            initial_equity=None,
            portfolio_a_pct=33.33,
            portfolio_b_pct=66.67,
            portfolio_a_cash=40_000.0,
            portfolio_b_cash=60_000.0,
        )
        alloc = resolve_from_tenant(tenant)
        assert alloc.portfolio_a_cash == 40_000.0
        assert alloc.portfolio_b_cash == 60_000.0

    def test_tenant_defaults(self) -> None:
        """Tenant with all defaults resolves to standard 33/66 split."""
        tenant = TenantRow(id="t3", name="Default")
        alloc = resolve_from_tenant(tenant)
        assert alloc.portfolio_a_cash == 33_000.0
        assert alloc.portfolio_b_cash == 66_000.0
        assert alloc.portfolio_a_pct == 33.33
        assert alloc.portfolio_b_pct == 66.67


class TestDefaultAllocations:
    def test_module_level_constant(self) -> None:
        assert DEFAULT_ALLOCATIONS.portfolio_a_cash == PORTFOLIO_A.allocation_usd
        assert DEFAULT_ALLOCATIONS.portfolio_b_cash == PORTFOLIO_B.allocation_usd

    def test_for_portfolio_works(self) -> None:
        assert DEFAULT_ALLOCATIONS.for_portfolio("A") == PORTFOLIO_A.allocation_usd
        assert DEFAULT_ALLOCATIONS.for_portfolio("B") == PORTFOLIO_B.allocation_usd
