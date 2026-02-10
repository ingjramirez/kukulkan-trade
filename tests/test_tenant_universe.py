"""Tests for tenant-specific ticker universe resolution."""

import json
from unittest.mock import MagicMock

import pytest

from config.universe import PORTFOLIO_A_UNIVERSE, PORTFOLIO_B_UNIVERSE
from src.utils.tenant_universe import get_tenant_universe


def _mock_tenant(
    whitelist: list[str] | None = None,
    additions: list[str] | None = None,
    exclusions: list[str] | None = None,
) -> MagicMock:
    tenant = MagicMock()
    tenant.id = "test-tenant"
    tenant.ticker_whitelist = json.dumps(whitelist) if whitelist else None
    tenant.ticker_additions = json.dumps(additions) if additions else None
    tenant.ticker_exclusions = json.dumps(exclusions) if exclusions else None
    return tenant


class TestWhitelistMode:
    def test_whitelist_overrides_base(self):
        tenant = _mock_tenant(whitelist=["AAPL", "TSLA", "NVDA"])
        result = get_tenant_universe(tenant, "B")
        assert result == ["AAPL", "NVDA", "TSLA"]

    def test_whitelist_ignores_additions(self):
        tenant = _mock_tenant(
            whitelist=["AAPL"],
            additions=["COIN", "MSTR"],
        )
        result = get_tenant_universe(tenant, "B")
        assert result == ["AAPL"]

    def test_whitelist_deduplicates(self):
        tenant = _mock_tenant(whitelist=["AAPL", "AAPL", "TSLA"])
        result = get_tenant_universe(tenant, "B")
        assert result == ["AAPL", "TSLA"]


class TestAdditiveMode:
    def test_additions_expand_universe(self):
        tenant = _mock_tenant(additions=["COIN", "MSTR"])
        result = get_tenant_universe(tenant, "B")
        assert "COIN" in result
        assert "MSTR" in result
        # Original universe tickers should also be present
        assert "AAPL" in result

    def test_exclusions_shrink_universe(self):
        tenant = _mock_tenant(exclusions=["AAPL", "MSFT"])
        result = get_tenant_universe(tenant, "B")
        assert "AAPL" not in result
        assert "MSFT" not in result
        # Other tickers should remain
        assert "NVDA" in result

    def test_additions_and_exclusions_combined(self):
        tenant = _mock_tenant(
            additions=["COIN"],
            exclusions=["AAPL"],
        )
        result = get_tenant_universe(tenant, "B")
        assert "COIN" in result
        assert "AAPL" not in result


class TestDefaultUniverse:
    def test_no_customization_returns_base(self):
        tenant = _mock_tenant()
        result_a = get_tenant_universe(tenant, "A")
        result_b = get_tenant_universe(tenant, "B")
        assert result_a == sorted(set(PORTFOLIO_A_UNIVERSE))
        assert result_b == sorted(set(PORTFOLIO_B_UNIVERSE))

    def test_portfolio_a_uses_a_universe(self):
        tenant = _mock_tenant()
        result = get_tenant_universe(tenant, "A")
        # Portfolio A should not have individual stocks
        assert "COIN" not in result


class TestEdgeCases:
    def test_empty_whitelist_treated_as_none(self):
        tenant = _mock_tenant()
        tenant.ticker_whitelist = json.dumps([])
        result = get_tenant_universe(tenant, "B")
        # Empty list should be treated as no whitelist
        assert len(result) == len(sorted(set(PORTFOLIO_B_UNIVERSE)))

    def test_invalid_json_treated_as_none(self):
        tenant = _mock_tenant()
        tenant.ticker_whitelist = "not-json"
        result = get_tenant_universe(tenant, "B")
        assert len(result) == len(sorted(set(PORTFOLIO_B_UNIVERSE)))

    def test_uppercase_normalization(self):
        tenant = _mock_tenant(additions=["coin", "mstr"])
        result = get_tenant_universe(tenant, "B")
        assert "COIN" in result
        assert "MSTR" in result

    def test_exclusion_of_nonexistent_ticker(self):
        """Excluding a ticker not in the base universe is a no-op."""
        tenant = _mock_tenant(exclusions=["ZZZZZ"])
        result = get_tenant_universe(tenant, "B")
        assert len(result) == len(sorted(set(PORTFOLIO_B_UNIVERSE)))
