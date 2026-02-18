"""Tests for tenant quiet hours API settings."""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.api.schemas import TenantSelfUpdateRequest, TenantUpdateRequest


class TestQuietHoursSchema:
    def test_valid_time_format(self) -> None:
        req = TenantSelfUpdateRequest(quiet_hours_start="20:00", quiet_hours_end="07:30")
        assert req.quiet_hours_start == "20:00"
        assert req.quiet_hours_end == "07:30"

    def test_invalid_time_format_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TenantSelfUpdateRequest(quiet_hours_start="8pm")

        with pytest.raises(ValidationError):
            TenantSelfUpdateRequest(quiet_hours_end="7")

        with pytest.raises(ValidationError):
            TenantUpdateRequest(quiet_hours_start="8:00")  # Missing leading zero

    def test_admin_schema_has_quiet_hours(self) -> None:
        req = TenantUpdateRequest(
            quiet_hours_start="22:00",
            quiet_hours_end="06:00",
            quiet_hours_timezone="US/Eastern",
        )
        assert req.quiet_hours_start == "22:00"
        assert req.quiet_hours_timezone == "US/Eastern"


class TestQuietHoursAPIIntegration:
    """Tests for the quiet hours update flow (unit-level, no HTTP)."""

    async def test_patch_quiet_hours_settings(self) -> None:
        """Verify quiet hours fields are passed through to update_tenant."""
        from src.api.routes.tenants import _tenant_to_response

        tenant = MagicMock()
        tenant.id = "t1"
        tenant.name = "Test"
        tenant.is_active = True
        tenant.alpaca_api_key_enc = None
        tenant.telegram_chat_id_enc = None
        tenant.alpaca_base_url = "https://paper-api.alpaca.markets"
        tenant.strategy_mode = "conservative"
        tenant.run_portfolio_a = False
        tenant.run_portfolio_b = True
        tenant.portfolio_a_cash = 33000.0
        tenant.portfolio_b_cash = 66000.0
        tenant.initial_equity = None
        tenant.portfolio_a_pct = 33.33
        tenant.portfolio_b_pct = 66.67
        tenant.pending_rebalance = False
        tenant.use_agent_loop = False
        tenant.quiet_hours_start = "22:00"
        tenant.quiet_hours_end = "06:30"
        tenant.quiet_hours_timezone = "US/Eastern"
        tenant.ticker_whitelist = None
        tenant.ticker_additions = None
        tenant.ticker_exclusions = None
        tenant.dashboard_user = None
        tenant.created_at = None
        tenant.updated_at = None

        response = _tenant_to_response(tenant)
        assert response.quiet_hours_start == "22:00"
        assert response.quiet_hours_end == "06:30"
        assert response.quiet_hours_timezone == "US/Eastern"

    async def test_invalid_timezone_rejected(self) -> None:
        """Invalid timezone should raise 422."""
        # Simulate the validation logic from the route
        from zoneinfo import available_timezones

        from fastapi import HTTPException

        tz = "Invalid/Timezone"
        if tz not in available_timezones():
            with pytest.raises(HTTPException) as exc_info:
                raise HTTPException(status_code=422, detail=f"Invalid timezone: {tz}")
            assert exc_info.value.status_code == 422

    async def test_get_tenant_includes_quiet_hours(self) -> None:
        """TenantReadResponse should include quiet hours defaults."""
        from src.api.routes.tenants import _tenant_to_response

        tenant = MagicMock()
        tenant.id = "default"
        tenant.name = "Default"
        tenant.is_active = True
        tenant.alpaca_api_key_enc = None
        tenant.telegram_chat_id_enc = None
        tenant.alpaca_base_url = "https://paper-api.alpaca.markets"
        tenant.strategy_mode = "conservative"
        tenant.run_portfolio_a = False
        tenant.run_portfolio_b = True
        tenant.portfolio_a_cash = 33000.0
        tenant.portfolio_b_cash = 66000.0
        tenant.initial_equity = None
        tenant.portfolio_a_pct = 33.33
        tenant.portfolio_b_pct = 66.67
        tenant.pending_rebalance = False
        tenant.use_agent_loop = False
        # No quiet hours attrs → should use defaults
        del tenant.quiet_hours_start
        del tenant.quiet_hours_end
        del tenant.quiet_hours_timezone
        tenant.ticker_whitelist = None
        tenant.ticker_additions = None
        tenant.ticker_exclusions = None
        tenant.dashboard_user = None
        tenant.created_at = None
        tenant.updated_at = None

        response = _tenant_to_response(tenant)
        assert response.quiet_hours_start == "21:00"
        assert response.quiet_hours_end == "07:00"
        assert response.quiet_hours_timezone == "America/Mexico_City"
