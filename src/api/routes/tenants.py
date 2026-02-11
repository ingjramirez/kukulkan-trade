"""Tenant management API — admin CRUD + tenant self-service."""

from __future__ import annotations

import json
import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_current_user, get_db, require_admin
from src.api.schemas import (
    TenantCreateRequest,
    TenantReadResponse,
    TenantSelfUpdateRequest,
    TenantUpdateRequest,
)
from src.storage.database import Database
from src.storage.models import TenantRow
from src.utils.crypto import decrypt_value, encrypt_value, hash_password, mask_credential

log = structlog.get_logger()

router = APIRouter(prefix="/api/tenants", tags=["tenants"])


def _tenant_to_response(tenant: TenantRow) -> TenantReadResponse:
    """Convert a TenantRow to the safe API response schema."""
    api_key_masked = (
        mask_credential(decrypt_value(tenant.alpaca_api_key_enc))
        if tenant.alpaca_api_key_enc else None
    )
    chat_id_masked = (
        mask_credential(decrypt_value(tenant.telegram_chat_id_enc))
        if tenant.telegram_chat_id_enc else None
    )
    return TenantReadResponse(
        id=tenant.id,
        name=tenant.name,
        is_active=tenant.is_active,
        alpaca_api_key_masked=api_key_masked,
        alpaca_base_url=tenant.alpaca_base_url or "https://paper-api.alpaca.markets",
        telegram_chat_id_masked=chat_id_masked,
        strategy_mode=tenant.strategy_mode,
        run_portfolio_a=tenant.run_portfolio_a,
        run_portfolio_b=tenant.run_portfolio_b,
        portfolio_a_cash=tenant.portfolio_a_cash,
        portfolio_b_cash=tenant.portfolio_b_cash,
        initial_equity=tenant.initial_equity,
        portfolio_a_pct=tenant.portfolio_a_pct,
        portfolio_b_pct=tenant.portfolio_b_pct,
        ticker_whitelist=(
            json.loads(tenant.ticker_whitelist) if tenant.ticker_whitelist else None
        ),
        ticker_additions=(
            json.loads(tenant.ticker_additions) if tenant.ticker_additions else None
        ),
        ticker_exclusions=(
            json.loads(tenant.ticker_exclusions) if tenant.ticker_exclusions else None
        ),
        dashboard_user=tenant.dashboard_user,
        created_at=tenant.created_at,
        updated_at=tenant.updated_at,
    )


# ── Tenant self-service (must be before /{tenant_id} routes) ─────────────


@router.get("/me", response_model=TenantReadResponse)
async def get_my_tenant(
    db: Database = Depends(get_db),
    user: dict = Depends(get_current_user),
) -> TenantReadResponse:
    """Get the current tenant user's own tenant info."""
    tenant_id = user.get("tenant_id")
    if tenant_id is None:
        raise HTTPException(status_code=403, detail="Not a tenant user")
    tenant = await db.get_tenant(tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return _tenant_to_response(tenant)


@router.patch("/me", response_model=TenantReadResponse)
async def update_my_tenant(
    body: TenantSelfUpdateRequest,
    db: Database = Depends(get_db),
    user: dict = Depends(get_current_user),
) -> TenantReadResponse:
    """Update the current tenant user's own credentials and ticker customizations."""
    tenant_id = user.get("tenant_id")
    if tenant_id is None:
        raise HTTPException(status_code=403, detail="Not a tenant user")
    tenant = await db.get_tenant(tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail="Tenant not found")

    updates: dict = {}
    if body.alpaca_api_key is not None:
        updates["alpaca_api_key_enc"] = encrypt_value(body.alpaca_api_key)
    if body.alpaca_api_secret is not None:
        updates["alpaca_api_secret_enc"] = encrypt_value(body.alpaca_api_secret)
    if body.alpaca_base_url is not None:
        updates["alpaca_base_url"] = body.alpaca_base_url
    if body.telegram_bot_token is not None:
        updates["telegram_bot_token_enc"] = encrypt_value(body.telegram_bot_token)
    if body.telegram_chat_id is not None:
        updates["telegram_chat_id_enc"] = encrypt_value(body.telegram_chat_id)
    if body.ticker_whitelist is not None:
        updates["ticker_whitelist"] = (
            json.dumps(body.ticker_whitelist) if body.ticker_whitelist else None
        )
    if body.ticker_additions is not None:
        updates["ticker_additions"] = (
            json.dumps(body.ticker_additions) if body.ticker_additions else None
        )
    if body.ticker_exclusions is not None:
        updates["ticker_exclusions"] = (
            json.dumps(body.ticker_exclusions) if body.ticker_exclusions else None
        )

    if not updates:
        return _tenant_to_response(tenant)

    # Invalidate cached clients when credentials change
    if any(k.endswith("_enc") for k in updates):
        from src.execution.client_factory import AlpacaClientFactory
        from src.notifications.telegram_factory import TelegramFactory
        AlpacaClientFactory.invalidate(tenant_id)
        TelegramFactory.invalidate(tenant_id)

    updated = await db.update_tenant(tenant_id, updates)
    return _tenant_to_response(updated)


@router.post("/me/test-alpaca")
async def test_my_alpaca(
    db: Database = Depends(get_db),
    user: dict = Depends(get_current_user),
) -> dict:
    """Test the current tenant user's own Alpaca connection."""
    tenant_id = user.get("tenant_id")
    if tenant_id is None:
        raise HTTPException(status_code=403, detail="Not a tenant user")
    tenant = await db.get_tenant(tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail="Tenant not found")

    if not tenant.alpaca_api_key_enc or not tenant.alpaca_api_secret_enc:
        return {"success": False, "error": "Alpaca credentials not configured"}

    try:
        import asyncio

        from src.execution.client_factory import AlpacaClientFactory
        client = AlpacaClientFactory.get_trading_client(tenant)
        account = await asyncio.to_thread(client.get_account)
        return {"success": True, "equity": float(account.equity)}
    except Exception:
        log.error("test_alpaca_failed", tenant_id=tenant_id)
        return {"success": False, "error": "Connection failed. Check credentials and try again."}


@router.post("/me/test-telegram")
async def test_my_telegram(
    db: Database = Depends(get_db),
    user: dict = Depends(get_current_user),
) -> dict:
    """Send a test message via the current tenant user's own Telegram bot."""
    tenant_id = user.get("tenant_id")
    if tenant_id is None:
        raise HTTPException(status_code=403, detail="Not a tenant user")
    tenant = await db.get_tenant(tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail="Tenant not found")

    if not tenant.telegram_bot_token_enc or not tenant.telegram_chat_id_enc:
        return {"success": False, "error": "Telegram credentials not configured"}

    try:
        from src.notifications.telegram_factory import TelegramFactory
        notifier = TelegramFactory.get_notifier(tenant)
        success = await notifier.send_message(
            "🐍 Kukulkan test message — connection verified!",
        )
        if success:
            return {"success": True, "message": "Test message sent"}
        return {"success": False, "error": "Send returned False"}
    except Exception:
        log.error("test_telegram_failed", tenant_id=tenant_id)
        return {
            "success": False, "error": "Connection failed. Check credentials and try again.",
        }


# ── Admin CRUD ───────────────────────────────────────────────────────────


@router.post("", response_model=TenantReadResponse, status_code=201)
async def create_tenant(
    body: TenantCreateRequest,
    db: Database = Depends(get_db),
    _user: dict = Depends(require_admin),
) -> TenantReadResponse:
    """Create a new tenant with encrypted credentials."""
    # Check username uniqueness
    if body.username:
        existing = await db.get_tenant_by_username(body.username)
        if existing:
            raise HTTPException(
                status_code=409, detail="Username already taken",
            )

    tenant = TenantRow(
        id=str(uuid.uuid4()),
        name=body.name,
        alpaca_api_key_enc=(
            encrypt_value(body.alpaca_api_key) if body.alpaca_api_key else None
        ),
        alpaca_api_secret_enc=(
            encrypt_value(body.alpaca_api_secret) if body.alpaca_api_secret else None
        ),
        alpaca_base_url=body.alpaca_base_url,
        telegram_bot_token_enc=(
            encrypt_value(body.telegram_bot_token) if body.telegram_bot_token else None
        ),
        telegram_chat_id_enc=(
            encrypt_value(body.telegram_chat_id) if body.telegram_chat_id else None
        ),
        strategy_mode=body.strategy_mode,
        run_portfolio_a=body.run_portfolio_a,
        run_portfolio_b=body.run_portfolio_b,
        portfolio_a_cash=body.portfolio_a_cash,
        portfolio_b_cash=body.portfolio_b_cash,
        portfolio_a_pct=body.portfolio_a_pct,
        portfolio_b_pct=body.portfolio_b_pct,
        ticker_whitelist=(
            json.dumps(body.ticker_whitelist) if body.ticker_whitelist else None
        ),
        ticker_additions=(
            json.dumps(body.ticker_additions) if body.ticker_additions else None
        ),
        ticker_exclusions=(
            json.dumps(body.ticker_exclusions) if body.ticker_exclusions else None
        ),
        dashboard_user=body.username,
        dashboard_password_enc=(
            hash_password(body.password) if body.password else None
        ),
    )
    await db.create_tenant(tenant)
    return _tenant_to_response(tenant)


@router.get("", response_model=list[TenantReadResponse])
async def list_tenants(
    db: Database = Depends(get_db),
    _user: dict = Depends(require_admin),
) -> list[TenantReadResponse]:
    """List all tenants (active and inactive, credentials masked)."""
    tenants = await db.get_all_tenants()
    return [_tenant_to_response(t) for t in tenants]


@router.get("/{tenant_id}", response_model=TenantReadResponse)
async def get_tenant(
    tenant_id: str,
    db: Database = Depends(get_db),
    _user: dict = Depends(require_admin),
) -> TenantReadResponse:
    """Get a single tenant by ID."""
    tenant = await db.get_tenant(tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return _tenant_to_response(tenant)


@router.patch("/{tenant_id}", response_model=TenantReadResponse)
async def update_tenant(
    tenant_id: str,
    body: TenantUpdateRequest,
    db: Database = Depends(get_db),
    _user: dict = Depends(require_admin),
) -> TenantReadResponse:
    """Update a tenant's config (strategy, tickers, active status, credentials)."""
    tenant = await db.get_tenant(tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail="Tenant not found")

    # Portfolio config fields require Alpaca + Telegram to be configured
    # (ticker lists are preferences, not trading ops — allowed without creds)
    _portfolio_fields = {
        "run_portfolio_a", "run_portfolio_b", "portfolio_a_cash",
        "portfolio_b_cash", "strategy_mode",
    }
    has_portfolio_update = any(
        getattr(body, f) is not None for f in _portfolio_fields
    )
    if has_portfolio_update:
        # Check current + incoming credentials
        has_alpaca = bool(
            (body.alpaca_api_key or tenant.alpaca_api_key_enc)
            and (body.alpaca_api_secret or tenant.alpaca_api_secret_enc)
        )
        has_telegram = bool(
            (body.telegram_bot_token or tenant.telegram_bot_token_enc)
            and (body.telegram_chat_id or tenant.telegram_chat_id_enc)
        )
        if not (has_alpaca and has_telegram):
            raise HTTPException(
                status_code=422,
                detail="Alpaca and Telegram credentials must be configured "
                       "before setting portfolio configuration",
            )

    updates: dict = {}
    if body.name is not None:
        updates["name"] = body.name
    if body.alpaca_api_key is not None:
        updates["alpaca_api_key_enc"] = encrypt_value(body.alpaca_api_key)
    if body.alpaca_api_secret is not None:
        updates["alpaca_api_secret_enc"] = encrypt_value(body.alpaca_api_secret)
    if body.alpaca_base_url is not None:
        updates["alpaca_base_url"] = body.alpaca_base_url
    if body.telegram_bot_token is not None:
        updates["telegram_bot_token_enc"] = encrypt_value(body.telegram_bot_token)
    if body.telegram_chat_id is not None:
        updates["telegram_chat_id_enc"] = encrypt_value(body.telegram_chat_id)
    if body.strategy_mode is not None:
        updates["strategy_mode"] = body.strategy_mode
    if body.run_portfolio_a is not None:
        updates["run_portfolio_a"] = body.run_portfolio_a
    if body.run_portfolio_b is not None:
        updates["run_portfolio_b"] = body.run_portfolio_b
    if body.portfolio_a_cash is not None:
        updates["portfolio_a_cash"] = body.portfolio_a_cash
    if body.portfolio_b_cash is not None:
        updates["portfolio_b_cash"] = body.portfolio_b_cash
    if body.portfolio_a_pct is not None:
        updates["portfolio_a_pct"] = body.portfolio_a_pct
    if body.portfolio_b_pct is not None:
        updates["portfolio_b_pct"] = body.portfolio_b_pct
    if body.is_active is not None:
        updates["is_active"] = body.is_active
    if body.username is not None:
        # Check uniqueness (exclude current tenant)
        existing = await db.get_tenant_by_username(body.username)
        if existing and existing.id != tenant_id:
            raise HTTPException(
                status_code=409, detail="Username already taken",
            )
        updates["dashboard_user"] = body.username
    if body.password is not None:
        updates["dashboard_password_enc"] = hash_password(body.password)
    # Ticker lists: allow setting to empty [] or null
    if body.ticker_whitelist is not None:
        updates["ticker_whitelist"] = (
            json.dumps(body.ticker_whitelist) if body.ticker_whitelist else None
        )
    if body.ticker_additions is not None:
        updates["ticker_additions"] = (
            json.dumps(body.ticker_additions) if body.ticker_additions else None
        )
    if body.ticker_exclusions is not None:
        updates["ticker_exclusions"] = (
            json.dumps(body.ticker_exclusions) if body.ticker_exclusions else None
        )

    if not updates:
        return _tenant_to_response(tenant)

    # Invalidate cached clients when credentials change
    if any(k.endswith("_enc") for k in updates):
        from src.execution.client_factory import AlpacaClientFactory
        from src.notifications.telegram_factory import TelegramFactory
        AlpacaClientFactory.invalidate(tenant_id)
        TelegramFactory.invalidate(tenant_id)

    updated = await db.update_tenant(tenant_id, updates)
    return _tenant_to_response(updated)


@router.delete("/{tenant_id}", status_code=204)
async def deactivate_tenant(
    tenant_id: str,
    db: Database = Depends(get_db),
    _user: dict = Depends(require_admin),
):
    """Soft-delete a tenant (set is_active=False)."""
    found = await db.deactivate_tenant(tenant_id)
    if not found:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return None


@router.post("/{tenant_id}/test-alpaca")
async def test_alpaca(
    tenant_id: str,
    db: Database = Depends(get_db),
    _user: dict = Depends(require_admin),
) -> dict:
    """Test a tenant's Alpaca connection by calling get_account()."""
    tenant = await db.get_tenant(tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail="Tenant not found")

    if not tenant.alpaca_api_key_enc or not tenant.alpaca_api_secret_enc:
        return {"success": False, "error": "Alpaca credentials not configured"}

    try:
        from src.execution.client_factory import AlpacaClientFactory
        client = AlpacaClientFactory.get_trading_client(tenant)
        import asyncio
        account = await asyncio.to_thread(client.get_account)
        return {
            "success": True,
            "equity": float(account.equity),
        }
    except Exception:
        log.error("test_alpaca_failed", tenant_id=tenant_id)
        return {"success": False, "error": "Connection failed. Check credentials and try again."}


@router.post("/{tenant_id}/test-telegram")
async def test_telegram(
    tenant_id: str,
    db: Database = Depends(get_db),
    _user: dict = Depends(require_admin),
) -> dict:
    """Send a test message via a tenant's Telegram bot."""
    tenant = await db.get_tenant(tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail="Tenant not found")

    if not tenant.telegram_bot_token_enc or not tenant.telegram_chat_id_enc:
        return {"success": False, "error": "Telegram credentials not configured"}

    try:
        from src.notifications.telegram_factory import TelegramFactory
        notifier = TelegramFactory.get_notifier(tenant)
        success = await notifier.send_message(
            "🐍 Kukulkan test message — connection verified!",
        )
        if success:
            return {"success": True, "message": "Test message sent"}
        return {"success": False, "error": "Send returned False"}
    except Exception:
        log.error("test_telegram_failed", tenant_id=tenant_id)
        return {
            "success": False, "error": "Connection failed. Check credentials and try again.",
        }
