"""CLI for tenant management.

Usage:
    python -m src.cli.tenant_cli add-tenant --name "Papa" ...
    python -m src.cli.tenant_cli list-tenants
    python -m src.cli.tenant_cli update-tenant <id> --strategy aggressive
    python -m src.cli.tenant_cli deactivate-tenant <id>
    python -m src.cli.tenant_cli test-tenant <id>
"""

import argparse
import asyncio
import json
import sys
import uuid

from config.settings import settings
from src.storage.database import Database
from src.storage.models import TenantRow
from src.utils.crypto import decrypt_value, encrypt_value, mask_credential


async def _get_db() -> Database:
    db = Database(url=settings.database_url)
    await db.init_db()
    return db


async def add_tenant(args: argparse.Namespace) -> None:
    """Create a new tenant."""
    db = await _get_db()
    try:
        tenant = TenantRow(
            id=str(uuid.uuid4()),
            name=args.name,
            alpaca_api_key_enc=(
                encrypt_value(args.alpaca_key) if args.alpaca_key else None
            ),
            alpaca_api_secret_enc=(
                encrypt_value(args.alpaca_secret) if args.alpaca_secret else None
            ),
            alpaca_base_url=args.alpaca_url,
            telegram_bot_token_enc=(
                encrypt_value(args.telegram_token) if args.telegram_token else None
            ),
            telegram_chat_id_enc=(
                encrypt_value(args.telegram_chat_id)
                if args.telegram_chat_id else None
            ),
            strategy_mode=args.strategy,
            run_portfolio_a=not args.portfolio_b_only,
            run_portfolio_b=True,
            portfolio_a_cash=args.portfolio_a_cash,
            portfolio_b_cash=args.portfolio_b_cash,
            ticker_additions=(
                json.dumps(args.add_tickers.split(","))
                if args.add_tickers else None
            ),
            ticker_exclusions=(
                json.dumps(args.remove_tickers.split(","))
                if args.remove_tickers else None
            ),
            dashboard_user=args.username,
            dashboard_password_enc=(
                encrypt_value(args.password) if args.password else None
            ),
        )
        await db.create_tenant(tenant)
        print(f"Tenant created: {tenant.id}")
        print(f"  Name: {tenant.name}")
        print(f"  Strategy: {tenant.strategy_mode}")
        print(f"  Portfolio A: {'yes' if tenant.run_portfolio_a else 'no'}")
        print(f"  Portfolio B: {'yes' if tenant.run_portfolio_b else 'no'}")
    finally:
        await db.close()


async def list_tenants(args: argparse.Namespace) -> None:
    """List all tenants."""
    db = await _get_db()
    try:
        tenants = await db.get_all_tenants()
        if not tenants:
            print("No tenants configured.")
            return
        for t in tenants:
            status = "active" if t.is_active else "INACTIVE"
            api_key = (
                mask_credential(decrypt_value(t.alpaca_api_key_enc))
                if t.alpaca_api_key_enc else "not set"
            )
            print(
                f"  [{status}] {t.id[:8]}... "
                f"{t.name} | {t.strategy_mode} | "
                f"Alpaca: {api_key} | "
                f"A={'yes' if t.run_portfolio_a else 'no'} "
                f"B={'yes' if t.run_portfolio_b else 'no'}"
            )
    finally:
        await db.close()


async def update_tenant(args: argparse.Namespace) -> None:
    """Update a tenant."""
    db = await _get_db()
    try:
        tenant = await db.get_tenant(args.tenant_id)
        if tenant is None:
            print(f"Tenant {args.tenant_id} not found.")
            sys.exit(1)

        updates: dict = {}
        if args.strategy:
            updates["strategy_mode"] = args.strategy
        if args.name:
            updates["name"] = args.name
        if args.add_tickers:
            updates["ticker_additions"] = json.dumps(args.add_tickers.split(","))
        if args.remove_tickers:
            updates["ticker_exclusions"] = json.dumps(args.remove_tickers.split(","))

        if not updates:
            print("No updates provided.")
            return

        updated = await db.update_tenant(args.tenant_id, updates)
        print(f"Tenant {updated.id[:8]}... updated: {list(updates.keys())}")
    finally:
        await db.close()


async def deactivate_tenant(args: argparse.Namespace) -> None:
    """Deactivate a tenant."""
    db = await _get_db()
    try:
        found = await db.deactivate_tenant(args.tenant_id)
        if found:
            print(f"Tenant {args.tenant_id} deactivated.")
        else:
            print(f"Tenant {args.tenant_id} not found.")
            sys.exit(1)
    finally:
        await db.close()


async def test_tenant(args: argparse.Namespace) -> None:
    """Test a tenant's Alpaca and Telegram connections."""
    db = await _get_db()
    try:
        tenant = await db.get_tenant(args.tenant_id)
        if tenant is None:
            print(f"Tenant {args.tenant_id} not found.")
            sys.exit(1)

        print(f"Testing tenant: {tenant.name} ({tenant.id[:8]}...)")

        # Test Alpaca
        print("\n  Alpaca connection...")
        try:
            from src.execution.client_factory import AlpacaClientFactory
            client = AlpacaClientFactory.get_trading_client(tenant)
            account = client.get_account()
            print(f"    OK — equity: ${float(account.equity):,.2f}")
        except Exception as e:
            print(f"    FAILED — {e}")

        # Test Telegram
        print("\n  Telegram connection...")
        try:
            from src.notifications.telegram_factory import TelegramFactory
            notifier = TelegramFactory.get_notifier(tenant)
            success = await notifier.send_message(
                "🐍 Kukulkan test message — connection verified!",
            )
            if success:
                print("    OK — test message sent")
            else:
                print("    FAILED — send returned False")
        except Exception as e:
            print(f"    FAILED — {e}")
    finally:
        await db.close()


async def seed_default(args: argparse.Namespace) -> None:
    """Seed the default tenant from environment variables."""
    db = await _get_db()
    try:
        existing = await db.get_all_tenants()
        if existing:
            print(f"Tenants already exist ({len(existing)}). Skipping seed.")
            return

        if not settings.alpaca.api_key or not settings.telegram.bot_token:
            print("Missing ALPACA_API_KEY or TELEGRAM_BOT_TOKEN in .env. Cannot seed.")
            sys.exit(1)

        tenant = TenantRow(
            id="default",
            name="Default",
            alpaca_api_key_enc=encrypt_value(settings.alpaca.api_key),
            alpaca_api_secret_enc=encrypt_value(settings.alpaca.secret_key),
            alpaca_base_url=(
                "https://paper-api.alpaca.markets"
                if settings.alpaca.paper
                else "https://api.alpaca.markets"
            ),
            telegram_bot_token_enc=encrypt_value(settings.telegram.bot_token),
            telegram_chat_id_enc=encrypt_value(settings.telegram.chat_id),
            strategy_mode=settings.agent.strategy_mode,
            run_portfolio_a=True,
            run_portfolio_b=True,
        )
        await db.create_tenant(tenant)
        print(f"Default tenant seeded from .env: {tenant.name}")
    finally:
        await db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Kukulkan Tenant Management")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # add-tenant
    add_p = subparsers.add_parser("add-tenant", help="Add a new tenant")
    add_p.add_argument("--name", required=True)
    add_p.add_argument("--alpaca-key", default=None, help="Alpaca API key (optional)")
    add_p.add_argument("--alpaca-secret", default=None, help="Alpaca API secret (optional)")
    add_p.add_argument("--alpaca-url", default="https://paper-api.alpaca.markets")
    add_p.add_argument("--telegram-token", default=None, help="Telegram bot token (optional)")
    add_p.add_argument("--telegram-chat-id", default=None, help="Telegram chat ID (optional)")
    add_p.add_argument("--strategy", default="conservative",
                        choices=["conservative", "standard", "aggressive"])
    add_p.add_argument("--portfolio-b-only", action="store_true")
    add_p.add_argument("--portfolio-a-cash", type=float, default=33000.0)
    add_p.add_argument("--portfolio-b-cash", type=float, default=66000.0)
    add_p.add_argument("--add-tickers", help="Comma-separated tickers to add")
    add_p.add_argument("--remove-tickers", help="Comma-separated tickers to remove")
    add_p.add_argument("--username", help="Dashboard login username")
    add_p.add_argument("--password", help="Dashboard login password")

    # list-tenants
    subparsers.add_parser("list-tenants", help="List all tenants")

    # update-tenant
    upd_p = subparsers.add_parser("update-tenant", help="Update a tenant")
    upd_p.add_argument("tenant_id")
    upd_p.add_argument("--name")
    upd_p.add_argument("--strategy", choices=["conservative", "standard", "aggressive"])
    upd_p.add_argument("--add-tickers", help="Comma-separated tickers to add")
    upd_p.add_argument("--remove-tickers", help="Comma-separated tickers to remove")

    # deactivate-tenant
    deact_p = subparsers.add_parser("deactivate-tenant", help="Deactivate a tenant")
    deact_p.add_argument("tenant_id")

    # test-tenant
    test_p = subparsers.add_parser("test-tenant", help="Test tenant connections")
    test_p.add_argument("tenant_id")

    # seed-default
    subparsers.add_parser("seed-default", help="Seed default tenant from .env")

    args = parser.parse_args()

    commands = {
        "add-tenant": add_tenant,
        "list-tenants": list_tenants,
        "update-tenant": update_tenant,
        "deactivate-tenant": deactivate_tenant,
        "test-tenant": test_tenant,
        "seed-default": seed_default,
    }

    asyncio.run(commands[args.command](args))


if __name__ == "__main__":
    main()
