# Multi-Tenant System

Machine-readable context for Claude. Covers tenant model, lifecycle, credential management, allocations, crypto, CLI, and SQL migrations.

## Key Files

| File | Purpose |
|------|---------|
| `src/storage/models.py` | TenantRow ORM model with all columns |
| `src/storage/database.py` | Tenant CRUD methods |
| `src/api/routes/tenants.py` | Admin CRUD + self-service /me endpoints |
| `src/api/deps.py` | get_authorized_tenant_id (IDOR protection) |
| `src/utils/allocations.py` | TenantAllocations, resolve_allocations(), deposit detection |
| `src/utils/crypto.py` | encrypt_value(), decrypt_value(), Fernet key management |
| `src/cli/tenant_cli.py` | CLI commands for tenant management |
| `src/execution/client_factory.py` | AlpacaClientFactory: per-tenant cached TradingClient |
| `src/notifications/telegram_factory.py` | TelegramFactory: per-tenant cached notifier |
| `scripts/migrate.py` | SQL migration runner |
| `migrations/` | Numbered SQL migration files (001-008) |
| `config/universe.py` | get_tenant_universe(): per-tenant ticker customization |

## TenantRow (`src/storage/models.py`)

| Column | Type | Default | Nullable | Notes |
|--------|------|---------|----------|-------|
| `id` | String(36) | uuid4 | no | PK |
| `name` | String(100) | -- | no | Display name |
| `is_active` | Boolean | True | no | Soft-delete flag |
| `alpaca_api_key_enc` | Text | -- | yes | Fernet-encrypted |
| `alpaca_api_secret_enc` | Text | -- | yes | Fernet-encrypted |
| `alpaca_base_url` | String(200) | "https://paper-api.alpaca.markets" | no | |
| `telegram_bot_token_enc` | Text | -- | yes | Fernet-encrypted |
| `telegram_chat_id_enc` | Text | -- | yes | Fernet-encrypted |
| `strategy_mode` | String(20) | "conservative" | no | conservative/standard/aggressive |
| `run_portfolio_a` | Boolean | True | no | |
| `run_portfolio_b` | Boolean | True | no | |
| `portfolio_a_cash` | Float | 0.0 | no | Tracked cash for Portfolio A |
| `portfolio_b_cash` | Float | 0.0 | no | Tracked cash for Portfolio B |
| `initial_equity` | Float | -- | yes | Captured from Alpaca on first run |
| `portfolio_a_pct` | Float | 33.33 | no | Allocation percentage |
| `portfolio_b_pct` | Float | 66.67 | no | Allocation percentage |
| `pending_rebalance` | Boolean | False | no | Set by API on toggle change |
| `use_agent_loop` | Boolean | False | no | Enable agentic tool-use for Portfolio B |
| `ticker_whitelist` | JSON | -- | yes | If set, replaces base universe |
| `ticker_additions` | JSON | -- | yes | Tickers added to base universe |
| `ticker_exclusions` | JSON | -- | yes | Tickers removed from base universe |
| `dashboard_user` | String(50) | -- | yes | Login username |
| `dashboard_password_enc` | Text | -- | yes | bcrypt hash (starts with "$2") or Fernet (legacy) |
| `created_at` | DateTime | utcnow | no | |
| `updated_at` | DateTime | utcnow | no | |

## Tenant Lifecycle

1. **Create:** `POST /api/tenants` or CLI `add-tenant` -- requires name + username + password only
2. **Configure credentials:** `PATCH /api/tenants/me` -- add Alpaca + Telegram credentials
3. **Test connections:** `POST /api/tenants/me/test-alpaca` and `test-telegram`
4. **Enable portfolios:** `PATCH /api/tenants/me` -- set `run_portfolio_a/b`, `strategy_mode`
5. **Bot runs:** Orchestrator iterates active tenants, skips those without complete credentials
6. **Toggle portfolios:** API sets `pending_rebalance=True`; next bot run liquidates + redistributes

### Credential Guards

- **Orchestrator:** `_tenant_fully_configured()` skips tenants missing Alpaca or Telegram credentials
- **API PATCH:** Rejects portfolio config changes (run_portfolio_a/b, strategy_mode) with 422 unless both Alpaca AND Telegram credentials are set
- **Test endpoints:** Check credentials exist before attempting connection

## JWT & Auth

- Admin users: `tenant_id=None` in JWT claims
- Tenant users: `tenant_id=<uuid>` in JWT claims
- `get_authorized_tenant_id`: tenant users ALWAYS get their JWT tenant_id (IDOR protection); admins can use query param
- `require_admin`: checks `user["tenant_id"] is None`

## Allocations (`src/utils/allocations.py`)

```python
RECONCILE_THRESHOLD = 10  # $10 minimum drift to correct

@dataclass
class TenantAllocations:
    initial_equity: float
    portfolio_a_pct: float      # default 33.33
    portfolio_b_pct: float      # default 66.67
    portfolio_a_cash: float
    portfolio_b_cash: float
    def for_portfolio(self, name: str) -> float

DEFAULT_ALLOCATIONS = TenantAllocations(initial_equity=99_000, portfolio_a_pct=33.33, portfolio_b_pct=66.67, ...)

def resolve_allocations(initial_equity, a_pct, b_pct, a_cash, b_cash) -> TenantAllocations
def resolve_from_tenant(tenant: TenantRow) -> TenantAllocations
```

### Deposit Detection

Uses Alpaca `/v2/account/activities` API (CSD/JNLC activity types) instead of equity comparison (which causes false positives from overnight price movements). Equity-gap guard prevents double-counting across intraday runs.

### Equity Reconciliation

Step 8.5 in orchestrator: compares broker equity to sum of tracked portfolio totals. Corrects $10-$50 drift proportionally across enabled portfolios. Skips if drift <= RECONCILE_THRESHOLD or > $50 (likely a deposit).

## Crypto (`src/utils/crypto.py`)

```python
def encrypt_value(value: str) -> str    # Fernet encryption using TENANT_ENCRYPTION_KEY
def decrypt_value(encrypted: str) -> str
```

- `TENANT_ENCRYPTION_KEY` loaded from env at startup, validated in `config/settings.py`
- `encrypt_value()` expects `str`, crashes on `None` -- guard with `if value else None`
- Passwords: bcrypt hashes stored in `dashboard_password_enc` (starts with `$2`)
- Legacy Fernet-encrypted passwords: fallback on login, re-hashed to bcrypt on success

## Factories

### AlpacaClientFactory (`src/execution/client_factory.py`)

```python
class AlpacaClientFactory:
    _cache: dict[str, TradingClient]
    @classmethod def get_trading_client(cls, tenant: TenantRow) -> TradingClient
    @classmethod def invalidate(cls, tenant_id: str) -> None
    @classmethod def clear_cache(cls) -> None
```

Decrypts tenant's Alpaca credentials, creates and caches `TradingClient`. Invalidated on credential update.

### TelegramFactory (`src/notifications/telegram_factory.py`)

```python
class TelegramFactory:
    _cache: dict[str, TelegramNotifier]
    @classmethod def get_notifier(cls, tenant: TenantRow) -> TelegramNotifier
    @classmethod def invalidate(cls, tenant_id: str) -> None
```

## CLI (`src/cli/tenant_cli.py`)

```bash
python -m src.cli.tenant_cli add-tenant --name "User" --username user1 --password pass123
python -m src.cli.tenant_cli list-tenants
python -m src.cli.tenant_cli seed-default  # Create default tenant from .env
```

## Per-Tenant Ticker Customization (`config/universe.py`)

```python
async def get_tenant_universe(tenant: TenantRow, discovered_tickers: list[str] | None = None) -> list[str]
```

- `ticker_whitelist`: if set, REPLACES base universe entirely
- `ticker_additions`: appended to base (or whitelist)
- `ticker_exclusions`: removed from final set
- Discovered tickers (approved) are always included

## SQL Migration System

### Runner (`scripts/migrate.py`)

```python
# Tracks applied migrations in `schema_migrations` table
# Supports --dry-run
# CI/CD: deploy.yml runs between pip install and service restart
# Manual: .github/workflows/migrate.yml (workflow_dispatch)
```

### SQLite Constraints

No `ALTER COLUMN` in SQLite. Must recreate table:
1. `CREATE TABLE new_table (...)`
2. `INSERT INTO new_table SELECT ... FROM old_table`
3. `DROP TABLE old_table`
4. `ALTER TABLE new_table RENAME TO old_table`

### Migration Files (`migrations/`)

| File | Description |
|------|-------------|
| `001_tenants_nullable_credentials.sql` | Make Alpaca/Telegram credentials nullable |
| `002_dashboard_user_column.sql` | Add dashboard_user + dashboard_password_enc |
| `003_portfolio_toggle.sql` | Add pending_rebalance flag |
| `004_trailing_stops_earnings_watchlist.sql` | 3 new tables |
| `005_discovered_tenant_id.sql` | Add tenant_id to discovered_tickers |
| `006_intraday_snapshots.sql` | IntradaySnapshotRow table |
| `007_agent_loop_flag.sql` | Add use_agent_loop to tenants |
| `008_outcome_tracking.sql` | Add regime/session_label to decisions, influenced_decision to tool logs |

## Gotchas

- `encrypt_value()` expects `str`, crashes on `None` -- guard with `if value else None`
- `_tenant_to_response()` must check enc fields before `decrypt_value()` (nullable now)
- `update_tenant()` expects a plain dict, NOT a Pydantic model -- call with `{"field": value}`
- SQLAlchemy column defaults (e.g. `default=33.33`) only apply on INSERT -- in-memory TenantRow without DB round-trip has `None` for defaulted fields
- `dashboard_password_enc` now stores bcrypt hashes (start with `$2`), login has Fernet fallback for migration
- `get_authorized_tenant_id` dependency: tenant users always get JWT tenant_id, admins can use query param
- CLI test Namespace must include all args or AttributeError
- Tests that set `app.state.db` must clean up (`app.state.db = None`) after yield
