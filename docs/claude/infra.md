# Infrastructure & DevOps

Machine-readable context for Claude. Covers server, CI/CD, Docker, testing patterns, ruff, migration system, and utility scripts.

## Key Files

| File | Purpose |
|------|---------|
| `.github/workflows/deploy.yml` | CI/CD: test -> lint -> rsync -> migrate -> restart |
| `.github/workflows/migrate.yml` | Manual migration trigger (workflow_dispatch) |
| `deploy/nginx/kukulkan.trade` | Combined nginx config (both domains) |
| `deploy/systemd/kukulkan-bot.service` | Bot systemd service |
| `deploy/systemd/kukulkan-api.service` | API systemd service |
| `deploy/systemd/kukulkan-fe.service` | Frontend systemd service |
| `docker-compose.yml` | ChromaDB on port 8000 |
| `pyproject.toml` | Ruff, pytest, dependencies config |
| `scripts/migrate.py` | SQL migration runner |
| `scripts/backfill_snapshots.py` | Backfill missing daily snapshots |
| `scripts/fix_dup_positions.py` | Fix duplicate position rows |
| `scripts/rebalance_cash.py` | Manual cash rebalance between portfolios |
| `Makefile` | Common commands (test, lint, format, run) |
| `tests/conftest.py` | Shared test fixtures |

## Server (Hetzner)

- IP: `128.140.102.191`
- SSH: `ssh -i ~/.ssh/id_ed25519_personal root@128.140.102.191`
- Path: `/opt/kukulkan-trade`
- DB: `/opt/kukulkan-trade/data/kukulkan.db`
- Env: `/opt/kukulkan-trade/.env`
- User: `kukulkan` (services run as this user, not root)

### Systemd Services

| Service | Command | Port |
|---------|---------|------|
| `kukulkan-bot` | `python -m src.main` | -- |
| `kukulkan-api` | `uvicorn src.api.main:app --port 8001` | 8001 |
| `kukulkan-fe` | `npm start` (Next.js) | 3000 |

All run as `kukulkan` user with `WorkingDirectory=/opt/kukulkan-trade`.

### Nginx Config (`deploy/nginx/kukulkan.trade`)

- `kukulkan.trade` -> Next.js :3000 (landing page)
- `app.kukulkan.trade /api/auth/` -> Next.js :3000 (NextAuth)
- `app.kukulkan.trade /api/chat/` -> Next.js :3000 (chat auth proxy, SSE streaming)
- `app.kukulkan.trade /api/` -> FastAPI :8001
- `app.kukulkan.trade` (everything else) -> Next.js :3000
- Cloudflare Full SSL with self-signed origin cert (`/etc/nginx/ssl/kukulkan.crt`)
- Security headers: TLS 1.2+, CSP, HSTS, X-Frame-Options, rate limiting

## CI/CD

### Deploy Workflow (`.github/workflows/deploy.yml`)

Triggers on push to `main`. Steps:

1. **Test:** `python -m pytest tests/ -x -q`
2. **Lint:** `ruff check .`
3. **Deploy:** `rsync` to server (excludes: .git, .env, .venv, data/, __pycache__, .pytest_cache)
4. **Migrate:** `python scripts/migrate.py` (on server)
5. **Restart:** `systemctl restart kukulkan-bot kukulkan-api`

### Manual Migration (`.github/workflows/migrate.yml`)

- Trigger: `workflow_dispatch` (manual from GitHub Actions UI)
- SSHs to server, runs `python scripts/migrate.py`
- Supports `--dry-run` flag

## Docker

### ChromaDB (`docker-compose.yml`)

```yaml
services:
  chromadb:
    image: chromadb/chroma:latest
    ports: ["8000:8000"]
    volumes: ["./chroma_data:/chroma/chroma"]
```

Local storage at `./chroma_data/`. Used for news article vector storage.

## Testing Patterns

### Config (`pyproject.toml`)

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 120
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
```

### Database Fixture

```python
# tests/conftest.py
@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()
```

### API Test Pattern

```python
from httpx import ASGITransport, AsyncClient

@pytest.fixture
async def client(db):
    app.state.db = db
    _reset_rate_limiter()  # CRITICAL: rate limiter state persists
    app.dependency_overrides[get_current_user] = lambda: {"username": "admin", "tenant_id": None}
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.state.db = None  # CRITICAL: clean up stale DB reference
    app.dependency_overrides.clear()
```

### Key Test Fixtures & Resets

| What | How | Why |
|------|-----|-----|
| Rate limiter | `_reset_rate_limiter()` | State persists across tests |
| Run state | `_reset_run_state()` | Module-level `_running`/`_last_trigger` dicts |
| DB reference | `app.state.db = None` after yield | Prevents stale DB in login() |
| Auth bypass | Override `get_current_user` | Returns dict `{"username": ..., "tenant_id": ...}` |
| Executors | Mock must accept `**kwargs` | `execute_trades()` now takes `tenant_id` |
| Agentic tests | Patch `orch._strategy_b._agent.analyze()` | Two-phase flow calls it in Phase 1 |

### Test Count

1681 tests as of 2026-02-18.

## Ruff Configuration

```toml
[tool.ruff]
line-length = 120
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
```

- Run before committing: `ruff check . && ruff format .`
- `scripts/` directory has pre-existing E402 errors (load_dotenv before imports) -- acceptable
- Common lint errors to watch: F401 (unused imports), F841 (unused vars), E501 (line length), I001 (import sorting)

## Migration System (`scripts/migrate.py`)

```python
# Tracks applied migrations in `schema_migrations` table
# Files in migrations/ directory, numbered (001, 002, ...)
# Supports --dry-run flag
# Idempotent: skips already-applied migrations
```

### Migration History

| Migration | Phase | Description |
|-----------|-------|-------------|
| 001 | 23.1 | `tenants` table with nullable credentials |
| 002 | 23.1 | Schema migrations tracking table |
| 003 | 26 | `pending_rebalance` flag on tenants |
| 004 | 27 | `trailing_stops`, `earnings_calendar`, `watchlist` tables |
| 005 | 28 | `tenant_id` on `discovered_tickers` (composite unique) |
| 006 | 31 | `intraday_snapshots` table |
| 007 | 32 | `use_agent_loop` tenant flag + `tool_call_logs` table |
| 008 | 32.1 | `regime`/`session_label`/`influenced_decision` columns |
| 009 | 33 | `agent_conversations` table + `use_persistent_agent` tenant flag |
| 010 | 33.2 | `posture_history`, `playbook_snapshots`, `conviction_calibration` tables |
| 011 | 34 | `agent_budget_log` table + `use_tiered_models` tenant flag |
| 012 | 36 | Foreign key constraints on all 16 `tenant_id` columns → `tenants(id) ON DELETE CASCADE` |
| 013 | 38 | Improvement loop tables (proposals, trend analysis) |
| 014 | 42 | Extended hours: `intraday_snapshots` columns, `sentinel_actions` table, tenant quiet hours |

### SQLite Constraints

- No `ALTER COLUMN` -- must recreate table (CREATE new -> INSERT SELECT -> DROP old -> RENAME)
- No concurrent writes -- single-writer mode
- `create_all()` doesn't alter existing tables -- need migrations for schema changes

## Utility Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `migrate.py` | SQL migration runner |
| `backfill_snapshots.py` | Backfill missing daily snapshots from trade history |
| `fix_dup_positions.py` | Deduplicate position rows (data cleanup) |
| `rebalance_cash.py` | Manual cash redistribution between portfolios |

All scripts use `load_dotenv()` before imports (E402 acceptable).

## Dev-Only Tools

### Backtest Runner (`src/backtest/`)

Offline simulation engine. **Not deployed to production** — dev/research use only. Replays historical data through the pipeline and records hypothetical decisions. Results stored in `data/backtest_decisions/`.

### CLI (`src/cli/`)

Admin command-line interface for tenant management and manual operations. **Not deployed as a service** — used for ad-hoc server administration via SSH. Commands: `create-tenant`, `list-tenants`, `test-connection`, `run-pipeline`.

## Settings (`config/settings.py`)

```python
class Settings(BaseSettings):
    # Sub-settings (env prefix: ALPACA_, TELEGRAM_, FINNHUB_, AGENT_, CHROMA_, DASHBOARD_)
    alpaca: AlpacaSettings       # api_key, secret_key, paper
    telegram: TelegramSettings   # bot_token, chat_id
    finnhub: FinnhubSettings     # api_key
    chroma: ChromaSettings       # host, port
    dashboard: DashboardSettings # user, password
    agent: AgentSettings         # strategy_mode, daily_budget, monthly_budget, scan_model, etc.

    # Direct fields
    anthropic_api_key: str = ""
    fred_api_key: str = ""
    database_url: str = "sqlite+aiosqlite:///data/kukulkan.db"
    executor: str = "paper"
    log_level: str = "INFO"
    jwt_secret: str = "change-me-in-production"
    tenant_encryption_key: str = ""
    inter_tenant_delay: float = 2.0  # seconds between tenant runs

    # Paths
    project_root: Path; data_dir: Path; logs_dir: Path

settings = Settings()  # singleton, loaded at import
```

### Startup Validation

`_is_dev_or_test()` detects pytest via `sys.modules`, prevents validation in tests. In production, validates:
- `jwt_secret` is set and not default
- `tenant_encryption_key` is valid Fernet key (if set)

## Gotchas

- `_is_dev_or_test()` in settings.py detects pytest via `sys.modules` -- prevents startup validation in tests
- Rate limiter state persists across tests -- call `_reset_rate_limiter()` in API test fixtures
- Tests that set `app.state.db` must clean up after yield -- otherwise `login()` uses stale DB
- `POST /api/run` module-level state -- call `_reset_run_state()` in test fixtures
- Mock executors must accept `**kwargs` (tenant_id parameter added)
- Agentic tests must patch `orch._strategy_b._agent.analyze()` to avoid ANTHROPIC_API_KEY dependency in CI
- `AgentSettings` env prefix is `AGENT_` -- field `agent_tool_model` becomes env var `AGENT_AGENT_TOOL_MODEL`
