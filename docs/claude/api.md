# API & Authentication

Machine-readable context for Claude. Covers the FastAPI REST API, auth, routes, rate limiting, and schemas.

## Key Files

| File | Purpose |
|------|---------|
| `src/api/main.py` | FastAPI app, lifespan, CORS, security middleware, error handler, router registration |
| `src/api/auth.py` | JWT create/decode/revoke, login/logout endpoints, bcrypt + Fernet fallback |
| `src/api/deps.py` | get_current_user, require_admin, get_db, get_authorized_tenant_id |
| `src/api/rate_limit.py` | RateLimitMiddleware: sliding window per IP |
| `src/api/schemas.py` | All Pydantic response/request models |
| `src/api/alpaca_client.py` | Cached async Alpaca wrapper (30s TTL) |
| `src/api/routes/` | 13 route modules |

## FastAPI App (`src/api/main.py`)

```python
app = FastAPI(title="Kukulkan Trade API", version="0.1.0", lifespan=lifespan)
```

### Lifespan

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    db = Database(url=settings.database_url)
    await db.init_db()
    app.state.db = db
    yield
    await db.close()
```

### Middleware Chain (order matters)

1. `RateLimitMiddleware(general_rpm=60, login_rpm=5)`
2. `CORSMiddleware` -- origins: `https://app.kukulkan.trade`, `http://localhost:3000`
3. Security headers middleware: `Cache-Control: no-store` for `/api/*`, audit logging for login

### Error Handling

- `@app.exception_handler(Exception)` -- never leaks stack traces, returns generic 500
- HTTP error codes: 400 (validation), 401 (auth), 403 (admin), 404 (not found), 409 (conflict), 422 (precondition), 429 (rate limit), 503 (service unavailable)

### Health Endpoint

```python
@app.get("/api/health") -> {"status": "ok"}
```

## Auth (`src/api/auth.py`)

Router prefix: `/api/auth`, tags: `["auth"]`

```python
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 2
_revoked_tokens: set[str]  # in-memory JTI revocation

def create_access_token(subject: str, *, tenant_id: str | None = None) -> str
def decode_access_token(token: str) -> dict[str, str | None]  # {"username", "tenant_id"}
def revoke_token(token: str) -> None  # adds JTI to _revoked_tokens
```

### Endpoints

| Method | Path | Auth | Returns | Status |
|--------|------|------|---------|--------|
| POST | `/api/auth/login` | None | `TokenResponse` | 200 |
| POST | `/api/auth/logout` | Bearer | None | 204 |

### Login Flow

1. Try tenant login: DB lookup by username, check `dashboard_password_enc`
   - bcrypt if starts with `$2`, else Fernet fallback (re-hashes on success)
2. Fallback to global admin: timing-safe `hmac.compare_digest`
3. Return JWT with `sub=username`, `tenant_id=tenant.id` (or None for admin)
4. Auth failure: 401 "Invalid credentials"

**Important:** `login()` accesses DB via `request.app.state.db` (not Depends) to avoid circular import.

## Dependencies (`src/api/deps.py`)

```python
def get_db(request: Request) -> Database
    # Returns request.app.state.db

async def get_current_user(credentials) -> dict[str, str | None]
    # Decodes JWT -> {"username": str, "tenant_id": str | None}
    # On error: 401 "Invalid or expired token"

async def require_admin(user) -> dict[str, str | None]
    # Checks user["tenant_id"] is None (admins have no tenant_id)
    # On failure: 403 "Admin access required"

async def get_authorized_tenant_id(tenant_id: str = Query("default"), user=Depends(get_current_user)) -> str
    # Tenant users: ALWAYS returns their JWT tenant_id (IDOR protection)
    # Admins: returns requested tenant_id query param (default="default")
```

## Rate Limiting (`src/api/rate_limit.py`)

```python
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, general_rpm: int = 60, login_rpm: int = 5)
    def reset(self) -> None  # clears all state (for tests)
```

- Sliding window (60s), per-IP
- Login: 5 req/min -> 429 "Too many login attempts"
- General: 60 req/min -> 429 "Rate limit exceeded"
- `/api/health` is exempt

**Test gotcha:** Rate limiter state persists across tests -- call `_reset_rate_limiter()` in fixtures.

## Route Modules

### `src/api/routes/account.py` -- prefix `/api`, tags `["account"]`

| Method | Path | Query | Auth | Returns |
|--------|------|-------|------|---------|
| GET | `/account` | -- | `get_authorized_tenant_id` | `AccountResponse` |
| GET | `/account/history` | period, timeframe, extended_hours | `get_authorized_tenant_id` | `PortfolioHistoryResponse` |

History query params: `period: "1D|1W|1M|3M|1A"`, `timeframe: "1Min|5Min|15Min|1H|1D"`, `extended_hours: bool`

### `src/api/routes/portfolios.py` -- prefix `/api/portfolios`, tags `["portfolios"]`

| Method | Path | Auth | Returns |
|--------|------|------|---------|
| GET | `/` | `get_authorized_tenant_id` | `list[PortfolioSummary]` |
| GET | `/{name}` | `get_authorized_tenant_id` | `PortfolioDetail` |
| GET | `/{name}/positions` | `get_authorized_tenant_id` | `list[PositionResponse]` |
| GET | `/{name}/trailing-stops` | `get_authorized_tenant_id` | `list[dict]` |
| GET | `/{name}/watchlist` | `get_authorized_tenant_id` | `list[dict]` |

### `src/api/routes/snapshots.py` -- prefix `/api`, tags `["snapshots"]`

| Method | Path | Query | Auth | Returns |
|--------|------|-------|------|---------|
| GET | `/snapshots` | portfolio, since | `get_authorized_tenant_id` | `list[SnapshotResponse]` |
| GET | `/snapshots/intraday` | portfolio, period (1d/3d/1w/1m) | `get_authorized_tenant_id` | `list[IntradaySnapshotResponse]` |

### `src/api/routes/trades.py` -- prefix `/api`, tags `["trades"]`

| Method | Path | Query | Auth | Returns |
|--------|------|-------|------|---------|
| GET | `/trades` | portfolio, side, limit (1-1000) | `get_authorized_tenant_id` | `list[TradeResponse]` |

### `src/api/routes/momentum.py` -- prefix `/api/momentum`, tags `["momentum"]`

| Method | Path | Auth | Returns |
|--------|------|------|---------|
| GET | `/rankings` | `get_authorized_tenant_id` | `list[MomentumRankingResponse]` |

Note: Rankings are global (not tenant-scoped).

### `src/api/routes/decisions.py` -- prefix `/api/agent`, tags `["agent"]`

| Method | Path | Query | Auth | Returns |
|--------|------|-------|------|---------|
| GET | `/decisions` | limit (1-100) | `get_authorized_tenant_id` | `list[AgentDecisionResponse]` |
| GET | `/tool-logs` | session_date, limit (1-200) | `get_authorized_tenant_id` | `list[ToolCallLogResponse]` |

### `src/api/routes/tenants.py` -- prefix `/api/tenants`, tags `["tenants"]`

**Self-service (any authenticated user):**

| Method | Path | Auth | Returns | Status |
|--------|------|------|---------|--------|
| GET | `/me` | `get_current_user` | `TenantReadResponse` | 200 |
| PATCH | `/me` | `get_current_user` | `TenantReadResponse` | 200 |
| POST | `/me/test-alpaca` | `get_current_user` | `ConnectionTestResponse` | 200 |
| POST | `/me/test-telegram` | `get_current_user` | `ConnectionTestResponse` | 200 |

**Admin CRUD:**

| Method | Path | Auth | Returns | Status |
|--------|------|------|---------|--------|
| POST | `/` | `require_admin` | `TenantReadResponse` | 201 |
| GET | `/` | `require_admin` | `list[TenantReadResponse]` | 200 |
| GET | `/{tenant_id}` | `require_admin` | `TenantReadResponse` | 200 |
| PATCH | `/{tenant_id}` | `require_admin` | `TenantReadResponse` | 200 |
| DELETE | `/{tenant_id}` | `require_admin` | None | 204 |
| POST | `/{tenant_id}/test-alpaca` | `require_admin` | `ConnectionTestResponse` | 200 |
| POST | `/{tenant_id}/test-telegram` | `require_admin` | `ConnectionTestResponse` | 200 |

### `src/api/routes/universe.py` -- prefix `/api/universe`, tags `["universe"]`

| Method | Path | Auth | Returns |
|--------|------|------|---------|
| GET | `/base` | `get_current_user` | `{total: int, sectors: dict}` |

### `src/api/routes/earnings.py` -- prefix `/api/earnings`, tags `["earnings"]`

| Method | Path | Query | Auth | Returns |
|--------|------|-------|------|---------|
| GET | `/upcoming` | days_ahead (default 14) | `get_authorized_tenant_id` | `list[EarningsUpcomingResponse]` |

### `src/api/routes/discovered.py` -- prefix `/api/discovered`, tags `["discovered"]`

| Method | Path | Query/Body | Auth | Returns | Status |
|--------|------|------------|------|---------|--------|
| GET | `/` | status_filter | `get_authorized_tenant_id` | `list[DiscoveredTickerResponse]` | 200 |
| PATCH | `/{ticker}` | `{status: "approved|rejected"}` | `get_authorized_tenant_id` | `DiscoveredTickerResponse` | 200 |

Errors: 404 not found, 409 already approved/rejected.

### `src/api/routes/outcomes.py` -- prefix `/api/portfolios/B`, tags `["outcomes"]`

| Method | Path | Query | Auth | Returns |
|--------|------|-------|------|---------|
| GET | `/outcomes` | days (1-365, default 30) | `get_authorized_tenant_id` | `list[TradeOutcomeResponse]` |
| GET | `/track-record` | days (1-365, default 90) | `get_authorized_tenant_id` | `TrackRecordResponse` |
| GET | `/decision-quality` | days (1-365, default 30) | `get_authorized_tenant_id` | `DecisionQualityResponse` |

### `src/api/routes/run.py` -- prefix `/api`, tags `["run"]`

| Method | Path | Auth | Returns | Status |
|--------|------|------|---------|--------|
| POST | `/run` | `get_authorized_tenant_id` + `get_current_user` | `{status, tenant_id}` | 202 |

- Per-tenant concurrency lock: 1 active run -> 409
- Per-tenant rate limit: 60s cooldown -> 429
- Validates: tenant exists (404), credentials configured (422)
- Background task runs full orchestrator pipeline

```python
_running: dict[str, bool] = {}; _last_trigger: dict[str, float] = {}
def _reset_run_state() -> None  # for tests
```

### `src/api/routes/agent_insights.py` -- prefix `/api/agent`, tags `["agent-insights"]`

| Method | Path | Auth | Returns |
|--------|------|------|---------|
| GET | `/posture` | `get_authorized_tenant_id` | `list[PostureHistoryResponse]` |
| GET | `/playbook` | `get_authorized_tenant_id` | `list[PlaybookCellResponse]` |
| GET | `/calibration` | `get_authorized_tenant_id` | `list[ConvictionCalibrationResponse]` |
| GET | `/budget` | `get_authorized_tenant_id` | `BudgetStatusResponse` |
| GET | `/inverse-exposure` | `get_authorized_tenant_id` | `InverseExposureResponse` |

### `src/api/routes/conversations.py` -- prefix `/api/agent`, tags `["agent"]`

| Method | Path | Query | Auth | Returns |
|--------|------|-------|------|---------|
| GET | `/conversations` | limit (1-100, default 30) | `get_authorized_tenant_id` | `list[ConversationSessionResponse]` |
| GET | `/conversations/{session_id}` | -- | `get_authorized_tenant_id` | `ConversationDetailResponse` |

## Alpaca Client (`src/api/alpaca_client.py`)

```python
async def get_live_account() -> dict | None   # 30s TTL cache, returns account + positions
async def get_portfolio_history(period="1D", timeframe="5Min", extended_hours=False) -> dict | None
```

Uses `asyncio.to_thread()` for sync Alpaca SDK calls. Must use `client.get_portfolio_history(GetPortfolioHistoryRequest(...))` -- raw `client.get()` does NOT work for this endpoint.

## Key Schemas (`src/api/schemas.py`)

| Schema | Key Fields |
|--------|------------|
| `LoginRequest` | username, password |
| `TokenResponse` | access_token, token_type, tenant_id |
| `AccountResponse` | equity, last_equity, daily_pl, daily_pl_pct, cash, buying_power, positions |
| `PortfolioSummary` | name, cash, total_value, updated_at |
| `PortfolioDetail` | name, cash, total_value, updated_at, positions |
| `SnapshotResponse` | portfolio, date, total_value, cash, daily_return_pct, cumulative_return_pct |
| `IntradaySnapshotResponse` | portfolio, timestamp, total_value, cash, positions_value |
| `TradeResponse` | id, portfolio, ticker, side, shares, price, total, reason, executed_at |
| `AgentDecisionResponse` | id, date, proposed_trades, reasoning, model_used, tokens_used, regime, session_label |
| `TenantCreateRequest` | name, username, password + optional credentials/config |
| `TenantSelfUpdateRequest` | credentials, strategy_mode, portfolio toggles, tickers, use_agent_loop (NO username/password) |
| `TenantReadResponse` | all fields, credentials masked, nullable |
| `TradeOutcomeResponse` | ticker, side, pnl_pct, alpha_vs_spy, verdict, regime_at_entry, session_at_entry |
| `TrackRecordResponse` | total_trades, wins, losses, win_rate_pct, by_sector/conviction/regime/session |
| `DecisionQualityResponse` | total_decisions, favorable_1d/3d/5d_pct |
| `ToolCallLogResponse` | session_date, turn, tool_name, tool_input, success, influenced_decision |
| `ConversationSessionResponse` | session_id, trigger_type, summary, token_count, cost_usd, session_status |
| `ConversationDetailResponse` | session_id, tenant_id, trigger_type, messages, created_at |
| `PostureHistoryResponse` | session_date, posture, effective_posture, reason |
| `PlaybookCellResponse` | regime, sector, total_trades, win_rate_pct, recommendation |
| `ConvictionCalibrationResponse` | conviction_level, win_rate_pct, assessment, suggested_multiplier |
| `BudgetStatusResponse` | daily_spent/limit/remaining, monthly_spent/limit/remaining, haiku_only |
| `InverseExposureResponse` | total_value, total_pct, net_equity_pct, positions, rules |
| `EarningsUpcomingResponse` | ticker, earnings_date, source |
| `ConnectionTestResponse` | success, equity, message, error |

Custom validator: `UTCDatetime = Annotated[datetime, AfterValidator(_ensure_utc)]`

## Gotchas

- `login()` accesses DB via `request.app.state.db`, NOT Depends -- avoids circular import
- Tests that set `app.state.db` must clean up (`app.state.db = None`) after yield
- `decode_access_token` returns `dict` -- all `_bypass_user` test fixtures must return dict
- `get_authorized_tenant_id`: tenant users always get JWT tenant_id, admins can use query param
- Rate limiter state persists across tests -- call `_reset_rate_limiter()` in fixtures
- `POST /api/run` uses module-level `_running`/`_last_trigger` dicts -- call `_reset_run_state()` in test fixtures
- Portfolio config changes (toggle, strategy_mode) via PATCH require Alpaca+Telegram credentials -> 422
- Credential updates invalidate cached clients (AlpacaClientFactory, TelegramFactory)
