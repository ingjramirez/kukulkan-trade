# Kukulkan Trade — Project Rules

## Mandatory: Update Project Memory
After completing any significant work (features, bug fixes, refactors, config changes), **always update the project memory** at `~/.claude/projects/-Users-jramirezolmos-Documents-personal-kukulkan-trade/memory/MEMORY.md` before finishing the session. This includes:
- New phases, features, or architectural changes
- Updated test counts
- New gotchas or lessons learned
- Changed file paths or key configurations

## Code Style
- Python 3.11, type hints on all function signatures
- Pydantic v2 for validation, structlog for logging
- `ruff` for linting and formatting (line-length=120, target py311). Run `ruff format` before committing.
- pytest-asyncio with `asyncio_mode = "auto"`
- **Before committing, always run `ruff check` and fix all lint errors** (unused imports F401, unused variables F841, line length E501, import sorting I001). The `scripts/` directory has pre-existing E402 errors that are acceptable (load_dotenv before imports).

## Testing
- All changes must include tests. Run `python -m pytest tests/ -x -q` before committing.
- In-memory SQLite for tests (`sqlite+aiosqlite:///:memory:`)
- FastAPI tests use httpx.AsyncClient + ASGITransport with dependency overrides
- Rate limiter state persists across tests — call `_reset_rate_limiter()` in API test fixtures

## Deployment
- Server: Hetzner 128.140.102.191, path `/opt/kukulkan-trade`
- Deploy via GitHub Actions CI/CD (push to main triggers rsync)
- Services: `kukulkan-bot`, `kukulkan-api`, `kukulkan-fe` (systemd)
- Services run as `kukulkan` user (not root)

## Security
- API is read-only (all GET except login/logout)
- JWT auth with 2-hour expiry, in-memory token revocation
- Rate limiting: 60 req/min general, 5 req/min login
- CORS restricted to `app.kukulkan.trade` + `localhost:3000`
- Timing-safe password comparison (hmac.compare_digest)
- Telegram callbacks validated against authorized chat_id
