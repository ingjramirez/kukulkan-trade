# Kukulkan Trade — Project Rules

## Context Efficiency

### Subagent Discipline
Prefer inline work for tasks under ~5 tool calls. Subagents have overhead — don't delegate trivially.
When using subagents, include output rules: "Final response under 2000 characters. List outcomes, not process."
Never call TaskOutput twice for the same subagent. If it times out, increase the timeout — don't re-read.

### File Reading
Read files with purpose. Before reading a file, know what you're looking for.
Use Grep to locate relevant sections before reading entire large files.
Never re-read a file you've already read in this session.
For files over 500 lines, use offset/limit to read only the relevant section.

### Responses
Don't echo back file contents you just read — the user can see them.
Don't narrate tool calls ("Let me read the file..." / "Now I'll edit..."). Just do it.
Keep explanations proportional to complexity. Simple changes need one sentence, not three paragraphs.

For markdown tables, use the minimum valid separator (`|-|-|` — one hyphen per column). Never use repeated hyphens (`|---|---|`), box-drawing characters, or padded separators. This saves tokens.

## Context Loading Rules
Before modifying code, read the relevant context file first:
- `src/agent/`, `src/analysis/` → read `docs/claude/agent.md`
- `src/api/routes/chat.py`, chat SSE, MCP tools → read `docs/claude/chat.md`
- `src/api/`, API schemas → read `docs/claude/api.md`
- `src/orchestrator.py`, `src/main.py`, `src/execution/` → read `docs/claude/pipeline.md`
- `src/storage/`, migrations → read `docs/claude/data.md`
- Regime, momentum, risk manager → read `docs/claude/analysis.md`
- Tenant system, allocations, crypto → read `docs/claude/tenants.md`
- Deploy, CI/CD, nginx, systemd → read `docs/claude/infra.md`
- Debugging unexpected behavior → read `memory/gotchas.md`

## Mandatory: Update Project Memory
After completing significant work, update `~/.claude/projects/-Users-jramirezolmos-Documents-personal-kukulkan-kukulkan-trade/memory/MEMORY.md`:
- Update test count, current state
- Add new gotchas to `memory/gotchas.md` (NOT MEMORY.md — keep it lean)
- MEMORY.md should stay under 80 lines — it's an index, not an encyclopedia

## Code Style
- Python 3.11, type hints on all function signatures
- Pydantic v2 for validation, structlog for logging
- `ruff` for linting and formatting (line-length=120, target py311). Run `ruff format` before committing.
- pytest-asyncio with `asyncio_mode = "auto"`
- **Before committing, always run `ruff check` and fix all lint errors** (F401, F841, E501, I001). The `scripts/` directory has pre-existing E402 errors that are acceptable.

## Testing
- All changes must include tests. Run `python -m pytest tests/ -x -q` before committing.
- In-memory SQLite for tests (`sqlite+aiosqlite:///:memory:`)
- FastAPI tests use httpx.AsyncClient + ASGITransport with dependency overrides
- Rate limiter state persists across tests — call `_reset_rate_limiter()` in API test fixtures

## Deployment
- Server: Hetzner 128.140.102.191, path `/opt/kukulkan-trade`
- Deploy via GitHub Actions CI/CD (push to main triggers rsync)
- Services: `kukulkan-bot`, `kukulkan-api`, `kukulkan-fe` (systemd, runs as `kukulkan` user)

## Security
- API is read-only except: login/logout (POST) and chat endpoints (POST /api/chat, POST /api/chat/stream)
- JWT auth with 2-hour expiry, in-memory token revocation
- Rate limiting: 60 req/min general, 5 req/min login
- CORS restricted to `app.kukulkan.trade` + `localhost:3000`
- Timing-safe password comparison (hmac.compare_digest)
- Telegram callbacks validated against authorized chat_id
