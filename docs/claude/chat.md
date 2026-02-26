# Chat System

Interactive conversation with the AI agent via the dashboard.

## Request Flow
```
Browser → nginx /api/chat/ → Next.js (auth proxy) → FastAPI → Claude CLI subprocess
```
- Nginx routes `/api/chat/` to Next.js (port 3000), NOT to FastAPI directly
- Next.js extracts NextAuth session → gets backend JWT via `getToken()` → forwards as Bearer token
- FastAPI streams SSE events from Claude CLI's `--output-format stream-json --verbose`

## Endpoints (src/api/routes/chat.py)
- `POST /api/chat` — non-streaming, returns complete response
- `POST /api/chat/stream` — SSE streaming (text, tool_use, tool_result, done, error events)
- `GET /api/chat/history` — recent messages from DB

## Session Persistence
- Chat sessions use `.chat-session-id` (no date expiry) — separate from orchestrator's daily `.session-id`
- Claude CLI `--resume <session_id>` continues the conversation
- Stale sessions (deleted on Anthropic's side) auto-clear and start fresh

## MCP Tools (src/agent/mcp_server.py)
- Registered at CLI startup from `mcp.json` (written by `_write_mcp_config()`)
- Tools read `session-state.json` for market data (closes, prices, VIX, regime)
- If no trading session ran today, `_refresh_session_state_if_stale()` fetches data on first chat message
- Tool categories: portfolio, market/technicals, news, actions (trade proposals, ticker discovery)

## Frontend (kukulkan-trade-fe)
- `src/hooks/useChatStream.ts` — SSE reader with line buffering (handles TCP chunk splits)
- `src/app/api/chat/stream/route.ts` — Next.js auth proxy for stream
- `src/app/api/chat/history/route.ts` — Next.js auth proxy for history

## Ticker Discovery via Chat
- Agent can propose tickers via MCP `propose_ticker_discovery` tool
- `_process_chat_discoveries()` in chat.py sends Telegram approval
- Uses `wait_for_ticker_approval(request_id, timeout_seconds=120)` — note: `timeout_seconds` not `timeout`

## Key Gotchas
- See `memory/gotchas.md` → "Chat System" section
