# Frontend API Changes — Phase 33 (Persistent Agent)

## New Endpoints

### 1. `GET /api/agent/conversations`
List persistent agent conversation sessions (most recent first).

**Query Parameters:**
- `limit` (int, default 30, max 100): Number of sessions to return

**Response:** `ConversationSessionResponse[]`
```json
[
  {
    "session_id": "t1-morning-abc123",
    "trigger_type": "morning",
    "summary": "Bought NVDA at $118, set trailing stop...",
    "token_count": 5000,
    "cost_usd": 0.15,
    "session_status": "completed",
    "created_at": "2026-02-15T15:00:00Z"
  }
]
```

**Fields:**
- `session_id`: Unique ID (format: `{tenant_id}-{trigger_type}-{random}`)
- `trigger_type`: One of `morning`, `midday`, `close`, `event`, `weekly_review`
- `summary`: Compressed summary (null for recent sessions not yet compressed)
- `token_count`: Total tokens used in the session
- `cost_usd`: Estimated API cost
- `session_status`: `completed` or `started` (started = crashed/incomplete)
- `created_at`: UTC timestamp

### 2. `GET /api/agent/conversations/{session_id}`
Get a single conversation session with full message history.

**Response:** Full session object
```json
{
  "session_id": "t1-morning-abc123",
  "tenant_id": "default",
  "trigger_type": "morning",
  "messages": [
    {"role": "user", "content": "Good morning. VIX 18.2..."},
    {"role": "assistant", "content": [
      {"type": "text", "text": "Let me check the portfolio."},
      {"type": "tool_use", "id": "t1", "name": "get_portfolio_state", "input": {}}
    ]},
    {"role": "user", "content": [
      {"type": "tool_result", "tool_use_id": "t1", "content": "{\"cash\": 25000}"}
    ]},
    {"role": "assistant", "content": "Portfolio healthy. No trades needed."}
  ],
  "summary": null,
  "token_count": 5000,
  "cost_usd": 0.15,
  "session_status": "completed",
  "created_at": "2026-02-15T15:00:00Z"
}
```

**Notes:**
- `messages` follows Anthropic's message format (content can be string or array of blocks)
- Tool use blocks have `type: "tool_use"` with `name` and `input`
- Tool result blocks have `type: "tool_result"` with `tool_use_id` and `content`
- Returns `{"detail": "Session not found"}` for non-existent or unauthorized sessions
- Messages are cleared after 30 days (will be empty `[]`) but summary persists

## Tenant Changes

### `TenantRow` new field
- `use_persistent_agent` (boolean, default false): Enables persistent conversation mode for Portfolio B

This field is set via `PATCH /api/tenants/{id}` (admin) or will be exposed via `PATCH /api/tenants/me` in a future update.

## Suggested Frontend Features

### Conversation History Page
- Table/list view using `GET /api/agent/conversations`
- Columns: trigger_type, created_at, token_count, cost_usd, session_status
- Click to expand: show full messages from detail endpoint
- Filter by trigger_type
- Summary preview on hover

### Message Rendering
- User messages: render as plain text
- Assistant messages: render text blocks, collapse tool_use blocks
- Tool results: collapsible JSON viewer
- Highlight trade proposals in assistant responses

### Cost Tracking Widget
- Sum `cost_usd` across recent sessions
- Daily/weekly/monthly cost trends
- Token usage chart
