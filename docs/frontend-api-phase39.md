# Phase 39: SSE Real-Time Events — Frontend API Reference

## Overview

Server-Sent Events (SSE) push trades, alerts, posture changes, and data refreshes to the dashboard in real time. The frontend should connect to the SSE stream on login and use events to trigger UI updates (toasts, re-fetches, live charts).

## New Endpoints

All endpoints require JWT authentication (`Authorization: Bearer <token>`).

### 1. SSE Event Stream

```
GET /api/events/stream?tenant_id=default
```

Long-lived SSE connection. Returns `text/event-stream`.

**Auth behavior:**
- Tenant users: receive only their own events (JWT `tenant_id` enforced, query param ignored)
- Admin users: receive all tenants' events (use `tenant_id` query param to filter, or omit for all)

**SSE message format:**

```
id: a1b2c3d4e5f6
event: trade_executed
data: {"ticker":"AAPL","side":"BUY","shares":10,"price":150.0,"portfolio":"B"}

```

Each message has: `id` (unique 12-char hex), `event` (event type string), `data` (JSON object). Messages are terminated by a blank line.

**Heartbeat:** Server sends a `heartbeat` event every 30 seconds to keep the connection alive.

```
id: f6e5d4c3b2a1
event: heartbeat
data: {"status":"ok"}

```

**Reconnection:** Use the `Last-Event-ID` header or call `/api/events/recent` on reconnect to catch up on missed events.

**Headers to expect:**

| Header | Value |
|-|-|
| `Content-Type` | `text/event-stream` |
| `Cache-Control` | `no-cache` |
| `Connection` | `keep-alive` |
| `X-Accel-Buffering` | `no` |

### 2. Recent Events (Catch-up)

```
GET /api/events/recent?tenant_id=default&limit=50
```

Returns the last N events from the server's in-memory buffer (max 100).

**Query params:**
- `tenant_id` (string, default `"default"`): Tenant scope (enforced for tenant users)
- `limit` (int, default 50, max 100): Number of events to return

**Response:** `RecentEvent[]`

```json
[
  {
    "id": "a1b2c3d4e5f6",
    "type": "trade_executed",
    "data": {"ticker": "AAPL", "side": "BUY", "shares": 10, "price": 150.0, "portfolio": "B"},
    "timestamp": 1739836800.123
  }
]
```

Events are ordered oldest-first. `timestamp` is Unix epoch seconds (float).

### 3. Active Connections (Admin Only)

```
GET /api/events/connections
```

**Requires:** Admin JWT (no `tenant_id` in token). Returns 403 for tenant users.

**Response:**

```json
{
  "total": 2,
  "connections": [
    {
      "id": "sub_abc123",
      "tenant_id": "default",
      "connected_seconds": 345,
      "queue_size": 0
    }
  ]
}
```

## Event Types & Payloads

### Trading Events

| Event | When | Payload |
|-|-|-|
| `trade_executed` | Each trade fills | `{"ticker","side","shares","price","portfolio"}` |
| `trailing_stop_triggered` | Trailing stop hit | `{"ticker","price","stop_price","portfolio"}` |
| `trade_rejected` | Risk manager blocks a trade | `{"ticker","side","shares","reason","portfolio"}` |
| `trade_approval_requested` | Large trade (>threshold%) sent for Telegram approval | `{"ticker","side","shares","price","value","portfolio_pct","reason"}` |
| `trade_approval_resolved` | Telegram approval/rejection received or timed out | `{"ticker","approved"}` |

### Alert Events

| Event | When | Payload |
|-|-|-|
| `circuit_breaker_triggered` | Portfolio halted | `{"portfolio","reason"}` |
| `session_started` | AI session begins | `{"trigger","session"}` |
| `session_completed` | AI session ends | `{"trades","cost_usd"}` |
| `session_skipped` | Budget exhausted | `{"reason","spent"}` |
| `posture_changed` | Posture declared | `{"declared","effective"}` |

### Data Refresh Events

| Event | When | Payload |
|-|-|-|
| `positions_updated` | After trades execute | `{"trades_executed"}` |
| `portfolio_snapshot` | End-of-day snapshot | `{"portfolio","date"}` |
| `intraday_update` | Intraday price sync | `{"portfolio","equity","cash"}` |
| `budget_updated` | After session cost recorded | `{"cost_usd","session_label"}` |
| `watchlist_updated` | After watchlist additions/removals | `{"additions","removals"}` |

### System Events

| Event | When | Payload |
|-|-|-|
| `heartbeat` | Every 30s (idle) | `{"status":"ok"}` |
| `system_error` | Pipeline step fails | `{"message","step"}` |
| `improvement_report` | Weekly improvement loop completes | `{"changes_applied","proposals_total","summary"}` |

## Design: Signals vs Payloads

Most events are **signals** — they carry a summary (for toasts) and the frontend should re-fetch via REST for full data. For example, on `trade_executed`, show a toast "Bought 10 AAPL @ $150" and call `GET /api/trades` to refresh the table.

**Exception:** `intraday_update` includes actual equity/cash values so the frontend can update live charts without a REST round-trip.

## Frontend Integration Guide

### 1. Connect on Login

```typescript
const token = getAccessToken();
const url = `${API_BASE}/api/events/stream?tenant_id=${tenantId}`;
const eventSource = new EventSource(url, {
  // Note: EventSource doesn't support custom headers natively.
  // Use a polyfill like 'eventsource-polyfill' or 'event-source-polyfill'
  // that supports Authorization headers, OR pass token as query param
  // if you add that to the backend.
});
```

Since the native `EventSource` API doesn't support custom headers, use one of:
- **`@microsoft/fetch-event-source`** — supports headers, POST, and auto-reconnect
- **`eventsource-polyfill`** — drop-in replacement with header support
- **Custom fetch-based SSE** — `fetch()` with `ReadableStream` parsing

Recommended approach with `@microsoft/fetch-event-source`:

```typescript
import { fetchEventSource } from '@microsoft/fetch-event-source';

fetchEventSource(`${API_BASE}/api/events/stream?tenant_id=${tenantId}`, {
  headers: { Authorization: `Bearer ${token}` },
  onmessage(event) {
    if (event.event === 'heartbeat') return;
    handleSSEEvent(event.event, JSON.parse(event.data));
  },
  onclose() { /* reconnect logic */ },
  onerror(err) { /* backoff + retry */ },
});
```

### 2. Event Handler Mapping

```typescript
function handleSSEEvent(type: string, data: Record<string, unknown>) {
  switch (type) {
    case 'trade_executed':
      showToast(`${data.side} ${data.shares} ${data.ticker} @ $${data.price}`);
      refetchTrades();
      refetchPositions();
      break;
    case 'trailing_stop_triggered':
      showToast(`Stop triggered: ${data.ticker} @ $${data.price}`, 'warning');
      refetchPositions();
      break;
    case 'trade_rejected':
      showToast(`Trade rejected: ${data.ticker} — ${data.reason}`, 'warning');
      break;
    case 'trade_approval_requested':
      showToast(`Approval pending: ${data.side} ${data.shares} ${data.ticker} (${data.portfolio_pct}%)`, 'info');
      break;
    case 'trade_approval_resolved':
      showToast(`${data.ticker}: ${data.approved ? 'Approved' : 'Rejected'}`, data.approved ? 'success' : 'warning');
      refetchTrades();
      break;
    case 'circuit_breaker_triggered':
      showToast(`Circuit breaker: ${data.portfolio} — ${data.reason}`, 'error');
      break;
    case 'session_started':
      setSessionStatus('running');
      break;
    case 'session_completed':
      setSessionStatus('idle');
      showToast(`Session done: ${data.trades} trades, $${data.cost_usd}`);
      refetchAll();
      break;
    case 'session_skipped':
      showToast(`Session skipped: ${data.reason}`, 'warning');
      break;
    case 'posture_changed':
      showToast(`Posture: ${data.declared} → ${data.effective}`);
      refetchPosture();
      break;
    case 'positions_updated':
      refetchPositions();
      break;
    case 'portfolio_snapshot':
      refetchSnapshots();
      break;
    case 'intraday_update':
      updateEquityCurve(data.portfolio, data.equity, data.cash);
      break;
    case 'budget_updated':
      refetchBudget();
      break;
    case 'watchlist_updated':
      showToast(`Watchlist: +${data.additions} / -${data.removals}`);
      break;
    case 'system_error':
      showToast(`Error in ${data.step}: ${data.message}`, 'error');
      break;
    case 'improvement_report':
      showToast(`Weekly improvement: ${data.changes_applied} changes applied`, 'info');
      break;
  }
}
```

### 3. Reconnection & Catch-up

On reconnect, call `GET /api/events/recent?limit=50` to fetch events that may have been missed during the disconnect window. Compare event `id` values to avoid duplicates.

### 4. Disconnect on Logout

Close the SSE connection when the user logs out or the token expires.

## TypeScript Types

```typescript
type SSEEventType =
  | 'trade_executed'
  | 'trailing_stop_triggered'
  | 'trade_rejected'
  | 'trade_approval_requested'
  | 'trade_approval_resolved'
  | 'circuit_breaker_triggered'
  | 'session_started'
  | 'session_completed'
  | 'session_skipped'
  | 'posture_changed'
  | 'positions_updated'
  | 'portfolio_snapshot'
  | 'intraday_update'
  | 'budget_updated'
  | 'watchlist_updated'
  | 'system_error'
  | 'improvement_report'
  | 'heartbeat';

interface TradeExecutedData {
  ticker: string;
  side: 'BUY' | 'SELL';
  shares: number;
  price: number;
  portfolio: 'A' | 'B';
}

interface TrailingStopTriggeredData {
  ticker: string;
  price: number;
  stop_price: number;
  portfolio: string;
}

interface CircuitBreakerData {
  portfolio: 'A' | 'B';
  reason: string;
}

interface SessionStartedData {
  trigger: string;
  session: string;
}

interface SessionCompletedData {
  trades: number;
  cost_usd: number;
}

interface SessionSkippedData {
  reason: string;
  spent: number;
}

interface PostureChangedData {
  declared: 'balanced' | 'defensive' | 'crisis' | 'aggressive';
  effective: 'balanced' | 'defensive' | 'crisis' | 'aggressive';
}

interface PositionsUpdatedData {
  trades_executed: number;
}

interface PortfolioSnapshotData {
  portfolio: 'A' | 'B';
  date: string; // YYYY-MM-DD
}

interface IntradayUpdateData {
  portfolio: 'A' | 'B';
  equity: number;
  cash: number;
}

interface BudgetUpdatedData {
  cost_usd: number;
  session_label: string;
}

interface TradeRejectedData {
  ticker: string;
  side: 'BUY' | 'SELL';
  shares: number;
  reason: string;
  portfolio: 'A' | 'B';
}

interface TradeApprovalRequestedData {
  ticker: string;
  side: 'BUY' | 'SELL';
  shares: number;
  price: number;
  value: number;
  portfolio_pct: number;
  reason: string;
}

interface TradeApprovalResolvedData {
  ticker: string;
  approved: boolean;
}

interface WatchlistUpdatedData {
  additions: number;
  removals: number;
}

interface SystemErrorData {
  message: string;
  step: string;
}

interface ImprovementReportData {
  changes_applied: number;
  proposals_total: number;
  summary: string;
}

interface RecentEvent {
  id: string;
  type: SSEEventType;
  data: Record<string, unknown>;
  timestamp: number; // Unix epoch seconds
}

interface ConnectionInfo {
  id: string;
  tenant_id: string | null;
  connected_seconds: number;
  queue_size: number;
}

interface ConnectionsResponse {
  total: number;
  connections: ConnectionInfo[];
}
```
