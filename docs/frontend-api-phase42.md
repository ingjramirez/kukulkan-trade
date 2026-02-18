# Phase 42: Extended Hours Monitoring — Frontend API Reference

## Overview

Phase 42 extends monitoring beyond regular market hours (9:30 AM – 4:00 PM ET) to pre-market (7:00–9:30 AM) and after-hours (4:00–8:00 PM). It adds overnight gap risk analysis, quiet hours per tenant, and enriches existing endpoints with market phase context.

## New Endpoints

All endpoints require JWT authentication (`Authorization: Bearer <token>`).

### 1. After-Hours P&L

```
GET /api/portfolios/after-hours-pnl
```

Returns portfolio P&L vs. the last regular-hours close snapshot. Only active during pre-market or after-hours.

**Response (active):**

```json
{
  "is_active": true,
  "market_phase": "afterhours",
  "as_of": "2026-02-17T17:30:00-05:00",
  "market_close_value": 52000.00,
  "current_value": 52150.00,
  "change": 150.00,
  "change_pct": 0.29,
  "movers": [
    {
      "ticker": "NVDA",
      "close_price": 850.00,
      "current_price": 855.00,
      "change_pct": 0.59,
      "contribution": 50.00
    }
  ]
}
```

**Response (inactive — during market hours or closed):**

```json
{
  "is_active": false,
  "market_phase": "market"
}
```

| Field | Type | Description |
|-|-|-|
| `is_active` | boolean | Whether extended hours session is in progress |
| `market_phase` | string | Current phase: `"premarket"`, `"afterhours"`, `"market"`, `"closed"` |
| `market_close_value` | number | Portfolio value at last regular-hours close |
| `current_value` | number | Portfolio value at extended-hours prices |
| `change` | number | Dollar change from close |
| `change_pct` | number | Percentage change from close |
| `movers` | array | Top 10 positions sorted by absolute contribution |

**Mover object:**

| Field | Type | Description |
|-|-|-|
| `ticker` | string | Ticker symbol |
| `close_price` | number | Price at regular-hours close |
| `current_price` | number | Current extended-hours price |
| `change_pct` | number | Percentage move from close |
| `contribution` | number | Dollar contribution to P&L (positive or negative) |

### 2. Overnight Gap Risk

```
GET /api/portfolios/overnight-risk
```

Returns overnight gap risk assessment. Can be called at any time; most useful near market close (2:30–4:00 PM ET).

**Response:**

```json
{
  "as_of": "2026-02-17T14:45:00-05:00",
  "aggregate_risk_score": 18.5,
  "rating": "HIGH",
  "earnings_tonight": ["NVDA"],
  "positions": [
    {
      "ticker": "NVDA",
      "weight_pct": 12.5,
      "gap_risk_score": 56.3,
      "reasons": ["Earnings tonight", "Volatile sector (Technology)"],
      "recommendation": "Consider reducing before close"
    },
    {
      "ticker": "AAPL",
      "weight_pct": 8.0,
      "gap_risk_score": 12.0,
      "reasons": ["Volatile sector (Technology)"],
      "recommendation": null
    }
  ]
}
```

| Field | Type | Description |
|-|-|-|
| `aggregate_risk_score` | number | Weighted sum of all position gap risk scores |
| `rating` | string | `"LOW"` (0–5), `"MODERATE"` (5–15), `"HIGH"` (15–30), `"EXTREME"` (30+) |
| `earnings_tonight` | string[] | Held tickers reporting earnings tonight |
| `positions` | array | Per-position risk breakdown, sorted by gap_risk_score desc |

**Position risk object:**

| Field | Type | Description |
|-|-|-|
| `ticker` | string | Ticker symbol |
| `weight_pct` | number | Position weight as % of portfolio |
| `gap_risk_score` | number | Computed risk score with multipliers |
| `reasons` | string[] | Risk factors: `"Earnings tonight"`, `"Volatile sector (X)"`, `"Large position (X%)"`, `"Inverse ETF (hedge)"` |
| `recommendation` | string\|null | Action suggestion if risk is elevated, null otherwise |

## Modified Endpoints

### 3. Intraday Snapshots (enriched)

```
GET /api/snapshots/intraday?tenant_id=default
```

Response now includes two new fields per snapshot:

```json
{
  "snapshots": [
    {
      "portfolio": "B",
      "timestamp": "2026-02-17T10:30:00",
      "total_value": 52000.00,
      "cash": 5000.00,
      "positions_value": 47000.00,
      "is_extended_hours": false,
      "market_phase": "market"
    },
    {
      "portfolio": "B",
      "timestamp": "2026-02-17T17:00:00",
      "total_value": 52150.00,
      "cash": 5000.00,
      "positions_value": 47150.00,
      "is_extended_hours": true,
      "market_phase": "afterhours"
    }
  ]
}
```

| New Field | Type | Default | Description |
|-|-|-|-|
| `is_extended_hours` | boolean | `false` | Whether snapshot was taken outside regular hours |
| `market_phase` | string | `"market"` | Phase when snapshot was taken: `"premarket"`, `"market"`, `"afterhours"`, `"closed"` |

### 4. Tenant Settings (quiet hours)

**Self-service:** `PATCH /api/tenants/me`
**Admin:** `PATCH /api/tenants/{tenant_id}`

New writable fields:

```json
{
  "quiet_hours_start": "22:00",
  "quiet_hours_end": "07:00",
  "quiet_hours_timezone": "America/Mexico_City"
}
```

| Field | Type | Validation | Default |
|-|-|-|-|
| `quiet_hours_start` | string | Regex `^\d{2}:\d{2}$` (24h format) | `"21:00"` |
| `quiet_hours_end` | string | Regex `^\d{2}:\d{2}$` (24h format) | `"07:00"` |
| `quiet_hours_timezone` | string | Must be a valid IANA timezone | `"America/Mexico_City"` |

Invalid timezone returns `422 Unprocessable Entity`:

```json
{"detail": "Invalid timezone: Foo/Bar"}
```

**Read response** (`GET /api/tenants/me`, `GET /api/tenants/{tenant_id}`) now includes:

```json
{
  "quiet_hours_start": "21:00",
  "quiet_hours_end": "07:00",
  "quiet_hours_timezone": "America/Mexico_City"
}
```

## SSE Event Changes

### Updated: `intraday_update`

Now includes extended hours context:

```json
{
  "portfolio": "B",
  "equity": 52150.00,
  "cash": 5000.00,
  "is_extended_hours": true,
  "market_phase": "afterhours"
}
```

| New Field | Type | Description |
|-|-|-|
| `is_extended_hours` | boolean | Whether this update is from an extended-hours snapshot |
| `market_phase` | string | `"premarket"`, `"market"`, `"afterhours"`, or `"closed"` |

### Existing: `sentinel_alert` / `sentinel_escalation`

No payload changes. These events now fire during extended hours too (pre-market and after-hours sentinel runs). During quiet hours, Telegram notifications are queued but SSE events still fire in real time.

## TypeScript Types

```typescript
// ── Market Phase ────────────────────────────────────────────────────
type MarketPhase = "premarket" | "market" | "afterhours" | "closed";

// ── After-Hours P&L ─────────────────────────────────────────────────
interface AfterHoursMover {
  ticker: string;
  close_price: number;
  current_price: number;
  change_pct: number;
  contribution: number;
}

interface AfterHoursPnLResponse {
  is_active: boolean;
  market_phase?: MarketPhase;
  as_of?: string;
  reason?: string; // present when is_active=false and no snapshot
  market_close_value?: number;
  current_value?: number;
  change?: number;
  change_pct?: number;
  movers?: AfterHoursMover[];
}

// ── Overnight Risk ──────────────────────────────────────────────────
type GapRiskRating = "LOW" | "MODERATE" | "HIGH" | "EXTREME";

interface PositionGapRisk {
  ticker: string;
  weight_pct: number;
  gap_risk_score: number;
  reasons: string[];
  recommendation: string | null;
}

interface OvernightRiskResponse {
  as_of: string;
  aggregate_risk_score: number;
  rating: GapRiskRating;
  earnings_tonight: string[];
  positions: PositionGapRisk[];
}

// ── Updated Intraday Snapshot ───────────────────────────────────────
interface IntradaySnapshot {
  portfolio: "A" | "B";
  timestamp: string;
  total_value: number;
  cash: number;
  positions_value: number;
  is_extended_hours: boolean;  // NEW
  market_phase: MarketPhase;   // NEW
}

// ── Updated SSE: intraday_update ────────────────────────────────────
interface IntradayUpdateData {
  portfolio: "A" | "B";
  equity: number;
  cash: number;
  is_extended_hours: boolean;  // NEW
  market_phase: MarketPhase;   // NEW
}

// ── Tenant Quiet Hours ──────────────────────────────────────────────
interface TenantQuietHours {
  quiet_hours_start: string;  // "HH:MM" 24h format
  quiet_hours_end: string;    // "HH:MM" 24h format
  quiet_hours_timezone: string; // IANA timezone
}

// Add to existing TenantReadResponse:
// extends ... & TenantQuietHours

// Add to existing TenantUpdateRequest:
// quiet_hours_start?: string;
// quiet_hours_end?: string;
// quiet_hours_timezone?: string;
```

## Frontend Component Suggestions

### After-Hours Banner

Show a banner at the top of the dashboard during pre-market/after-hours:

```
Poll GET /api/portfolios/after-hours-pnl every 60s when:
  - is_active === true
  - OR on intraday_update SSE events with is_extended_hours === true

Display:
  - Market phase badge: "Pre-Market" (blue) or "After-Hours" (purple)
  - Change: "+$150.00 (+0.29%)" with green/red coloring
  - Top movers as mini-cards or a compact table
  - Dismiss when is_active === false
```

### Overnight Risk Card

Show near market close (2:30–4:00 PM ET) or on the portfolio detail page:

```
Fetch GET /api/portfolios/overnight-risk once

Display:
  - Rating badge: LOW (green), MODERATE (yellow), HIGH (orange), EXTREME (red)
  - Aggregate score with simple risk meter/gauge
  - Earnings callout if earnings_tonight is non-empty
  - Position risk table with ticker, weight, score, reasons, recommendation
  - Highlight positions where recommendation is non-null
```

### Intraday Chart Enhancement

Extend the existing intraday equity chart:

```
When rendering IntradaySnapshot data:
  - Use market_phase to color-code or shade chart regions
  - Pre-market: light blue background
  - Market: default (white/transparent)
  - After-hours: light purple background
  - Add vertical lines at 9:30 AM ET (market open) and 4:00 PM ET (market close)
  - Extended snapshots appear as dashed lines or distinct markers
```

### Quiet Hours Settings

Add to the tenant settings form:

```
Fields:
  - Start time: time input (HH:MM, 24h) — default "21:00"
  - End time: time input (HH:MM, 24h) — default "07:00"
  - Timezone: dropdown of common timezones — default "America/Mexico_City"

PATCH /api/tenants/me with { quiet_hours_start, quiet_hours_end, quiet_hours_timezone }
Show validation error from 422 response if timezone is invalid
```

## SSE Event Handler Updates

Add to the existing `handleSSEEvent` switch:

```typescript
case 'intraday_update':
  updateEquityCurve(data.portfolio, data.equity, data.cash);
  // NEW: handle extended hours context
  if (data.is_extended_hours) {
    showExtendedHoursBanner(data.market_phase, data.equity);
  }
  break;
```

## Daily Timeline (for context)

| Time (ET) | Backend Activity | Frontend Impact |
|-|-|-|
| 7:00–9:30 | Pre-market snapshots (15min) + sentinel | `intraday_update` SSE with `market_phase: "premarket"` |
| 9:30 | Morning session + queue delivery | Session events + morning alert dismissal |
| 10:00–15:00 | Regular intraday (15min) + sentinel (30min) | Standard `intraday_update` SSE |
| 14:45 | Gap risk alert (Telegram) | Overnight risk data available via API |
| 15:30 | Close session + gap risk context | Session events |
| 16:00–20:00 | After-hours snapshots (15min) + sentinel | `intraday_update` SSE with `market_phase: "afterhours"` |
| 21:00–07:00 | Quiet hours (notifications queued) | SSE events still fire; no Telegram |
