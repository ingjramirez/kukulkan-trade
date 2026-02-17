# Phase 38: Weekly Self-Improvement Loop — Frontend API Reference

## New Endpoints

All endpoints require JWT authentication (`Authorization: Bearer <token>`).

### 1. List Improvement Snapshots

```
GET /api/agent/improvements
```

Returns recent weekly improvement snapshots (newest first, max 20).

**Response:** `ImprovementSnapshotListResponse[]`

```json
[
  {
    "id": 1,
    "week_start": "2026-02-10",
    "week_end": "2026-02-17",
    "total_trades": 15,
    "win_rate_pct": 60.0,
    "avg_pnl_pct": 1.5,
    "strategy_mode": "conservative",
    "trailing_stop_multiplier": 1.0,
    "created_at": "2026-02-17T16:00:00Z"
  }
]
```

### 2. Get Snapshot Detail

```
GET /api/agent/improvements/{snapshot_id}
```

Returns full details including AI proposal and applied changes.

**Response:** `ImprovementSnapshotDetailResponse`

```json
{
  "id": 1,
  "week_start": "2026-02-10",
  "week_end": "2026-02-17",
  "total_trades": 15,
  "win_rate_pct": 60.0,
  "avg_pnl_pct": 1.5,
  "avg_alpha_vs_spy": 0.8,
  "total_cost_usd": 2.50,
  "strategy_mode": "conservative",
  "trailing_stop_multiplier": 1.0,
  "proposal_json": {
    "changes": [
      {
        "category": "learning",
        "parameter": "sector_insight",
        "new_value": "Tech outperforms in bull regime",
        "reason": "Consistent pattern over 4 weeks"
      }
    ],
    "summary": "Solid week with 60% win rate."
  },
  "applied_changes": [
    {
      "parameter": "sector_insight",
      "old_value": null,
      "new_value": "Tech outperforms in bull regime",
      "reason": "Consistent pattern over 4 weeks",
      "status": "applied"
    }
  ],
  "report_text": "Weekly Improvement Report: 2026-02-10 to 2026-02-17\n...",
  "created_at": "2026-02-17T16:00:00Z"
}
```

**404** if snapshot not found or belongs to another tenant.

### 3. Parameter Changelog

```
GET /api/agent/improvements/changelog
```

Returns audit log of auto-applied parameter changes (newest first, max 50).

**Response:** `ParameterChangelogResponse[]`

```json
[
  {
    "id": 1,
    "parameter": "strategy_mode",
    "old_value": "conservative",
    "new_value": "standard",
    "reason": "Win rate supports standard mode",
    "snapshot_id": 1,
    "applied_at": "2026-02-17T16:00:00Z"
  }
]
```

### 4. Performance Trend

```
GET /api/agent/improvements/trend?weeks=8
```

Returns trend analysis (linear regression) across weekly snapshots.

**Query params:**
- `weeks` (int, default 8): Number of recent weeks to analyze.

**Response:** `ImprovementTrendResponse`

```json
{
  "classification": "improving",
  "win_rate_slope": 5.2,
  "pnl_slope": 0.8,
  "data_points": [
    {
      "week_label": "2026-01-06",
      "win_rate_pct": 45.0,
      "avg_pnl_pct": 0.5,
      "total_trades": 12
    },
    {
      "week_label": "2026-01-13",
      "win_rate_pct": 52.0,
      "avg_pnl_pct": 1.1,
      "total_trades": 10
    }
  ],
  "weeks_analyzed": 4
}
```

**Classification values:** `"improving"`, `"stable"`, `"declining"`, `"insufficient_data"`

**Slopes** are in percentage points per week. Positive = improving, negative = declining.

## Change Categories

The `proposal_json.changes[].category` field can be:

| Category | What it changes | Bounds |
|----------|----------------|--------|
| `strategy_mode` | Tenant strategy mode | `conservative`, `standard`, `aggressive` |
| `trailing_stop` | Trailing stop multiplier | 0.5-2.0 (scales TRAIL_PCT matrix) |
| `universe_exclude` | Adds ticker to exclusion list | Max 3 per week |
| `learning` | Saves insight to agent memory | Max 3 per week, 500 char limit |

## Applied Change Statuses

| Status | Meaning |
|--------|---------|
| `applied` | Change was successfully applied |
| `blocked_flipflop` | Blocked due to too many recent changes (3+ in 4 weeks) |
| `unknown_category` | Category not recognized |
| `error` | Exception during apply |

## Frontend Suggestions

1. **Improvements Dashboard**: Timeline of weekly snapshots with win rate chart
2. **Trend Widget**: Show classification badge (green=improving, yellow=stable, red=declining)
3. **Changelog Table**: Sortable audit log with parameter, old-new, reason, date
4. **Detail Modal**: Show AI proposal + applied changes for each snapshot week
