# Frontend API — Phase 33.2: Agent Insights

## New Endpoints

### GET /api/agent/posture
Returns posture history for the authenticated tenant.

**Auth:** Bearer token (tenant-scoped)

**Response:** `PostureHistoryResponse[]`
```json
[
  {
    "session_date": "2026-02-15",
    "session_label": "Morning",
    "posture": "defensive",
    "effective_posture": "defensive",
    "reason": "High VIX, uncertain regime transition",
    "created_at": "2026-02-15T15:00:00Z"
  }
]
```

### GET /api/agent/playbook
Returns the latest empirical playbook snapshot (regime×sector win rates).

**Auth:** Bearer token (tenant-scoped)

**Response:** `PlaybookCellResponse[]`
```json
[
  {
    "regime": "BULL",
    "sector": "Technology",
    "total_trades": 15,
    "wins": 11,
    "losses": 3,
    "win_rate_pct": 78.6,
    "avg_pnl_pct": 3.42,
    "recommendation": "sweet_spot"
  }
]
```

Recommendations: `sweet_spot` (>65% WR), `solid` (>55%), `avoid` (<45%), `neutral`, `insufficient_data` (<10 trades)

### GET /api/agent/calibration
Returns the latest conviction calibration snapshot.

**Auth:** Bearer token (tenant-scoped)

**Response:** `ConvictionCalibrationResponse[]`
```json
[
  {
    "conviction_level": "high",
    "total_trades": 20,
    "wins": 14,
    "losses": 4,
    "win_rate_pct": 77.8,
    "avg_pnl_pct": 4.21,
    "assessment": "validated",
    "suggested_multiplier": 1.2
  }
]
```

Assessments: `validated`, `overconfident`, `underconfident`, `neutral`, `insufficient` (<15 trades)

## Telegram Enhancements

The daily brief now shows posture after the investigation line:
```
  🤖 Investigation: 5 tools across 3 turns | $0.12
  🎯 Posture: Defensive
```

## Posture Levels

| Posture | Max Position | Max Sector | Max Equity |
|---------|-------------|------------|------------|
| Balanced | 35% | 50% | 80% |
| Defensive | 25% | 35% | 50% |
| Crisis | 15% | 25% | 30% |
| Aggressive* | 35% | 50% | 95% |

*Aggressive requires: 50+ trades, >55% win rate, positive alpha vs SPY
