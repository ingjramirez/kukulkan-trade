# Frontend API — Phase 4: Compute Optimization

## New Endpoint

### `GET /api/agent/budget`

Returns the current daily/monthly agent budget status for the authenticated tenant.

**Auth:** JWT required

**Response:**
```json
{
  "daily_spent": 1.25,
  "daily_limit": 3.0,
  "daily_remaining": 1.75,
  "monthly_spent": 42.50,
  "monthly_limit": 75.0,
  "monthly_remaining": 32.50,
  "daily_exhausted": false,
  "monthly_exhausted": false,
  "haiku_only": false
}
```

**Fields:**
- `daily_spent` / `monthly_spent` — Total agent cost (USD) for today / this month
- `daily_limit` / `monthly_limit` — Configured spend caps
- `daily_remaining` / `monthly_remaining` — Budget left (never negative)
- `daily_exhausted` — True when daily spend >= daily limit (sessions will be skipped)
- `monthly_exhausted` — True when monthly spend >= monthly limit
- `haiku_only` — True when monthly spend >= 80% of limit (non-FULL sessions use Haiku only)

## Architecture: Tiered Models

When `enable_tiered=True` (settings) and `use_tiered_models=True` (tenant flag):

1. **Haiku Scanner** — Fast ($0.002/scan), classifies session as ROUTINE/INVESTIGATE/URGENT
2. **Sonnet Investigator** — Standard tool-use loop ($0.05-0.50/session)
3. **Opus Validator** — Reviews proposed trades ($0.02/validation), only when trades exist

### Session Profiles

| Profile | Trigger | Flow |
|---------|---------|------|
| FULL | Morning open | Haiku → Sonnet → Opus (if trades) |
| LIGHT | Midday/Close | Haiku → skip if ROUTINE, else full |
| CRISIS | Event alert | Skip scan → full Sonnet |
| REVIEW | Weekly review | Skip scan + validation → Sonnet only |
| BUDGET_SAVING | Budget exhausted | Haiku scan only, no trades |

### Cost Savings

- LIGHT session with ROUTINE scan: ~$0.002 (vs ~$0.50 for full Sonnet)
- Prompt caching: ~90% input token savings on turns 2+ within a session
- Daily cap ($3) prevents runaway costs
- Monthly cap ($75) with 80% degradation threshold

## Existing Endpoints (Unchanged)

- `GET /api/agent/posture` — Posture history
- `GET /api/agent/playbook` — Empirical playbook
- `GET /api/agent/calibration` — Conviction calibration

## Tool Summary in Telegram

When tiered mode is active, the daily brief includes:
- Session profile (FULL/LIGHT/etc)
- Investigation details (tools, turns, cost)
- Posture declaration
