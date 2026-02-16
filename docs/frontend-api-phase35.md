# Frontend API — Phase 35: Inverse ETF / Market Hedging

## New Endpoint

### `GET /api/agent/inverse-exposure`
Returns current inverse ETF exposure for Portfolio B.

**Auth:** Required (JWT)

**Response:**
```json
{
  "total_value": 3000.0,
  "total_pct": 3.0,
  "net_equity_pct": 87.0,
  "positions": [
    {
      "ticker": "SH",
      "value": 3000.0,
      "pct": 3.0,
      "equity_hedge": true,
      "days_held": 2,
      "hold_alert": null
    }
  ],
  "rules": {
    "max_single_pct": 10.0,
    "max_total_pct": 15.0,
    "max_positions": 2
  }
}
```

**Fields:**
- `total_value` — Total inverse ETF position value
- `total_pct` — Inverse as % of Portfolio B
- `net_equity_pct` — Long equity minus equity hedges as % of portfolio
- `positions[].equity_hedge` — `true` for SH/PSQ/RWM, `false` for TBF
- `positions[].days_held` — Days since most recent BUY (null if no BUY found)
- `positions[].hold_alert` — `"warning"` (3-4d), `"review"` (5+d), or `null`
- `rules` — Current risk limits

## Changed Tool Responses

### `get_portfolio_state` — New field
Each position now includes `instrument_type`: `"stock"`, `"etf"`, `"inverse_etf"`, or `"crypto_proxy"`.

### `get_risk_assessment` — New field
Response now includes `inverse_exposure` dict with `total_value`, `total_pct`, `positions`, `net_equity_pct`.

### `get_batch_technicals` — New field
Each result entry now includes `instrument_type`.

## Inverse ETF Instruments
| Ticker | Description | Benchmark | Equity Hedge |
|--------|------------|-----------|-------------|
| SH | Short S&P 500 | SPY | Yes |
| PSQ | Short Nasdaq 100 | QQQ | Yes |
| RWM | Short Russell 2000 | IWM | Yes |
| TBF | Short 20+ Yr Treasury | TLT | No |
