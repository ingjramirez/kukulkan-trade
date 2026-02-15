# Frontend API Changes — Phase 33.1 (Upgraded Tool Kit)

## Overview

The agent tool kit expanded from 9 to 18 tools. This affects two frontend-facing surfaces:

1. **`GET /api/agent/tool-logs`** — `tool_name` field in `ToolCallLogResponse`
2. **`GET /api/agent/conversations/{session_id}`** — `name` field in `tool_use` content blocks

No new API endpoints were added. No schema changes. The only difference is **new tool name values** appearing in existing fields.

## Tool Name Reference

### Portfolio Tools (6 new + 3 legacy aliases)

| Tool Name | Parameters | Description |
|-----------|-----------|-------------|
| `get_portfolio_state` | _(none)_ | Full Portfolio B state: positions with P&L, trailing stops, cash, total value, sector exposure |
| `get_position_detail` | `ticker` (required) | Deep dive on one position: P&L, trailing stop status, recent trade history |
| `get_portfolio_performance` | `period` (optional: 7d/14d/30d/60d/90d) | Return, drawdown, daily return stats, trade counts for a period |
| `get_historical_trades` | `days` (optional: 1-90, default 30) | Past Portfolio B trade history |
| `get_correlation_check` | `tickers` (optional: string[]) | Pairwise correlations, high-correlation pairs (>0.7), diversification score |
| `get_risk_assessment` | _(none)_ | Sector concentration, position weights, stop distances, volatility estimate |
| `get_current_positions` | _(none)_ | **Legacy alias** → `get_portfolio_state` |
| `get_position_pnl` | `ticker` (required) | **Legacy alias** → `get_position_detail` |
| `get_portfolio_summary` | _(none)_ | **Legacy alias** → `get_portfolio_state` |

### Market Tools (4 new + 2 legacy aliases)

| Tool Name | Parameters | Description |
|-----------|-----------|-------------|
| `get_batch_technicals` | `tickers` (required: string[], max 20) | Bulk technical analysis: price, returns, RSI, MACD, SMA for multiple tickers |
| `get_sector_heatmap` | _(none)_ | All sector ETFs with 1d/5d/20d returns + RSI, sorted by 5d performance |
| `get_market_overview` | _(none)_ | SPY/VIX/yield curve snapshot + regime classification |
| `get_earnings_calendar` | `tickers` (optional), `days_ahead` (optional: 1-30) | Upcoming earnings dates from DB |
| `get_price_and_technicals` | `ticker` (required) | **Legacy alias** → `get_batch_technicals` (single ticker) |
| `get_market_context` | _(none)_ | **Legacy alias** → `get_market_overview` |

### News & Cross-Portfolio Tools (3 total)

| Tool Name | Parameters | Description |
|-----------|-----------|-------------|
| `search_news` | `ticker` (optional) | Filter today's pre-fetched news headlines by ticker |
| `search_historical_news` | `ticker` (required), `query` (optional), `n_results` (optional: 1-10) | ChromaDB semantic search for past articles |
| `get_portfolio_a_status` | _(none)_ | Read-only view of Portfolio A: held ETFs, momentum rankings, P&L |

### Action Tools (5 new + 2 legacy aliases)

| Tool Name | Parameters | Description |
|-----------|-----------|-------------|
| `execute_trade` | `ticker`, `side` (BUY/SELL), `shares` (required); `reason`, `conviction` (optional) | Submit a trade for execution (queued for risk check) |
| `set_trailing_stop` | `ticker`, `trail_pct` (required: 0.03-0.20); `reason` (optional) | Set or update trailing stop percentage for a position |
| `get_order_status` | `ticker` (optional) | Check pending session trades + recent fills from DB |
| `save_observation` | `key`, `content` (required) | Save an insight to persist across sessions |
| `update_watchlist` | `updates` (required: array of {action, ticker, reason?, conviction?, target_entry?}) | Add or remove tickers from watchlist |
| `propose_trades` | `trades` (required: array) | **Legacy alias** — old batch trade format |
| `save_memory_note` | `key`, `content` (required) | **Legacy alias** → `save_observation` |

## Backward Compatibility

Legacy tool names (`get_current_positions`, `get_position_pnl`, `get_portfolio_summary`, `get_price_and_technicals`, `get_market_context`, `propose_trades`, `save_memory_note`) are still registered as aliases. Claude may use either the new or legacy name in a given session.

**What this means for the frontend:**
- Historical tool-logs and conversations from **before** this update contain old tool names
- New sessions will **predominantly** use the new Phase 2 names
- Both old and new names can appear in the same response

## Suggested Frontend Updates

### Tool Name Display Mapping

If the frontend renders tool names with friendly labels or icons, add mappings for the new names:

```typescript
const TOOL_DISPLAY: Record<string, { label: string; icon: string; category: string }> = {
  // Portfolio
  get_portfolio_state:    { label: "Portfolio State",    icon: "briefcase",    category: "Portfolio" },
  get_position_detail:    { label: "Position Detail",    icon: "search",       category: "Portfolio" },
  get_portfolio_performance: { label: "Performance",     icon: "trending-up",  category: "Portfolio" },
  get_historical_trades:  { label: "Trade History",      icon: "clock",        category: "Portfolio" },
  get_correlation_check:  { label: "Correlation",        icon: "git-merge",    category: "Portfolio" },
  get_risk_assessment:    { label: "Risk Assessment",    icon: "shield",       category: "Portfolio" },

  // Market
  get_batch_technicals:   { label: "Technicals",         icon: "bar-chart-2",  category: "Market" },
  get_sector_heatmap:     { label: "Sector Heatmap",     icon: "grid",         category: "Market" },
  get_market_overview:    { label: "Market Overview",     icon: "globe",        category: "Market" },
  get_earnings_calendar:  { label: "Earnings Calendar",  icon: "calendar",     category: "Market" },

  // News
  search_news:            { label: "Search News",        icon: "newspaper",    category: "News" },
  search_historical_news: { label: "Historical News",    icon: "archive",      category: "News" },
  get_portfolio_a_status: { label: "Portfolio A",         icon: "eye",          category: "News" },

  // Actions
  execute_trade:          { label: "Execute Trade",      icon: "zap",          category: "Action" },
  set_trailing_stop:      { label: "Set Stop",           icon: "shield-off",   category: "Action" },
  get_order_status:       { label: "Order Status",       icon: "check-circle", category: "Action" },
  save_observation:       { label: "Save Observation",   icon: "bookmark",     category: "Action" },
  update_watchlist:       { label: "Update Watchlist",    icon: "star",         category: "Action" },

  // Legacy aliases (map to same display as their Phase 2 equivalents)
  get_current_positions:  { label: "Portfolio State",    icon: "briefcase",    category: "Portfolio" },
  get_position_pnl:       { label: "Position Detail",    icon: "search",       category: "Portfolio" },
  get_portfolio_summary:  { label: "Portfolio State",    icon: "briefcase",    category: "Portfolio" },
  get_price_and_technicals: { label: "Technicals",       icon: "bar-chart-2",  category: "Market" },
  get_market_context:     { label: "Market Overview",     icon: "globe",        category: "Market" },
  propose_trades:         { label: "Execute Trade",      icon: "zap",          category: "Action" },
  save_memory_note:       { label: "Save Observation",   icon: "bookmark",     category: "Action" },
};
```

### Tool Category Badges

In the tool-logs table or conversation viewer, group tools by category with colored badges:
- **Portfolio** (blue) — investigation tools
- **Market** (green) — market data tools
- **News** (orange) — news and cross-portfolio tools
- **Action** (red) — trade execution and state mutation tools

### Conversation Viewer Enhancements

When rendering `tool_use` blocks in conversation messages:
- Show the friendly label instead of the raw tool name
- Collapse `tool_result` blocks by default (can be large JSON)
- For `execute_trade` results, highlight the trade summary prominently
- For `get_sector_heatmap` results, consider rendering as a visual heatmap
