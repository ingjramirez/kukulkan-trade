-- Migration 002: Add missing indexes on PG hot-path columns
-- Covers tenant_id, date, ticker, published_at lookups that degrade under load.

BEGIN;

-- trades: history queries filter by tenant + portfolio + date
CREATE INDEX IF NOT EXISTS idx_trades_tenant_portfolio
    ON trades (tenant_id, portfolio, executed_at DESC);

CREATE INDEX IF NOT EXISTS idx_trades_tenant_ticker
    ON trades (tenant_id, ticker);

-- positions: get_positions() filters by tenant + portfolio
CREATE INDEX IF NOT EXISTS idx_positions_tenant_portfolio
    ON positions (tenant_id, portfolio);

-- daily_snapshots: snapshot lookups by tenant + date + portfolio
CREATE INDEX IF NOT EXISTS idx_daily_snapshots_tenant_date
    ON daily_snapshots (tenant_id, date DESC, portfolio);

-- agent_decisions: decision history by tenant + date
CREATE INDEX IF NOT EXISTS idx_agent_decisions_tenant_date
    ON agent_decisions (tenant_id, date DESC);

-- news_log: news queries filter by ticker and/or published_at
CREATE INDEX IF NOT EXISTS idx_news_log_ticker_published
    ON news_log (ticker, published_at DESC);

CREATE INDEX IF NOT EXISTS idx_news_log_published
    ON news_log (published_at DESC);

-- market_data: price lookups by ticker + date
CREATE INDEX IF NOT EXISTS idx_market_data_ticker_date
    ON market_data (ticker, date DESC);

-- technical_indicators: same hot path
CREATE INDEX IF NOT EXISTS idx_technical_indicators_ticker_date
    ON technical_indicators (ticker, date DESC);

-- intraday_snapshots: delete + select by tenant + portfolio + timestamp
CREATE INDEX IF NOT EXISTS idx_intraday_snapshots_tenant_portfolio_ts
    ON intraday_snapshots (tenant_id, portfolio, timestamp DESC);

-- trailing_stops: lookup by tenant + portfolio + ticker
CREATE INDEX IF NOT EXISTS idx_trailing_stops_tenant_portfolio
    ON trailing_stops (tenant_id, portfolio, ticker);

-- watchlist: filtered by tenant
CREATE INDEX IF NOT EXISTS idx_watchlist_tenant
    ON watchlist (tenant_id);

-- earnings_calendar: lookup by ticker + earnings_date
CREATE INDEX IF NOT EXISTS idx_earnings_calendar_ticker_date
    ON earnings_calendar (ticker, earnings_date DESC);

-- agent_memory: lookup by tenant + key (upsert pattern)
CREATE INDEX IF NOT EXISTS idx_agent_memory_tenant_key
    ON agent_memory (tenant_id, key);

-- momentum_rankings: lookup by date
CREATE INDEX IF NOT EXISTS idx_momentum_rankings_date
    ON momentum_rankings (date DESC);

-- tool_call_logs: lookup by session_id
CREATE INDEX IF NOT EXISTS idx_tool_call_logs_session
    ON tool_call_logs (session_id);

-- sentinel_actions: lookup by tenant + created_at
CREATE INDEX IF NOT EXISTS idx_sentinel_actions_tenant_created
    ON sentinel_actions (tenant_id, created_at DESC);

-- Mark migration applied
INSERT INTO schema_migrations (version) VALUES ('002_add_indexes.sql') ON CONFLICT DO NOTHING;

COMMIT;
