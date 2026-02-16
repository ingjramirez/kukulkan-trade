-- Migration 012: Add foreign key constraints on tenant_id → tenants(id)
-- SQLite cannot ALTER constraints, so we recreate each table.

-- Ensure default tenant exists as FK anchor for orphaned rows
INSERT OR IGNORE INTO tenants (id, name, is_active, strategy_mode, run_portfolio_a, run_portfolio_b, portfolio_a_cash, portfolio_b_cash, portfolio_a_pct, portfolio_b_pct, pending_rebalance, use_agent_loop, use_persistent_agent, use_tiered_models)
VALUES ('default', 'Default', 1, 'conservative', 0, 1, 33000.0, 66000.0, 33.33, 66.67, 0, 0, 0, 0);

-- Disable FK enforcement during migration (table drops would violate constraints)
PRAGMA foreign_keys = OFF;

-- 1. portfolios
CREATE TABLE portfolios_new (
    id INTEGER PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL DEFAULT 'default' REFERENCES tenants(id) ON DELETE CASCADE,
    name VARCHAR(1) NOT NULL,
    cash FLOAT NOT NULL DEFAULT 33000.0,
    total_value FLOAT NOT NULL DEFAULT 33000.0,
    updated_at DATETIME NOT NULL,
    UNIQUE(tenant_id, name)
);
INSERT INTO portfolios_new (id, tenant_id, name, cash, total_value, updated_at)
SELECT id, tenant_id, name, cash, total_value, updated_at FROM portfolios;
DROP TABLE portfolios;
ALTER TABLE portfolios_new RENAME TO portfolios;

-- 2. positions
CREATE TABLE positions_new (
    id INTEGER PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL DEFAULT 'default' REFERENCES tenants(id) ON DELETE CASCADE,
    portfolio VARCHAR(1) NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    shares FLOAT NOT NULL,
    avg_price FLOAT NOT NULL,
    current_price FLOAT,
    market_value FLOAT,
    updated_at DATETIME NOT NULL,
    UNIQUE(tenant_id, portfolio, ticker)
);
INSERT INTO positions_new (id, tenant_id, portfolio, ticker, shares, avg_price, current_price, market_value, updated_at)
SELECT id, tenant_id, portfolio, ticker, shares, avg_price, current_price, market_value, updated_at FROM positions;
DROP TABLE positions;
ALTER TABLE positions_new RENAME TO positions;

-- 3. trades
CREATE TABLE trades_new (
    id INTEGER PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL DEFAULT 'default' REFERENCES tenants(id) ON DELETE CASCADE,
    portfolio VARCHAR(1) NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    side VARCHAR(4) NOT NULL,
    shares FLOAT NOT NULL,
    price FLOAT NOT NULL,
    total FLOAT NOT NULL,
    reason TEXT,
    executed_at DATETIME NOT NULL
);
INSERT INTO trades_new (id, tenant_id, portfolio, ticker, side, shares, price, total, reason, executed_at)
SELECT id, tenant_id, portfolio, ticker, side, shares, price, total, reason, executed_at FROM trades;
DROP TABLE trades;
ALTER TABLE trades_new RENAME TO trades;

-- 4. daily_snapshots
CREATE TABLE daily_snapshots_new (
    id INTEGER PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL DEFAULT 'default' REFERENCES tenants(id) ON DELETE CASCADE,
    portfolio VARCHAR(1) NOT NULL,
    date DATE NOT NULL,
    total_value FLOAT NOT NULL,
    cash FLOAT NOT NULL,
    positions_value FLOAT NOT NULL,
    daily_return_pct FLOAT,
    cumulative_return_pct FLOAT,
    UNIQUE(tenant_id, portfolio, date)
);
INSERT INTO daily_snapshots_new (id, tenant_id, portfolio, date, total_value, cash, positions_value, daily_return_pct, cumulative_return_pct)
SELECT id, tenant_id, portfolio, date, total_value, cash, positions_value, daily_return_pct, cumulative_return_pct FROM daily_snapshots;
DROP TABLE daily_snapshots;
ALTER TABLE daily_snapshots_new RENAME TO daily_snapshots;

-- 5. agent_decisions
CREATE TABLE agent_decisions_new (
    id INTEGER PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL DEFAULT 'default' REFERENCES tenants(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    prompt_summary TEXT,
    response_summary TEXT,
    proposed_trades TEXT,
    reasoning TEXT,
    model_used VARCHAR(50),
    tokens_used INTEGER,
    regime VARCHAR(30),
    session_label VARCHAR(20),
    created_at DATETIME NOT NULL
);
INSERT INTO agent_decisions_new (id, tenant_id, date, prompt_summary, response_summary, proposed_trades, reasoning, model_used, tokens_used, regime, session_label, created_at)
SELECT id, tenant_id, date, prompt_summary, response_summary, proposed_trades, reasoning, model_used, tokens_used, regime, session_label, created_at FROM agent_decisions;
DROP TABLE agent_decisions;
ALTER TABLE agent_decisions_new RENAME TO agent_decisions;

-- 6. discovered_tickers
CREATE TABLE discovered_tickers_new (
    id INTEGER PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL DEFAULT 'default' REFERENCES tenants(id) ON DELETE CASCADE,
    ticker VARCHAR(10) NOT NULL,
    source VARCHAR(20) NOT NULL,
    rationale TEXT,
    status VARCHAR(10) NOT NULL DEFAULT 'proposed',
    proposed_at DATE NOT NULL,
    expires_at DATE NOT NULL,
    sector VARCHAR(50),
    market_cap FLOAT,
    UNIQUE(tenant_id, ticker)
);
INSERT INTO discovered_tickers_new (id, tenant_id, ticker, source, rationale, status, proposed_at, expires_at, sector, market_cap)
SELECT id, tenant_id, ticker, source, rationale, status, proposed_at, expires_at, sector, market_cap FROM discovered_tickers;
DROP TABLE discovered_tickers;
ALTER TABLE discovered_tickers_new RENAME TO discovered_tickers;

-- 7. agent_memory
CREATE TABLE agent_memory_new (
    id INTEGER PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL DEFAULT 'default' REFERENCES tenants(id) ON DELETE CASCADE,
    category VARCHAR(20) NOT NULL,
    key VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    created_at DATETIME NOT NULL,
    expires_at DATETIME,
    UNIQUE(tenant_id, category, key)
);
INSERT INTO agent_memory_new (id, tenant_id, category, key, content, created_at, expires_at)
SELECT id, tenant_id, category, key, content, created_at, expires_at FROM agent_memory;
DROP TABLE agent_memory;
ALTER TABLE agent_memory_new RENAME TO agent_memory;

-- 8. trailing_stops
CREATE TABLE trailing_stops_new (
    id INTEGER PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL DEFAULT 'default' REFERENCES tenants(id) ON DELETE CASCADE,
    portfolio VARCHAR(1) NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    entry_price FLOAT NOT NULL,
    peak_price FLOAT NOT NULL,
    trail_pct FLOAT NOT NULL,
    stop_price FLOAT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT 1,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    UNIQUE(tenant_id, portfolio, ticker)
);
INSERT INTO trailing_stops_new (id, tenant_id, portfolio, ticker, entry_price, peak_price, trail_pct, stop_price, is_active, created_at, updated_at)
SELECT id, tenant_id, portfolio, ticker, entry_price, peak_price, trail_pct, stop_price, is_active, created_at, updated_at FROM trailing_stops;
DROP TABLE trailing_stops;
ALTER TABLE trailing_stops_new RENAME TO trailing_stops;

-- 9. watchlist
CREATE TABLE watchlist_new (
    id INTEGER PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL DEFAULT 'default' REFERENCES tenants(id) ON DELETE CASCADE,
    portfolio VARCHAR(1) NOT NULL DEFAULT 'B',
    ticker VARCHAR(10) NOT NULL,
    reason TEXT,
    conviction VARCHAR(10) NOT NULL DEFAULT 'medium',
    target_entry FLOAT,
    added_date DATE NOT NULL,
    expires_at DATE NOT NULL,
    UNIQUE(tenant_id, ticker)
);
INSERT INTO watchlist_new (id, tenant_id, portfolio, ticker, reason, conviction, target_entry, added_date, expires_at)
SELECT id, tenant_id, portfolio, ticker, reason, conviction, target_entry, added_date, expires_at FROM watchlist;
DROP TABLE watchlist;
ALTER TABLE watchlist_new RENAME TO watchlist;

-- 10. intraday_snapshots
CREATE TABLE intraday_snapshots_new (
    id INTEGER PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL DEFAULT 'default' REFERENCES tenants(id) ON DELETE CASCADE,
    portfolio VARCHAR(1) NOT NULL,
    timestamp DATETIME NOT NULL,
    total_value FLOAT NOT NULL,
    cash FLOAT NOT NULL,
    positions_value FLOAT NOT NULL,
    UNIQUE(tenant_id, portfolio, timestamp)
);
INSERT INTO intraday_snapshots_new (id, tenant_id, portfolio, timestamp, total_value, cash, positions_value)
SELECT id, tenant_id, portfolio, timestamp, total_value, cash, positions_value FROM intraday_snapshots;
DROP TABLE intraday_snapshots;
ALTER TABLE intraday_snapshots_new RENAME TO intraday_snapshots;

-- 11. tool_call_logs
CREATE TABLE tool_call_logs_new (
    id INTEGER PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL DEFAULT 'default' REFERENCES tenants(id) ON DELETE CASCADE,
    session_date DATE NOT NULL,
    session_label VARCHAR(20),
    turn INTEGER NOT NULL,
    tool_name VARCHAR(50) NOT NULL,
    tool_input TEXT,
    tool_output_preview TEXT,
    success BOOLEAN NOT NULL DEFAULT 1,
    error TEXT,
    influenced_decision BOOLEAN NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL
);
INSERT INTO tool_call_logs_new (id, tenant_id, session_date, session_label, turn, tool_name, tool_input, tool_output_preview, success, error, influenced_decision, created_at)
SELECT id, tenant_id, session_date, session_label, turn, tool_name, tool_input, tool_output_preview, success, error, influenced_decision, created_at FROM tool_call_logs;
DROP TABLE tool_call_logs;
ALTER TABLE tool_call_logs_new RENAME TO tool_call_logs;

-- 12. agent_conversations
CREATE TABLE agent_conversations_new (
    id INTEGER PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL UNIQUE,
    trigger_type VARCHAR(20) NOT NULL,
    messages_json TEXT NOT NULL,
    summary TEXT,
    token_count INTEGER NOT NULL DEFAULT 0,
    cost_usd FLOAT NOT NULL DEFAULT 0.0,
    session_status VARCHAR(20) NOT NULL DEFAULT 'completed',
    created_at DATETIME NOT NULL
);
INSERT INTO agent_conversations_new (id, tenant_id, session_id, trigger_type, messages_json, summary, token_count, cost_usd, session_status, created_at)
SELECT id, tenant_id, session_id, trigger_type, messages_json, summary, token_count, cost_usd, session_status, created_at FROM agent_conversations;
DROP TABLE agent_conversations;
ALTER TABLE agent_conversations_new RENAME TO agent_conversations;

-- 13. posture_history
CREATE TABLE posture_history_new (
    id INTEGER PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL DEFAULT 'default' REFERENCES tenants(id) ON DELETE CASCADE,
    session_date DATE NOT NULL,
    session_label VARCHAR(20),
    posture VARCHAR(20) NOT NULL,
    effective_posture VARCHAR(20) NOT NULL,
    reason TEXT,
    created_at DATETIME NOT NULL
);
INSERT INTO posture_history_new (id, tenant_id, session_date, session_label, posture, effective_posture, reason, created_at)
SELECT id, tenant_id, session_date, session_label, posture, effective_posture, reason, created_at FROM posture_history;
DROP TABLE posture_history;
ALTER TABLE posture_history_new RENAME TO posture_history;

-- 14. playbook_snapshots
CREATE TABLE playbook_snapshots_new (
    id INTEGER PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL DEFAULT 'default' REFERENCES tenants(id) ON DELETE CASCADE,
    generated_at DATETIME NOT NULL,
    regime VARCHAR(30) NOT NULL,
    sector VARCHAR(50) NOT NULL,
    total_trades INTEGER NOT NULL,
    wins INTEGER NOT NULL,
    losses INTEGER NOT NULL,
    win_rate_pct FLOAT NOT NULL,
    avg_pnl_pct FLOAT NOT NULL,
    recommendation VARCHAR(30) NOT NULL
);
INSERT INTO playbook_snapshots_new (id, tenant_id, generated_at, regime, sector, total_trades, wins, losses, win_rate_pct, avg_pnl_pct, recommendation)
SELECT id, tenant_id, generated_at, regime, sector, total_trades, wins, losses, win_rate_pct, avg_pnl_pct, recommendation FROM playbook_snapshots;
DROP TABLE playbook_snapshots;
ALTER TABLE playbook_snapshots_new RENAME TO playbook_snapshots;

-- 15. conviction_calibration
CREATE TABLE conviction_calibration_new (
    id INTEGER PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL DEFAULT 'default' REFERENCES tenants(id) ON DELETE CASCADE,
    generated_at DATETIME NOT NULL,
    conviction_level VARCHAR(10) NOT NULL,
    total_trades INTEGER NOT NULL,
    wins INTEGER NOT NULL,
    losses INTEGER NOT NULL,
    win_rate_pct FLOAT NOT NULL,
    avg_pnl_pct FLOAT NOT NULL,
    assessment VARCHAR(30) NOT NULL,
    suggested_multiplier FLOAT NOT NULL DEFAULT 1.0
);
INSERT INTO conviction_calibration_new (id, tenant_id, generated_at, conviction_level, total_trades, wins, losses, win_rate_pct, avg_pnl_pct, assessment, suggested_multiplier)
SELECT id, tenant_id, generated_at, conviction_level, total_trades, wins, losses, win_rate_pct, avg_pnl_pct, assessment, suggested_multiplier FROM conviction_calibration;
DROP TABLE conviction_calibration;
ALTER TABLE conviction_calibration_new RENAME TO conviction_calibration;

-- 16. agent_budget_log
CREATE TABLE agent_budget_log_new (
    id INTEGER PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL DEFAULT 'default' REFERENCES tenants(id) ON DELETE CASCADE,
    session_date DATE NOT NULL,
    session_label VARCHAR(50) NOT NULL,
    session_id TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd FLOAT NOT NULL DEFAULT 0.0,
    session_profile VARCHAR(20),
    created_at DATETIME NOT NULL
);
INSERT INTO agent_budget_log_new (id, tenant_id, session_date, session_label, session_id, input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens, cost_usd, session_profile, created_at)
SELECT id, tenant_id, session_date, session_label, session_id, input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens, cost_usd, session_profile, created_at FROM agent_budget_log;
DROP TABLE agent_budget_log;
ALTER TABLE agent_budget_log_new RENAME TO agent_budget_log;

-- Re-enable FK enforcement
PRAGMA foreign_keys = ON;
