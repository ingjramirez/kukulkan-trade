-- PostgreSQL baseline migration: full schema for Kukulkan Trade
-- Equivalent to SQLite migrations 001-020 consolidated into PG-native DDL.
-- Tables are in FK dependency order: tenants first, then everything else.

BEGIN;

-- ── Schema migrations tracking ──────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT NOT NULL PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Tenants (root table, all FK refs point here) ────────────────────────────

CREATE TABLE IF NOT EXISTS tenants (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    -- Alpaca credentials (Fernet-encrypted)
    alpaca_api_key_enc TEXT,
    alpaca_api_secret_enc TEXT,
    alpaca_base_url VARCHAR(200) NOT NULL DEFAULT 'https://paper-api.alpaca.markets',

    -- Telegram credentials (Fernet-encrypted)
    telegram_bot_token_enc TEXT,
    telegram_chat_id_enc TEXT,

    -- Dashboard credentials
    dashboard_user VARCHAR(100),
    dashboard_password_enc TEXT,

    -- Strategy
    strategy_mode VARCHAR(20) NOT NULL DEFAULT 'conservative',

    -- Portfolio config
    run_portfolio_a BOOLEAN NOT NULL DEFAULT FALSE,
    run_portfolio_b BOOLEAN NOT NULL DEFAULT TRUE,
    portfolio_a_cash DOUBLE PRECISION NOT NULL DEFAULT 33000.0,
    portfolio_b_cash DOUBLE PRECISION NOT NULL DEFAULT 66000.0,

    -- Equity-based allocation
    initial_equity DOUBLE PRECISION,
    portfolio_a_pct DOUBLE PRECISION NOT NULL DEFAULT 33.33,
    portfolio_b_pct DOUBLE PRECISION NOT NULL DEFAULT 66.67,

    -- Rebalance flag
    pending_rebalance BOOLEAN NOT NULL DEFAULT FALSE,

    -- Claude Code CLI
    use_claude_code BOOLEAN NOT NULL DEFAULT FALSE,

    -- Trailing stop multiplier
    trailing_stop_multiplier DOUBLE PRECISION NOT NULL DEFAULT 1.0,

    -- Quiet hours
    quiet_hours_start VARCHAR(5) NOT NULL DEFAULT '21:00',
    quiet_hours_end VARCHAR(5) NOT NULL DEFAULT '07:00',
    quiet_hours_timezone VARCHAR(40) NOT NULL DEFAULT 'America/Mexico_City',

    -- Ticker customization (JSON arrays)
    ticker_whitelist TEXT,
    ticker_additions TEXT,
    ticker_exclusions TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Seed default tenant
INSERT INTO tenants (id, name) VALUES ('default', 'Default') ON CONFLICT DO NOTHING;

-- ── Portfolios ──────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS portfolios (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name VARCHAR(1) NOT NULL,
    cash DOUBLE PRECISION NOT NULL DEFAULT 33000.0,
    total_value DOUBLE PRECISION NOT NULL DEFAULT 33000.0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (tenant_id, name)
);

-- ── Positions ───────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    portfolio VARCHAR(1) NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    shares DOUBLE PRECISION NOT NULL,
    avg_price DOUBLE PRECISION NOT NULL,
    current_price DOUBLE PRECISION,
    market_value DOUBLE PRECISION,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (tenant_id, portfolio, ticker)
);

-- ── Trades ──────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    portfolio VARCHAR(1) NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    side VARCHAR(4) NOT NULL,
    shares DOUBLE PRECISION NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    total DOUBLE PRECISION NOT NULL,
    reason TEXT,
    executed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Daily Snapshots ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS daily_snapshots (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    portfolio VARCHAR(1) NOT NULL,
    date DATE NOT NULL,
    total_value DOUBLE PRECISION NOT NULL,
    cash DOUBLE PRECISION NOT NULL,
    positions_value DOUBLE PRECISION NOT NULL,
    daily_return_pct DOUBLE PRECISION,
    cumulative_return_pct DOUBLE PRECISION,
    UNIQUE (tenant_id, portfolio, date)
);

-- ── Momentum Rankings ───────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS momentum_rankings (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    return_63d DOUBLE PRECISION NOT NULL,
    rank INTEGER NOT NULL,
    UNIQUE (date, ticker)
);

-- ── Agent Decisions ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS agent_decisions (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    prompt_summary TEXT,
    response_summary TEXT,
    proposed_trades TEXT,
    reasoning TEXT,
    model_used VARCHAR(50),
    tokens_used INTEGER,
    regime VARCHAR(30),
    session_label VARCHAR(20),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Market Data ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    UNIQUE (ticker, date)
);

-- ── Technical Indicators ────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS technical_indicators (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    rsi_14 DOUBLE PRECISION,
    macd DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    macd_hist DOUBLE PRECISION,
    sma_20 DOUBLE PRECISION,
    sma_50 DOUBLE PRECISION,
    sma_200 DOUBLE PRECISION,
    bb_upper DOUBLE PRECISION,
    bb_middle DOUBLE PRECISION,
    bb_lower DOUBLE PRECISION,
    UNIQUE (ticker, date)
);

-- ── Macro Data ──────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS macro_data (
    id SERIAL PRIMARY KEY,
    indicator VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    UNIQUE (indicator, date)
);

-- ── News Log ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS news_log (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    headline TEXT NOT NULL,
    source VARCHAR(100),
    url TEXT,
    published_at TIMESTAMPTZ,
    sentiment DOUBLE PRECISION,
    embedding_id VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Discovered Tickers ──────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS discovered_tickers (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    ticker VARCHAR(10) NOT NULL,
    source VARCHAR(20) NOT NULL,
    rationale TEXT,
    status VARCHAR(10) NOT NULL DEFAULT 'proposed',
    proposed_at DATE NOT NULL,
    expires_at DATE NOT NULL,
    sector VARCHAR(50),
    market_cap DOUBLE PRECISION,
    UNIQUE (tenant_id, ticker)
);

-- ── Agent Memory ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS agent_memory (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    category VARCHAR(20) NOT NULL,
    key VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    UNIQUE (tenant_id, category, key)
);

-- ── Trailing Stops ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS trailing_stops (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    portfolio VARCHAR(1) NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    peak_price DOUBLE PRECISION NOT NULL,
    trail_pct DOUBLE PRECISION NOT NULL,
    stop_price DOUBLE PRECISION NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (tenant_id, portfolio, ticker)
);

-- ── Earnings Calendar ───────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS earnings_calendar (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    earnings_date DATE NOT NULL,
    source VARCHAR(20) DEFAULT 'yfinance',
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (ticker, earnings_date)
);

-- ── Watchlist ───────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS watchlist (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    portfolio VARCHAR(1) NOT NULL DEFAULT 'B',
    ticker VARCHAR(10) NOT NULL,
    reason TEXT,
    conviction VARCHAR(10) NOT NULL DEFAULT 'medium',
    target_entry DOUBLE PRECISION,
    added_date DATE NOT NULL,
    expires_at DATE NOT NULL,
    UNIQUE (tenant_id, ticker)
);

-- ── Intraday Snapshots ──────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS intraday_snapshots (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    portfolio VARCHAR(1) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    total_value DOUBLE PRECISION NOT NULL,
    cash DOUBLE PRECISION NOT NULL,
    positions_value DOUBLE PRECISION NOT NULL,
    is_extended_hours BOOLEAN NOT NULL DEFAULT FALSE,
    market_phase VARCHAR(20) NOT NULL DEFAULT 'market',
    UNIQUE (tenant_id, portfolio, timestamp)
);

-- ── Sentinel Actions ────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS sentinel_actions (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    action_type VARCHAR(20) NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    reason TEXT NOT NULL,
    source VARCHAR(30) NOT NULL,
    alert_level VARCHAR(10) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    resolved_at TIMESTAMPTZ,
    resolved_by VARCHAR(20)
);

-- ── Tool Call Logs ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS tool_call_logs (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    session_date DATE NOT NULL,
    session_label VARCHAR(20),
    turn INTEGER NOT NULL,
    tool_name VARCHAR(50) NOT NULL,
    tool_input TEXT,
    tool_output_preview TEXT,
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error TEXT,
    influenced_decision BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Agent Conversations ─────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS agent_conversations (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL UNIQUE,
    trigger_type VARCHAR(20) NOT NULL,
    messages_json TEXT NOT NULL,
    summary TEXT,
    session_status VARCHAR(20) NOT NULL DEFAULT 'completed',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Posture History ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS posture_history (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    session_date DATE NOT NULL,
    session_label VARCHAR(20),
    posture VARCHAR(20) NOT NULL,
    effective_posture VARCHAR(20) NOT NULL,
    reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Playbook Snapshots ──────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS playbook_snapshots (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    regime VARCHAR(30) NOT NULL,
    sector VARCHAR(50) NOT NULL,
    total_trades INTEGER NOT NULL,
    wins INTEGER NOT NULL,
    losses INTEGER NOT NULL,
    win_rate_pct DOUBLE PRECISION NOT NULL,
    avg_pnl_pct DOUBLE PRECISION NOT NULL,
    recommendation VARCHAR(30) NOT NULL
);

-- ── Conviction Calibration ──────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS conviction_calibration (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    conviction_level VARCHAR(10) NOT NULL,
    total_trades INTEGER NOT NULL,
    wins INTEGER NOT NULL,
    losses INTEGER NOT NULL,
    win_rate_pct DOUBLE PRECISION NOT NULL,
    avg_pnl_pct DOUBLE PRECISION NOT NULL,
    assessment VARCHAR(30) NOT NULL,
    suggested_multiplier DOUBLE PRECISION NOT NULL DEFAULT 1.0
);

-- ── Agent Budget Log ────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS agent_budget_log (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    session_date DATE NOT NULL,
    session_label VARCHAR(50) NOT NULL,
    session_id TEXT,
    num_turns INTEGER NOT NULL DEFAULT 0,
    tool_calls INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Improvement Snapshots ───────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS improvement_snapshots (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    week_start DATE NOT NULL,
    week_end DATE NOT NULL,
    total_trades INTEGER NOT NULL DEFAULT 0,
    win_rate_pct DOUBLE PRECISION,
    avg_pnl_pct DOUBLE PRECISION,
    avg_alpha_vs_spy DOUBLE PRECISION,
    total_cost_usd DOUBLE PRECISION DEFAULT 0.0,
    strategy_mode VARCHAR(20),
    trailing_stop_multiplier DOUBLE PRECISION DEFAULT 1.0,
    proposal_json TEXT,
    applied_changes TEXT,
    report_text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Parameter Changelog ─────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS parameter_changelog (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    snapshot_id INTEGER REFERENCES improvement_snapshots(id) ON DELETE SET NULL,
    parameter VARCHAR(50) NOT NULL,
    old_value TEXT,
    new_value TEXT,
    reason TEXT,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Sentiment Indicators ────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS sentiment_indicators (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name VARCHAR(50) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    classification VARCHAR(30) NOT NULL,
    sub_indicators TEXT,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sentiment_tenant_name
    ON sentiment_indicators (tenant_id, name, fetched_at);

-- ── Ticker Signals ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ticker_signals (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    ticker VARCHAR(10) NOT NULL,
    composite_score DOUBLE PRECISION NOT NULL,
    rank INTEGER NOT NULL,
    prev_rank INTEGER,
    rank_velocity DOUBLE PRECISION NOT NULL DEFAULT 0,
    momentum_20d DOUBLE PRECISION,
    momentum_63d DOUBLE PRECISION,
    rsi DOUBLE PRECISION,
    macd_histogram DOUBLE PRECISION,
    sma_trend_score DOUBLE PRECISION,
    bollinger_pct_b DOUBLE PRECISION,
    volume_ratio DOUBLE PRECISION,
    alerts VARCHAR(255) DEFAULT '[]',
    scored_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_signals_tenant_scored
    ON ticker_signals (tenant_id, scored_at);

CREATE INDEX IF NOT EXISTS idx_signals_tenant_ticker
    ON ticker_signals (tenant_id, ticker, scored_at);

-- Mark this migration as applied
INSERT INTO schema_migrations (version) VALUES ('001_baseline.sql') ON CONFLICT DO NOTHING;

COMMIT;
