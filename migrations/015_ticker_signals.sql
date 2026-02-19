-- Ticker signal rankings (computed every 10 min by SignalEngine, pure local computation)
CREATE TABLE IF NOT EXISTS ticker_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id VARCHAR(36) NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    composite_score FLOAT NOT NULL,
    rank INTEGER NOT NULL,
    prev_rank INTEGER,
    rank_velocity FLOAT NOT NULL DEFAULT 0,
    momentum_20d FLOAT,
    momentum_63d FLOAT,
    rsi FLOAT,
    macd_histogram FLOAT,
    sma_trend_score FLOAT,
    bollinger_pct_b FLOAT,
    volume_ratio FLOAT,
    alerts TEXT DEFAULT '[]',
    scored_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE
);

CREATE INDEX idx_signals_tenant_scored ON ticker_signals(tenant_id, scored_at DESC);
CREATE INDEX idx_signals_tenant_ticker ON ticker_signals(tenant_id, ticker, scored_at DESC);
