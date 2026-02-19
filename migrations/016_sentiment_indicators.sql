-- Migration 016: Sentiment indicators table (Fear & Greed Index, etc.)
CREATE TABLE IF NOT EXISTS sentiment_indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name VARCHAR(50) NOT NULL,
    value REAL NOT NULL,
    classification VARCHAR(30) NOT NULL,
    sub_indicators TEXT,
    fetched_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sentiment_tenant_name ON sentiment_indicators(tenant_id, name, fetched_at DESC);
