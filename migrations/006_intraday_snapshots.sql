-- Migration 006: Intraday snapshots table for per-portfolio high-frequency data
-- Stores values every 15 minutes during market hours for rich equity curves

CREATE TABLE IF NOT EXISTS intraday_snapshots (
    id INTEGER PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL DEFAULT 'default',
    portfolio VARCHAR(1) NOT NULL,
    timestamp DATETIME NOT NULL,
    total_value FLOAT NOT NULL,
    cash FLOAT NOT NULL,
    positions_value FLOAT NOT NULL,
    UNIQUE(tenant_id, portfolio, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_intraday_tenant_ts
    ON intraday_snapshots(tenant_id, timestamp);
