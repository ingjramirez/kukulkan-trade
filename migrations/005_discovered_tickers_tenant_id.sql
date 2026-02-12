-- Migration 005: Add tenant_id to discovered_tickers
-- SQLite cannot ALTER UNIQUE constraints, so we recreate the table.

CREATE TABLE discovered_tickers_new (
    id INTEGER PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL DEFAULT 'default',
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
SELECT id, 'default', ticker, source, rationale, status, proposed_at, expires_at, sector, market_cap
FROM discovered_tickers;

DROP TABLE discovered_tickers;

ALTER TABLE discovered_tickers_new RENAME TO discovered_tickers;
