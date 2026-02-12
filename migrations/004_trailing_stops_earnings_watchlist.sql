-- Phase 27: Trailing stops, earnings calendar, and dynamic watchlist tables.

CREATE TABLE IF NOT EXISTS trailing_stops (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id VARCHAR(36) NOT NULL DEFAULT 'default',
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

CREATE TABLE IF NOT EXISTS earnings_calendar (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker VARCHAR(10) NOT NULL,
    earnings_date DATE NOT NULL,
    source VARCHAR(20) DEFAULT 'yfinance',
    fetched_at DATETIME NOT NULL,
    UNIQUE(ticker, earnings_date)
);

CREATE TABLE IF NOT EXISTS watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id VARCHAR(36) NOT NULL DEFAULT 'default',
    portfolio VARCHAR(1) NOT NULL DEFAULT 'B',
    ticker VARCHAR(10) NOT NULL,
    reason TEXT,
    conviction VARCHAR(10) NOT NULL DEFAULT 'medium',
    target_entry FLOAT,
    added_date DATE NOT NULL,
    expires_at DATE NOT NULL,
    UNIQUE(tenant_id, ticker)
);
