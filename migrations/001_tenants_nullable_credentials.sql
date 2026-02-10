-- Migration 001: Make tenant integration credentials nullable
-- Allows creating login-only tenants who configure Alpaca/Telegram later.
-- SQLite doesn't support ALTER COLUMN, so we recreate the table.

BEGIN TRANSACTION;

CREATE TABLE tenants_new (
    id VARCHAR(36) NOT NULL,
    name VARCHAR(100) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT 1,
    alpaca_api_key_enc TEXT,
    alpaca_api_secret_enc TEXT,
    alpaca_base_url VARCHAR(200) NOT NULL DEFAULT 'https://paper-api.alpaca.markets',
    telegram_bot_token_enc TEXT,
    telegram_chat_id_enc TEXT,
    claude_api_key_enc TEXT,
    strategy_mode VARCHAR(20) NOT NULL DEFAULT 'conservative',
    run_portfolio_a BOOLEAN NOT NULL DEFAULT 0,
    run_portfolio_b BOOLEAN NOT NULL DEFAULT 1,
    portfolio_a_cash FLOAT NOT NULL DEFAULT 33000.0,
    portfolio_b_cash FLOAT NOT NULL DEFAULT 66000.0,
    ticker_whitelist TEXT,
    ticker_additions TEXT,
    ticker_exclusions TEXT,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    dashboard_user VARCHAR(100),
    dashboard_password_enc TEXT,
    PRIMARY KEY (id)
);

INSERT INTO tenants_new SELECT * FROM tenants;

DROP TABLE tenants;

ALTER TABLE tenants_new RENAME TO tenants;

COMMIT;
