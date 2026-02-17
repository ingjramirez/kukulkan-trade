-- Phase 38: Weekly Self-Improvement Loop
-- Adds improvement_snapshots, parameter_changelog tables and trailing_stop_multiplier to tenants.

CREATE TABLE IF NOT EXISTS improvement_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    week_start DATE NOT NULL,
    week_end DATE NOT NULL,
    total_trades INTEGER NOT NULL DEFAULT 0,
    win_rate_pct REAL,
    avg_pnl_pct REAL,
    avg_alpha_vs_spy REAL,
    total_cost_usd REAL DEFAULT 0.0,
    strategy_mode VARCHAR(20),
    trailing_stop_multiplier REAL DEFAULT 1.0,
    proposal_json TEXT,
    applied_changes TEXT,
    report_text TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS parameter_changelog (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    snapshot_id INTEGER REFERENCES improvement_snapshots(id) ON DELETE SET NULL,
    parameter VARCHAR(50) NOT NULL,
    old_value TEXT,
    new_value TEXT,
    reason TEXT,
    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE tenants ADD COLUMN trailing_stop_multiplier REAL NOT NULL DEFAULT 1.0;
