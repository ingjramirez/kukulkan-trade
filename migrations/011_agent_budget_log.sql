-- Phase 4: Agent budget log table for daily/monthly cost tracking

CREATE TABLE IF NOT EXISTS agent_budget_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    session_date DATE NOT NULL,
    session_label TEXT NOT NULL,
    session_id TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    session_profile TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Phase 4: Tiered model flag on tenants
ALTER TABLE tenants ADD COLUMN use_tiered_models INTEGER NOT NULL DEFAULT 0;
