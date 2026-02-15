-- Migration 009: Persistent agent conversation table + tenant flag

CREATE TABLE IF NOT EXISTS agent_conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id VARCHAR(36) NOT NULL,
    session_id TEXT NOT NULL UNIQUE,
    trigger_type VARCHAR(20) NOT NULL,
    messages_json TEXT NOT NULL,
    summary TEXT,
    token_count INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    session_status VARCHAR(20) NOT NULL DEFAULT 'completed',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (tenant_id) REFERENCES tenants(id)
);

CREATE INDEX IF NOT EXISTS idx_agent_conv_tenant_created
    ON agent_conversations(tenant_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_conv_tenant_status
    ON agent_conversations(tenant_id, session_status);

-- Add persistent agent flag to tenants (separate from use_agent_loop)
ALTER TABLE tenants ADD COLUMN use_persistent_agent BOOLEAN NOT NULL DEFAULT 0;
