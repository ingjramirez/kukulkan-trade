-- Phase 32: Agent loop support
-- Adds use_agent_loop flag to tenants and tool_call_logs table

ALTER TABLE tenants ADD COLUMN use_agent_loop BOOLEAN NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS tool_call_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id VARCHAR(36) NOT NULL DEFAULT 'default',
    session_date DATE NOT NULL,
    session_label VARCHAR(20),
    turn INTEGER NOT NULL,
    tool_name VARCHAR(50) NOT NULL,
    tool_input TEXT,
    tool_output_preview TEXT,
    success BOOLEAN NOT NULL DEFAULT 1,
    error TEXT,
    created_at DATETIME NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_tool_calls_tenant_date ON tool_call_logs(tenant_id, session_date);
