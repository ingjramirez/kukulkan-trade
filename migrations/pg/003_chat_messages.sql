-- Migration 003: chat_messages table
-- Stores user↔agent chat history for the interactive chat feature.

CREATE TABLE IF NOT EXISTS chat_messages (
    id          SERIAL PRIMARY KEY,
    tenant_id   VARCHAR(36) NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    session_id  TEXT,                           -- Claude session_id (for correlation with trading sessions)
    role        VARCHAR(10) NOT NULL,           -- 'user' | 'assistant'
    content     TEXT NOT NULL,
    tool_calls_json TEXT,                       -- JSON array of tool call summaries (assistant only)
    created_at  TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT (NOW() AT TIME ZONE 'utc')
);

CREATE INDEX idx_chat_messages_tenant_created ON chat_messages (tenant_id, created_at);
CREATE INDEX idx_chat_messages_session ON chat_messages (tenant_id, session_id);
