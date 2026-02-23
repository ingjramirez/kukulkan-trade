-- Migration 020: Drop stale cost columns from agent_conversations
-- These columns were never written in production after Phase 49 (Claude Code CLI migration).

ALTER TABLE agent_conversations DROP COLUMN cost_usd;
ALTER TABLE agent_conversations DROP COLUMN token_count;
