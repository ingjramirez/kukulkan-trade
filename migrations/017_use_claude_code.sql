-- Migration 017: Add use_claude_code flag to tenants
-- Routes Portfolio B through Claude Code CLI (Max subscription) instead of AgentRunner

ALTER TABLE tenants ADD COLUMN use_claude_code BOOLEAN NOT NULL DEFAULT 0;
