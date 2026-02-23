-- Migration 018: Post-rearchitecture cleanup
-- Phase 49 replaced Anthropic SDK with Claude Code CLI (Max subscription).
-- Drop stale flags, unused table, and adapt budget log for CLI metrics.

-- Drop stale TenantRow columns (old agent flags + unused API key)
ALTER TABLE tenants DROP COLUMN use_agent_loop;
ALTER TABLE tenants DROP COLUMN use_persistent_agent;
ALTER TABLE tenants DROP COLUMN use_tiered_models;
ALTER TABLE tenants DROP COLUMN claude_api_key_enc;

-- Drop stale AgentBudgetLogRow column (tiered model concept)
ALTER TABLE agent_budget_log DROP COLUMN session_profile;

-- Add CLI tracking columns to agent_budget_log
ALTER TABLE agent_budget_log ADD COLUMN num_turns INTEGER NOT NULL DEFAULT 0;
ALTER TABLE agent_budget_log ADD COLUMN tool_calls INTEGER NOT NULL DEFAULT 0;
ALTER TABLE agent_budget_log ADD COLUMN duration_ms INTEGER NOT NULL DEFAULT 0;

-- Drop unused WeeklyReportRow table (replaced by ImprovementSnapshotRow)
DROP TABLE IF EXISTS weekly_reports;
