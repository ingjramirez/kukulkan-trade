-- Migration 019: Drop stale cost/token columns from agent_budget_log
-- These columns are meaningless after switching to Claude Max subscription
-- (flat-rate, no per-token billing). cost_usd was always 0.0.
-- Token columns (input/output/cache) were from the old Anthropic SDK path.

ALTER TABLE agent_budget_log DROP COLUMN cost_usd;
ALTER TABLE agent_budget_log DROP COLUMN input_tokens;
ALTER TABLE agent_budget_log DROP COLUMN output_tokens;
ALTER TABLE agent_budget_log DROP COLUMN cache_read_tokens;
ALTER TABLE agent_budget_log DROP COLUMN cache_creation_tokens;
