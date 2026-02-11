-- Add pending_rebalance flag to tenants table.
-- Set by API when portfolio toggles change; cleared by orchestrator after rebalance.
ALTER TABLE tenants ADD COLUMN pending_rebalance BOOLEAN NOT NULL DEFAULT 0;
