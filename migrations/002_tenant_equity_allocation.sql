-- Add initial equity tracking and percentage-based allocation to tenants.
-- initial_equity is captured from Alpaca on first run.
-- portfolio_a_pct / portfolio_b_pct replace hardcoded dollar allocations.

ALTER TABLE tenants ADD COLUMN initial_equity FLOAT;
ALTER TABLE tenants ADD COLUMN portfolio_a_pct FLOAT NOT NULL DEFAULT 33.33;
ALTER TABLE tenants ADD COLUMN portfolio_b_pct FLOAT NOT NULL DEFAULT 66.67;
