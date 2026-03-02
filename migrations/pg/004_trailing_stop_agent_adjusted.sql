-- Add agent_adjusted_at to trailing_stops for sentinel grace period suppression.
-- When the AI bot tightens a stop, this timestamp is set so sentinel
-- can skip WARNING alerts for that stop within the grace window.
ALTER TABLE trailing_stops ADD COLUMN IF NOT EXISTS agent_adjusted_at TIMESTAMP NULL;
