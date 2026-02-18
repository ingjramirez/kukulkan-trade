-- Extended hours flag on intraday snapshots
ALTER TABLE intraday_snapshots ADD COLUMN is_extended_hours BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE intraday_snapshots ADD COLUMN market_phase VARCHAR(20) NOT NULL DEFAULT 'market';
-- market_phase: 'premarket' | 'market' | 'afterhours'

-- Sentinel action queue (for after-hours and quiet hours)
CREATE TABLE IF NOT EXISTS sentinel_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id VARCHAR(36) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    action_type VARCHAR(20) NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    reason TEXT NOT NULL,
    source VARCHAR(30) NOT NULL,
    alert_level VARCHAR(10) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    resolved_at TIMESTAMP,
    resolved_by VARCHAR(20),
    FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE
);

CREATE INDEX idx_sentinel_actions_tenant_status
    ON sentinel_actions(tenant_id, status, created_at DESC);

-- Quiet hours config on tenant
ALTER TABLE tenants ADD COLUMN quiet_hours_start VARCHAR(5) DEFAULT '21:00';
ALTER TABLE tenants ADD COLUMN quiet_hours_end VARCHAR(5) DEFAULT '07:00';
ALTER TABLE tenants ADD COLUMN quiet_hours_timezone VARCHAR(40) DEFAULT 'America/Mexico_City';
