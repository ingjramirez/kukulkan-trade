-- Migration 010: Posture history, playbook snapshots, conviction calibration
-- Phase 33.2 — Dynamic Strategy + Self-Improvement

CREATE TABLE IF NOT EXISTS posture_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    session_date DATE NOT NULL,
    session_label TEXT,
    posture TEXT NOT NULL,
    effective_posture TEXT NOT NULL,
    reason TEXT,
    created_at DATETIME NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS playbook_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    generated_at DATETIME NOT NULL DEFAULT (datetime('now')),
    regime TEXT NOT NULL,
    sector TEXT NOT NULL,
    total_trades INTEGER NOT NULL,
    wins INTEGER NOT NULL,
    losses INTEGER NOT NULL,
    win_rate_pct REAL NOT NULL,
    avg_pnl_pct REAL NOT NULL,
    recommendation TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conviction_calibration (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    generated_at DATETIME NOT NULL DEFAULT (datetime('now')),
    conviction_level TEXT NOT NULL,
    total_trades INTEGER NOT NULL,
    wins INTEGER NOT NULL,
    losses INTEGER NOT NULL,
    win_rate_pct REAL NOT NULL,
    avg_pnl_pct REAL NOT NULL,
    assessment TEXT NOT NULL,
    suggested_multiplier REAL NOT NULL DEFAULT 1.0
);
