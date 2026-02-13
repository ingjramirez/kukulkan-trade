-- Migration 008: Add regime/session to agent_decisions, influenced_decision to tool_call_logs
ALTER TABLE agent_decisions ADD COLUMN regime VARCHAR(30);
ALTER TABLE agent_decisions ADD COLUMN session_label VARCHAR(20);
ALTER TABLE tool_call_logs ADD COLUMN influenced_decision BOOLEAN DEFAULT 0;
