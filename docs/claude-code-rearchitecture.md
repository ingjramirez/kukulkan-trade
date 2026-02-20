# Kukulkan Trade: Claude Code Agent Rearchitecture

## Executive Summary

Replace the custom Python agent loop (`AgentRunner` + `RequestPacer` + `TokenTracker` + tiered model routing) with Claude Code CLI running under a Claude Max subscription. This eliminates per-token API costs, removes rate limit constraints, provides a 1M-token context window, and gives the agent native memory management.

**What changes:** The agent execution layer (how Portfolio B talks to Claude).
**What stays:** The Python orchestrator, APScheduler, data pipeline, Alpaca execution, FastAPI dashboard, Telegram notifications, and all 24 tool implementations.

---

## 1. Problem Statement

### Current Architecture Pain Points

| Problem | Impact | Root Cause |
|-|-|-|
| 429 rate limit errors | Failed sessions, no trades executed | 30K TPM org limit (Tier 1 API) |
| $0.50-0.67/session cost | $3/day budget cap limits investigation depth | Per-token API billing |
| 8-turn max with budget cutoff | Agent can't finish complex analysis | Cost guard in TokenTracker |
| 116 tool calls, 11x portfolio_state | Wastes tokens on redundant data | No cross-turn context awareness |
| Lossy session compression | Agent forgets prior reasoning | Custom compression to stay in budget |
| 3 separate sessions/day (morning/midday/close) | Context lost between sessions | Budget forces short sessions |
| HaikuScanner + OpusValidator untracked | Eat into same org quota, cause 429s | Bolt-on tiered model routing outside pacer |
| Custom memory system (DB rows) | Limited, no semantic search | Built from scratch, 3 categories only |

### Quantified Waste (from Feb 20 audit)

- Morning run: $0.59, hit 429 on finalize, **zero trades** — entire session wasted
- Manual run: $0.67, 211K tokens, 116 tool calls — `get_portfolio_state` called 11 times
- Combined daily cost: ~$2.00-2.50 for sessions that frequently produce no trades
- Monthly API spend: ~$60-75

---

## 2. Proposed Architecture

### High-Level Design

```
┌──────────────────────────────────────────────────┐
│           Python Orchestrator (unchanged)          │
│  APScheduler: 3x daily + sentinel + signal engine  │
│  Data pipeline: yfinance, Alpaca, news, ChromaDB   │
│  Writes: data/agent-workspace/context.md            │
└────────────────────┬─────────────────────────────┘
                     │ subprocess: claude -p ... --resume
                     │ --mcp-config mcp.json
                     │ --output-format json
                     ▼
┌──────────────────────────────────────────────────┐
│           Claude Code CLI (Max subscription)        │
│  Model: Sonnet 4.6 | Context: 1M tokens            │
│  Working dir: /opt/kukulkan-trade/data/agent-workspace/ │
│  CLAUDE.md: trading directives + hard rules         │
│  MEMORY.md: auto-memory (theses, observations)      │
│  Reads context.md, calls MCP tools, returns JSON    │
└────────────────────┬─────────────────────────────┘
                     │ MCP protocol (stdio)
                     ▼
┌──────────────────────────────────────────────────┐
│           MCP Tool Server (Python, stdio)           │
│  Wraps existing 24 agent tools via ToolRegistry     │
│  Same async handlers, same Alpaca/DB connections    │
│  Runs as subprocess of Claude Code                  │
└──────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Rationale |
|-|-|-|
| Invocation method | CLI subprocess (`claude -p`) | No JS sidecar needed. Python calls `subprocess.run()` |
| Session strategy | One resumed session per day | Morning starts, midday/close resume. Full context preserved |
| Tool exposure | MCP server (stdio transport) | Standard protocol. Claude Code connects automatically |
| Workspace location | `data/agent-workspace/` | Inside repo, easy to back up, `.gitignore`-able |
| Memory | Claude Code native (CLAUDE.md + auto-memory) | Replaces custom DB memory. Richer, persistent, semantic |
| Output format | `--output-format json` + `--json-schema` | Enforces structured trade response |
| Permissions | `--allowedTools` whitelist + `bypassPermissions` | Non-interactive, no prompts |

---

## 3. Component Design

### 3.1 MCP Tool Server (`src/agent/mcp_server.py`)

A single Python process exposing all 24 agent tools over stdio. Claude Code spawns it as a subprocess.

**Architecture:**
- Uses the `mcp` Python SDK (`pip install mcp`)
- Imports existing tool handler functions from `src/agent/tools/`
- Receives tool calls from Claude Code, dispatches to existing async handlers
- Returns results as `TextContent` (JSON-serialized)

**Tool mapping (all 24):**

| Category | Tools |
|-|-|
| Portfolio (10) | `get_portfolio_state`, `get_position_detail`, `get_portfolio_performance`, `get_historical_trades`, `get_correlation_check`, `get_risk_assessment`, `list_discovered_tickers`, `get_current_positions`, `get_position_pnl`, `get_portfolio_summary` |
| Market (8) | `get_batch_technicals`, `get_sector_heatmap`, `get_market_overview`, `get_earnings_calendar`, `get_signal_rankings`, `search_ticker_info`, `get_price_and_technicals`, `get_market_context` |
| Actions (9) | `execute_trade`, `set_trailing_stop`, `get_order_status`, `save_observation`, `declare_posture`, `update_watchlist`, `discover_ticker`, `propose_trades`, `save_memory_note` |
| News (4) | `search_news`, `search_historical_news`, `get_portfolio_a_status`, `get_portfolio_a_history` |

**MCP server initialization:**
- Receives database connection string, Alpaca credentials, and tenant_id via environment variables
- Initializes the same `Database`, `AlpacaClient`, and `ToolRegistry` instances the current system uses
- Tools register exactly as they do today — the MCP layer is a thin adapter

**Estimated size:** ~200 lines (adapter only, tool logic unchanged)

### 3.2 Claude Code Invoker (`src/agent/claude_invoker.py`)

Replaces `AgentRunner` + `PersistentAgent`. A Python module that:

1. Writes market context to `data/agent-workspace/context.md`
2. Invokes `claude` CLI via `subprocess.run()`
3. Parses JSON response
4. Returns the same `(trades, reasoning, tool_summary)` 3-tuple the orchestrator expects

**CLI invocation:**
```bash
claude \
  -p "$(cat data/agent-workspace/context.md)" \
  --resume "$SESSION_ID" \
  --model claude-sonnet-4-6 \
  --output-format json \
  --json-schema '{"type":"object","properties":{"trades":{"type":"array"},...}}' \
  --mcp-config data/agent-workspace/mcp.json \
  --allowedTools "mcp__kukulkan__*" \
  --permission-mode bypassPermissions \
  --max-turns 25 \
  --cwd /opt/kukulkan-trade/data/agent-workspace
```

**Session strategy:**
- Morning: starts a new session, stores session_id in DB
- Midday: resumes morning session via `--resume $SESSION_ID`
- Closing: resumes same session via `--resume $SESSION_ID`
- Result: one continuous session per day with full context

**Error handling:**
- Subprocess timeout (10 min) — return empty trades with error reasoning
- Non-zero exit code — log stderr, return graceful fallback
- JSON parse failure — extract text, attempt parse, fallback to empty

**Estimated size:** ~150 lines

### 3.3 Trading Workspace (`data/agent-workspace/`)

```
data/agent-workspace/
├── CLAUDE.md                  # Trading directives (replaces system prompt)
├── .claude/
│   └── settings.json          # Permissions, model config
├── mcp.json                   # MCP server configuration
├── context.md                 # Written by orchestrator before each invocation
└── .gitignore                 # Ignore context.md, session state
```

**CLAUDE.md** — replaces `build_system_prompt()` (currently 11 blocks, ~3000 tokens). Contains:
- Identity and role definition
- Decision framework and hard rules
- Risk rules (max position size, correlation limits, stop-loss policy)
- Output format (JSON schema with trades, reasoning, regime_assessment)
- Tool usage guidance ("use get_signal_rankings first, don't re-fetch data you already have")
- Session type behavior (morning = full analysis, midday = check + adjust, closing = final review)

**context.md** — written fresh by orchestrator before each `claude -p` call. Contains:
- Current date, session type (morning/midday/closing), regime
- Portfolio state (cash, positions, P&L)
- Signal engine rankings (top 20)
- News summary (from ChromaDB)
- Macro data (VIX, yield curve, Fear & Greed)
- Earnings calendar
- Any alerts from sentinel

**mcp.json:**
```json
{
  "mcpServers": {
    "kukulkan": {
      "type": "stdio",
      "command": "python",
      "args": ["/opt/kukulkan-trade/src/agent/mcp_server.py"],
      "env": {
        "DATABASE_URL": "${DATABASE_URL}",
        "ALPACA_API_KEY": "${ALPACA_API_KEY}",
        "ALPACA_SECRET_KEY": "${ALPACA_SECRET_KEY}",
        "TENANT_ID": "${TENANT_ID}"
      }
    }
  }
}
```

### 3.4 Orchestrator Changes (`src/orchestrator.py`)

**Modified method:** `_run_portfolio_b()` (line 1574)

Current flow:
```
_build_portfolio_b_context() → _build_dynamic_context() → _build_portfolio_b_prompt()
→ HaikuScanner.scan() → model routing → PersistentAgent.run_session()
→ AgentRunner.run() → parse response → (trades, reasoning, tool_summary)
```

New flow:
```
_build_portfolio_b_context() → write context.md
→ ClaudeInvoker.run(session_type, tenant_id) → subprocess claude -p
→ parse JSON response → (trades, reasoning, tool_summary)
```

The orchestrator's `_run_portfolio_b()` becomes ~30 lines instead of ~400. All the seed/investigate/tiered-model branching is deleted.

---

## 4. What Gets Deleted

| Component | Lines | Reason |
|-|-|-|
| `src/agent/agent_runner.py` | 493 | Replaced by Claude Code CLI |
| `src/agent/request_pacer.py` | 101 | No rate limits on Max |
| `src/agent/token_tracker.py` | 177 | No per-token billing |
| `src/agent/persistent_agent.py` | 300 | Session management handled by `--resume` |
| `src/agent/context_manager.py` | ~200 | Context built as markdown file |
| `src/agent/conversation_store.py` | ~150 | Sessions stored by Claude Code natively |
| `src/agent/session_compressor.py` | ~120 | 1M context, no compression needed |
| `src/agent/haiku_scanner.py` | 161 | No tiered model routing |
| `src/agent/opus_validator.py` | 157 | No tiered model routing |
| `src/agent/claude_agent.py` (build_system_prompt) | ~300 | Replaced by CLAUDE.md |
| Orchestrator branching (seed/investigate/tiered) | ~400 | Simplified to single invoke |
| **Total removed** | **~2,560** | |

| Component | Lines | Purpose |
|-|-|-|
| `src/agent/mcp_server.py` (new) | ~200 | MCP adapter for 24 tools |
| `src/agent/claude_invoker.py` (new) | ~150 | CLI subprocess wrapper |
| `data/agent-workspace/CLAUDE.md` (new) | ~200 | Trading directives |
| **Total added** | **~550** | |

**Net reduction: ~2,000 lines of agent infrastructure code.**

---

## 5. What Stays Unchanged

| Component | Reason |
|-|-|
| `src/agent/tools/` (portfolio.py, market.py, actions.py, news.py) | Tool logic is reusable — only the transport layer changes |
| `src/agent/tools/__init__.py` (ToolRegistry) | MCP server uses it to dispatch calls |
| `src/orchestrator.py` (data pipeline, Portfolio A, execution, notifications) | Only Portfolio B agent path changes |
| `src/main.py` (scheduler, all jobs) | Unchanged — calls orchestrator same as before |
| `src/storage/` (database, models, migrations) | All CRUD stays. Memory table becomes optional |
| `src/api/` (FastAPI dashboard) | Read-only, unaffected |
| `config/` (settings, universe, risk_rules) | Some agent settings become unused but harmless |
| Telegram notifications | Unchanged |
| Signal engine, sentinel, gap risk | Unchanged |

---

## 6. Cost Analysis

### Current (API Tier 1)

| Item | Monthly Cost |
|-|-|
| Morning sessions (20 days x $0.60) | $12.00 |
| Midday sessions (20 days x $0.15) | $3.00 |
| Closing sessions (20 days x $0.40) | $8.00 |
| Manual sessions (~10/month x $0.50) | $5.00 |
| Sentinel escalations (~4/month x $0.30) | $1.20 |
| Weekly reviews (4x x $0.20) | $0.80 |
| **Total** | **~$30-75/month** |

### Proposed (Claude Max)

| Item | Monthly Cost |
|-|-|
| Claude Max subscription | $100-200/month (depending on plan) |
| API calls | $0 |
| **Total** | **$100-200/month** |

### Cost Tradeoff

At current usage ($30-75/month API), Max is more expensive in raw dollars. The value proposition is not cost savings — it's **capability**:

- No 429 errors (sessions never fail due to rate limits)
- No 8-turn budget cap (agent can investigate as deeply as needed)
- 1M context (full day's context preserved across sessions)
- Better tool efficiency (fewer redundant calls)
- Native memory (richer reasoning across days/weeks)
- Eliminated complexity (2,000 fewer lines to maintain)

If API usage grows (more tenants, more frequent sessions, higher budgets), Max becomes cost-effective. The break-even is roughly **$100-200/month in API spend**.

---

## 7. Migration Strategy

### Phase 1: MCP Tool Server

Build and test the MCP server independently. Verify all 24 tools work over stdio.

**Deliverables:**
- `src/agent/mcp_server.py`
- `data/agent-workspace/mcp.json`
- Tests: MCP server unit tests (tool dispatch, error handling)

**Verification:** Run MCP server standalone, call each tool via `claude -p "call get_portfolio_state" --mcp-config mcp.json`

### Phase 2: Trading Workspace

Create the workspace with CLAUDE.md directives and context.md template.

**Deliverables:**
- `data/agent-workspace/CLAUDE.md`
- `data/agent-workspace/.claude/settings.json`
- Context builder function (writes context.md)

**Verification:** Run `claude -p "analyze this portfolio" --mcp-config mcp.json` from the workspace and verify it calls tools and returns structured JSON.

### Phase 3: Claude Invoker

Build the subprocess wrapper and wire it into the orchestrator.

**Deliverables:**
- `src/agent/claude_invoker.py`
- Orchestrator changes (`_run_portfolio_b()` simplified)
- Session ID management (DB column for daily session tracking)

**Verification:** Run a full morning pipeline on VPS with real market data. Compare output quality to current system.

### Phase 4: Parallel Run

Run both systems simultaneously for 1-2 weeks. Current API agent as primary, Claude Code agent as shadow.

**Verification:** Compare trade proposals, reasoning quality, tool efficiency, and session stability.

### Phase 5: Cutover + Cleanup

Switch Claude Code agent to primary. Delete old agent infrastructure.

**Deliverables:**
- Remove: AgentRunner, RequestPacer, TokenTracker, PersistentAgent, HaikuScanner, OpusValidator, etc.
- Update: settings.py (remove agent budget/pacer settings)
- Update: tests (remove old agent tests, add new invoker tests)

---

## 8. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|-|-|-|-|
| Claude Max usage limits (undocumented caps) | Medium | Agent sessions throttled | Monitor usage; keep API fallback path for first month |
| Claude Code CLI breaking changes | Low | Subprocess calls fail | Pin CLI version; wrap in try/except with API fallback |
| MCP server stability (long-running stdio) | Medium | Tool calls fail mid-session | Heartbeat monitoring; auto-restart on crash |
| Output format compliance | Medium | JSON parse failures | `--json-schema` enforcement; fallback parsing |
| Claude Code context compaction | Low | Older context summarized away | Use `/checkpoint` hooks; keep context.md under 50K tokens |
| Multi-tenant session isolation | Medium | Tenant A data leaks to B | Separate sessions per tenant; separate MCP server instances |
| VPS resource usage (Node.js + Python) | Low | Memory pressure | Monitor; Claude Code is lightweight (~100MB RSS) |

---

## 9. Prerequisites

| Requirement | Status |
|-|-|
| Claude Max subscription (Pro or Team) | Needed |
| Claude Code CLI installed on VPS | Done (v2.1.49) |
| Claude Code authenticated on VPS | Needed (`claude login`) |
| `mcp` Python SDK installed | Needed (`pip install mcp`) |
| Node.js 22+ on VPS | Done (v22.22.0) |

---

## 10. Success Metrics

| Metric | Current | Target |
|-|-|-|
| Session failure rate (429s) | ~20% of morning runs | 0% |
| Tool calls per session | 116 (50%+ redundant) | <40 |
| Agent turns per session | 4-5 (budget-capped) | 10-15 (investigation-complete) |
| Daily API cost | $2.00-2.50 | $0 |
| Lines of agent infrastructure | ~2,560 | ~550 |
| Context preserved across sessions | Lossy (compressed) | Full (1M window) |
| Memory richness | 3 DB categories | Native files (unlimited) |
