# Plan: Fully Agentic Trading Bot — Persistent Architecture

**Date**: 2026-02-15
**Author**: Claude (AI architect) + J. Ramirez (owner) + Claude (analyst)
**Status**: v2.1 — post analyst review, revised

---

## 1. Executive Summary

Kukulkan Trade is an educational trading bot managing $99K across two portfolios on Alpaca paper trading. Today, the AI (Claude) controls Portfolio B ($66K) with a limited tool kit and stateless sessions — each API call starts from scratch with no memory of prior conversations. Portfolio A ($33K) runs a purely mechanical momentum strategy with zero AI involvement.

**Proposal**: Transform the system into a **persistent agentic architecture** where Claude runs as a continuous agent on the VPS, maintains full conversation history across sessions, and uses the trading bot's capabilities as its toolset. This eliminates the "amnesia problem" and enables multi-day planning, organic learning from experience, and dramatically more informed decisions.

**Key architectural choice**: **API-only** — build a custom persistent agent process using the Anthropic Python SDK, running on the existing Hetzner VPS. The trading bot's capabilities are exposed as native Anthropic `tool_use` tools. All compute billed per-token via the existing `ANTHROPIC_API_KEY` — no subscription required, full cost control.

**Pricing decision**: API pay-per-token. Projected $12-30/month normal, $50-75 worst case. Owner retains full control over compute spend with configurable budget caps per session and per day.

**Portfolio A** remains rule-based as a control group. Claude gets read-only visibility into it for cross-portfolio coordination.

**Claude Code on VPS**: Installed for interactive development/monitoring (SSH in, run `claude`). Uses the same API key — costs $0 when idle, pay-per-token when used. Not part of the automated trading pipeline.

---

## 2. Current State Assessment

### What Works Today

| Capability | Implementation | Effectiveness |
|-----------|----------------|---------------|
| Market regime classification | 5-point scale (SPY/VIX/breadth) | Good signal |
| 3x daily sessions | APScheduler (10am/12:30pm/3:45pm ET) | Good cadence |
| Strategy directives | Conservative/Standard/Aggressive | Clear frameworks |
| Outcome tracking | P&L + alpha vs SPY + sector ETF | Strong feedback |
| Track record | Win rate by sector/regime/conviction | Useful learning |
| Risk management | Pre-trade checks, circuit breakers | Solid guardrails |
| Trailing stops | Per-position with configurable % | Working well |
| News pipeline | 3-source aggregation + ChromaDB vectors | Decent coverage |

### What Doesn't Work

| Problem | Impact | Root Cause |
|---------|--------|------------|
| **Stateless sessions** | Claude forgets everything between calls | API is request/response, no persistence |
| **Limited tools** | Can only query 3-4 tickers per session | 1-ticker-per-call tools, 8-turn cap |
| **No cross-portfolio visibility** | Can't coordinate A and B | Portfolio A is a black box to Claude |
| **Static strategy mode** | Can't adapt risk posture to conditions | Tenant-level setting, not dynamic |
| **Hardcoded constraints** | Conviction sizing doesn't learn | Fixed multipliers (1.0/0.7/0.4) |
| **Flat memory** | 10 notes dumped into prompt every time | No search, no structure, no evolution |
| **No multi-day planning** | Can't build positions over sessions | Each session is independent |
| **No historical self-awareness** | Can't query own performance via tools | Summary in prompt, not interactive |

### Budget Reality

| Metric | Current | Notes |
|--------|---------|-------|
| Daily API cost | $0.25-0.35 | 3 sessions × Sonnet agentic |
| Monthly cost | $7-10 | Negligible vs portfolio size |
| Turns per session | 3-4 useful | Out of 8 max (tools too limited) |
| Tickers analyzed per session | 3-4 | 1-per-call bottleneck |
| Context utilization | ~5K of 200K tokens | 2.5% of available context |

---

## 3. Proposed Architecture: Persistent Agentic Agent

### 3.1 — Core Concept: Triggered Process Model

Replace the current "bot calls Claude API" pattern with a **triggered process** running on the VPS. **Not a long-running daemon** — each scheduler trigger spawns a process that:

1. Loads conversation history from SQLite
2. Builds context (pinned + summaries + recent history)
3. Runs Claude tool loop (Anthropic SDK)
4. Saves session to DB
5. Exits cleanly

This avoids daemon complexity (memory leaks, zombie processes, socket management). The conversation state lives in the DB, not in memory. Crash recovery is free — if a process dies mid-session, the next trigger picks up from the last saved state.

```
┌─────────────────────────────────────────────────────────┐
│  VPS (Hetzner)                                          │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │  APScheduler (existing kukulkan-bot process)     │   │
│  │  10am / 12:30pm / 3:45pm ET + event triggers     │   │
│  └──────────────┬───────────────────────────────────┘   │
│                 │ spawns                                 │
│                 ▼                                        │
│  ┌──────────────────────────────────────────────────┐   │
│  │  PersistentAgent (triggered process)             │   │
│  │                                                  │   │
│  │  1. Load conversation history from SQLite        │   │
│  │  2. Build context (pinned + summaries + recent)  │   │
│  │  3. Call Claude API with tool loop               │   │
│  │  4. Save session + exit                          │   │
│  │                                                  │   │
│  │  ┌──────────────────────────────────────────┐    │   │
│  │  │         Claude API (Anthropic)           │    │   │
│  │  │  • Sonnet 4.5 (analysis + tools)         │    │   │
│  │  │  • Opus 4.6 (validation / crisis)        │    │   │
│  │  │  • Haiku 4.5 (scanning / compression)    │    │   │
│  │  └──────────────────┬───────────────────────┘    │   │
│  │                     │                            │   │
│  │                     ▼                            │   │
│  │  ┌──────────────────────────────────────────┐    │   │
│  │  │         Tool Registry (18 tools)         │    │   │
│  │  │  Portfolio │ Market │ Risk │ Actions      │    │   │
│  │  │  WRAPS existing bot infrastructure       │    │   │
│  │  └──────────────────┬───────────────────────┘    │   │
│  │                     │                            │   │
│  │                     ▼                            │   │
│  │  ┌──────────────────────────────────────────┐    │   │
│  │  │      Existing Bot Infrastructure         │    │   │
│  │  │  • Database (SQLite)                     │    │   │
│  │  │  • Alpaca executor + RiskManager         │    │   │
│  │  │  • News pipeline + ChromaDB              │    │   │
│  │  │  • Telegram notifier                     │    │   │
│  │  └──────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  Existing services: kukulkan-bot, kukulkan-api, nginx   │
└─────────────────────────────────────────────────────────┘
```

### 3.2 — Conversation Persistence

The agent maintains a continuous conversation stored in SQLite:

**Recent sessions** (last 5): Full message history, all tool calls and results.
**Older sessions** (6-30): Compressed to ~500-token summaries by Haiku.
**Pinned context**: Always present — portfolio state, active theses, key learnings, risk parameters.
**System prompt**: Strategy framework, tool definitions, guardrails.

**Context budget** (~45K of 200K tokens):
- System prompt + tools: ~5K
- Pinned context: ~3K
- Compressed history: ~12K (25 sessions × 500 tokens)
- Recent history: ~25K (5 sessions in full)

**Compression trigger**: When total context exceeds 150K tokens, compress oldest full session to summary.

**Compression validation**: Before committing to Haiku compression, test with 5 real sessions:
1. Compress each session with Haiku
2. Present compressed + original to Sonnet, ask "what's missing?"
3. If critical trading decisions or position rationale are lost, adjust compression prompt
4. Only deploy compression after validating that Sonnet can reconstruct key facts from summaries

**Crash recovery**: Each session is atomic — the triggered process loads from DB, runs, saves, exits. If a process dies mid-session:
- Last completed session is intact in DB
- Partial session state is lost (no harm — trades already submitted are tracked by Alpaca)
- `sync_positions()` on next trigger reconciles any partial state against broker positions
- No orphaned connections or zombie processes (process-per-trigger model)

### 3.3 — Session Triggers

The scheduler sends **event messages** to the agent (not API calls — messages in the ongoing conversation):

| Trigger | When | Message to Claude |
|---------|------|-------------------|
| Morning open | Mon-Fri 10:00 AM ET | "Good morning. Markets open. [regime, VIX, overnight summary]" |
| Midday check | Mon-Fri 12:30 PM ET | "Midday update. [intraday moves, any alerts]" |
| Pre-close | Mon-Fri 3:45 PM ET | "Markets closing in 15 min. [day summary, overnight risk]" |
| Trailing stop hit | Real-time | "ALERT: Trailing stop triggered on {ticker} at ${price}" |
| VIX spike | Real-time (VIX >28) | "ALERT: VIX crossed {level}. Current: {vix}" |
| Deposit detected | On detection | "Deposit of ${amount} detected. Available for allocation." |
| Weekly review | Sunday 6:00 PM | "Weekly review time. Here are your outcomes this week..." |

Claude responds naturally to each trigger, using tools as needed, referencing prior sessions.

### 3.4 — Conversation Flow Example

```
── Monday Morning ──────────────────────────────────────────
[Scheduler]: Good morning. Markets opening. VIX 18.2, futures +0.3%.
             Regime: CONSOLIDATION (unchanged). No overnight events.

[Claude]: Let me check the portfolio state.
          → uses get_portfolio_state()
          → uses get_batch_technicals(["XLK","NVDA","GLD","TLT","AAPL"])

[Claude]: Portfolio looks healthy — $62K invested, 38% cash.
          NVDA pulled back to $118, near the $115 support I identified
          last Thursday. RSI at 32 (oversold). News is mixed but no
          fundamental change. I'll start a half-size position — plan
          to add if it holds $115 by Wednesday.
          → uses execute_trade("NVDA", "BUY", 50)
          → uses set_trailing_stop("NVDA", 0.07)

── Monday Midday ───────────────────────────────────────────
[Scheduler]: Midday check. NVDA at $119.50 (+1.3% since open).

[Claude]: NVDA holding nicely since my morning entry at $118.
          No action needed. Keeping my plan to add Wednesday if
          support confirms. Let me check if anything else needs
          attention...
          → uses get_portfolio_summary()

[Claude]: All positions within parameters. XLK near trailing
          stop trigger (2.1% away) — I'll watch this at close.

── Monday Close ────────────────────────────────────────────
[Scheduler]: Pre-close. 15 minutes to bell. NVDA $120, XLK $154.

[Claude]: XLK recovered, now 3.8% from stop — safe for overnight.
          NVDA holding above $118 entry. Good day. No action.

── Wednesday Morning ───────────────────────────────────────
[Scheduler]: Good morning. VIX 17.5. NVDA closed at $121 yesterday.

[Claude]: NVDA has held above $115 support for 3 sessions as planned.
          Thesis confirmed — adding second tranche.
          → uses execute_trade("NVDA", "BUY", 50)

[Claude]: Full position now: 100 shares at $119.50 avg.
          Trailing stop at 7%. This is my highest-conviction
          position alongside XLK.
```

**Key observation**: Claude references "last Thursday", "my plan to add Wednesday", and tracks a multi-day thesis — all impossible with stateless sessions.

---

## 4. Implementation Phases

> **Key design principle (from analyst)**: Separate persistence infrastructure from tool upgrades. If bugs arise, we need to know whether they're from the persistence layer or from new tools. Phase 1 uses the **existing 9 tools** from Phase 32 to validate persistence in isolation.

### Phase 1: Persistent Agent Foundation (Week 1-3)

**Goal**: Build the persistent conversation infrastructure using the EXISTING 9 tools. Validate that persistence + context management works correctly before changing anything else.

#### 1A — ConversationStore (SQLite)

New table: `agent_conversations` (migration 009)

| Column | Type | Purpose |
|--------|------|---------|
| `id` | Integer PK | Auto-increment |
| `tenant_id` | String | Multi-tenant from day 1 |
| `session_id` | String | UUID per trigger |
| `trigger_type` | String | morning/midday/close/event |
| `messages_json` | Text | Full message history (JSON) |
| `summary` | Text | Haiku-compressed summary (null if recent) |
| `token_count` | Integer | For context budget tracking |
| `cost_usd` | Float | Session cost |
| `created_at` | DateTime | Session timestamp |

Methods:
- `save_session(trigger, messages, cost)` — persist after each session
- `load_recent(n=5)` — full message history for last N sessions
- `load_summaries(n=25)` — compressed summaries of older sessions
- `compress_session(session_id)` — call Haiku to summarize, save, free full history

#### 1B — ContextManager

Builds the `messages` array for each Claude API call:

```python
class ContextManager:
    def build_system_prompt(self, pinned: str, tool_defs: list) -> str
    def build_messages(self, summaries: list[str], recent: list[dict], trigger: dict) -> list[dict]
    def estimate_tokens(self, messages: list[dict]) -> int
    def should_compress(self, total_tokens: int) -> bool  # threshold: 150K
```

#### 1C — SessionCompressor

```python
class SessionCompressor:
    async def compress(self, session_messages: list[dict]) -> str
        # Calls Haiku with: "Summarize this trading session in ~500 tokens.
        # Preserve: trades executed, positions changed, key observations, theses."
```

**Validation gate (must pass before deploying)**:
1. Compress 5 real sessions with Haiku
2. Present compressed + original to Sonnet: "What trading-relevant facts are missing?"
3. If any trade, position change, or thesis is lost, adjust compression prompt
4. Record compression fidelity score (target: >95% of key facts preserved)

#### 1D — PersistentAgent (Hybrid Wrapping)

**Migration approach**: PersistentAgent **wraps** existing proven components, not replaces them.

```python
class PersistentAgent:
    def __init__(self, db, tenant, ...):
        self.conversation_store = ConversationStore(db)
        self.context_manager = ContextManager()
        self.session_compressor = SessionCompressor()
        # WRAP existing proven components — do NOT rewrite
        self.agent_runner = AgentRunner(...)      # existing tool loop
        self.tool_registry = ToolRegistry()        # existing 9 tools
        self.memory_manager = AgentMemoryManager() # existing memory

    async def on_session(self, trigger: str):
        history = self.conversation_store.load_recent(sessions=5)
        summaries = self.conversation_store.load_summaries()
        system_prompt = self.context_manager.build_system_prompt(
            pinned=self._pinned_context(),
        )
        user_message = self.context_manager.build_trigger_message(
            trigger=trigger, history=history, summaries=summaries,
        )
        # Use existing AgentRunner for the tool loop
        result = await self.agent_runner.run(system_prompt, user_message)
        self.conversation_store.save_session(trigger, result)
```

This means Phase 32's AgentRunner, ToolRegistry, and AgentMemoryManager continue working. PersistentAgent adds the persistence layer around them.

**Phase 1 deliverables**: ConversationStore, ContextManager, SessionCompressor, PersistentAgent wrapping existing 9 tools, compression validation, crash recovery.

**Estimated effort**: 10-12 days development + tests
**Tests**: ~100-120 new tests (conversation store, context manager, compressor, integration)

---

### Phase 2: Upgraded Tool Kit (Week 4-5)

**Goal**: Expand from 9 to 18 tools. Since persistence is proven in Phase 1, any issues in Phase 2 are clearly tool-related.

#### 2A — Batch Market Tools (eliminate turn waste)

| New Tool | Replaces | Improvement |
|----------|----------|-------------|
| `get_batch_technicals(tickers[])` | `get_price_and_technicals` (1-at-a-time) | 5-20 tickers per call |
| `get_sector_heatmap()` | Static sector data in `get_market_context` | Full rotation signals |
| `get_portfolio_performance(period)` | Text summary in system prompt | Interactive, queryable |
| `get_portfolio_a_status()` | Nothing (blind today) | Cross-portfolio awareness |
| `get_correlation_check(tickers[])` | Text in system prompt | On-demand analysis |
| `get_historical_trades(days)` | Last 5 trades in prompt | Full trade history |

#### 2B — Risk Management Tools (agent controls risk)

| New Tool | Replaces | Improvement |
|----------|----------|-------------|
| `set_trailing_stop(ticker, pct)` | Hardcoded conviction x mode matrix | Contextual stop placement |
| `get_earnings_calendar(tickers[])` | Text in system prompt | Interactive query |
| `get_risk_assessment()` | Nothing | Portfolio VaR, drawdown, exposure |

#### 2C — Execution Enhancement

| New Tool | Purpose |
|----------|---------|
| `execute_trade(ticker, side, shares, reason)` | Direct execution with fill result returned to agent |
| `get_order_status(ticker)` | Check if orders filled |

**execute_trade sequencing**: The tool submits the trade, polls for fill, and returns the fill result directly to the agent — no waiting for next session:

```python
# Agent calls:
execute_trade("XLK", "SELL", 100, "rotating to energy")

# Tool internally:
# 1. RiskManager.check_pre_trade() — blocks if rules violated
# 2. If trade > 10% of portfolio → Telegram approval (blocks until confirmed)
# 3. AlpacaExecutor.submit_order()
# 4. Poll for fill (up to 30s)
# 5. Return result to agent:

{"success": true, "filled_shares": 100, "fill_price": 154.20, "cash_freed": 15420.00}
```

**Telegram trade approval**: Trades exceeding 10% of portfolio value require human confirmation via Telegram before execution. The tool sends a Telegram message with trade details and waits for /approve or /reject (60s timeout, defaults to reject).

**Phase 2 deliverables**: 9 new tools (18 total), batch operations, fill results in tool loop, Telegram approval for large trades.

**Estimated effort**: 7-9 days development + tests
**Tests**: ~80-100 new tests (new tools, Telegram approval flow)

---

### Phase 3: Dynamic Strategy + Self-Improvement (Week 6-7)

**Goal**: Remove static constraints and let Claude adapt its behavior based on conditions and track record.

#### 3A — Dynamic Risk Posture

With persistent conversations, Claude naturally adjusts its behavior. But we formalize it:

**New tool: `declare_posture(posture, reason)`**

Claude announces its risk posture each session. The system tracks it and enforces corresponding limits:

| Posture | Cash Floor | Max Position | Max Equity | When to Use |
|---------|-----------|-------------|-----------|-------------|
| Balanced | 20% | 15% | 80% | Normal conditions |
| Defensive | 40% | 10% | 60% | Correction, uncertainty |
| Crisis | 70% | 5% | 30% | VIX >30, drawdown >10% |

> **Posture cap**: Initially, the agent is limited to **Balanced** as the most aggressive posture. "Aggressive" mode (5% cash, 95% equity) is gated behind a proven track record — unlocked only after 50+ trades with win rate >55% and positive alpha vs SPY. This prevents overconfidence before the system has calibrated.

Claude chooses posture based on regime, track record, VIX, earnings exposure. The system enforces the corresponding limits via the RiskManager.

#### 3B — Empirical Playbook Generation

Weekly automated process:
1. Query all closed trades from `OutcomeTracker`
2. Group by regime x sector x conviction
3. Compute win rates, avg P&L, alpha
4. Format as "playbook" injected into pinned context

Example output:
```
BULL REGIME (23 trades, 65% WR, +1.8% avg):
  + Technology: 78% WR, +2.9% — your sweet spot
  + Consumer Disc: 71% WR, +1.5% — solid
  - Energy: 33% WR, -1.2% — avoid
  - Fixed Income: 40% WR, -0.3% — opportunity cost

CORRECTION (8 trades, 50% WR, +0.4% avg):
  + Fixed Income: 80% WR, +1.1% — safe haven works
  - Technology: 25% WR, -3.1% — stop buying dips
```

Claude sees what has **actually worked** in each regime, not just theoretical rules.

#### 3C — Conviction Calibration (Human-Approved)

After 30+ trades, compute empirical conviction accuracy:
```
High conviction: 80% WR, avg +3.2% -> multiplier = 1.0 (validated)
Medium conviction: 45% WR, avg -0.1% -> multiplier = 0.3 (overconfident)
Low conviction: 60% WR, avg +1.5% -> multiplier = 0.6 (underconfident)
```

Injected into pinned context. Claude sees "my medium-conviction calls are barely better than random — I should either upgrade them to high or skip them."

**Note**: Conviction multiplier changes are **logged and reviewable** — the owner can inspect how sizing evolves and override if needed via Telegram.

**Phase 3 deliverables**: Dynamic posture system (capped at Balanced), weekly playbooks, conviction calibration, all integrated into persistent context.

**Estimated effort**: 5-7 days development + tests
**Tests**: ~60-80 new tests

---

### Phase 4: Compute Optimization (Week 8-9)

**Goal**: Optimize model usage for cost-efficiency and give Claude visibility into Portfolio A.

#### 4A — Tiered Model Strategy

| Step | Model | Purpose | Cost |
|------|-------|---------|------|
| **Scan** | Haiku 4.5 | "Anything unusual?" Quick triage | $0.01 |
| **Analyze** | Sonnet 4.5 | Full tool-use investigation | $0.08-0.15 |
| **Validate** | Opus 4.6 | Review proposed trades, challenge assumptions | $0.05-0.10 |

**Morning session** (full): Scan -> Analyze (8 turns) -> Validate = ~$0.20
**Midday session** (light): Scan -> Analyze (4 turns) = ~$0.10
**Closing session** (review): Scan -> Analyze (4 turns) = ~$0.10
**Crisis session** (auto-triggered): Full Analyze (12 turns) + Validate = ~$0.40

**Daily normal**: $0.40 | **Daily volatile**: $0.80
**Monthly normal**: $12.00 | **Monthly volatile**: $24.00

#### 4B — Prompt Caching Integration

Anthropic prompt caching reduces cost of repeated context by 90%:

| Component | Tokens | Cached? | Savings |
|-----------|--------|---------|---------|
| System prompt + tool defs | ~5,000 | Yes (stable within session) | 90% off |
| Pinned context (playbooks, state) | ~3,000 | Yes (stable within session) | 90% off |
| Conversation history | ~25,000 | Partially (grows per turn) | ~50% off |
| New session content | ~3,000 | No (unique) | 0% |

Cache TTL is 5 minutes (Anthropic default), auto-extends on use. During a tool-use loop (calls every 10-30 seconds), cache stays fully hot. **Between sessions** (2.5+ hours apart), cache expires — first call of each session pays full price.

**Estimated savings**: 30-35% reduction in daily input token costs. Cache savings only apply within sessions (tool loop turns), not between sessions.

#### 4C — Portfolio A Visibility

Claude gets **read-only** access to Portfolio A:

| Tool | Returns |
|------|---------|
| `get_portfolio_a_status()` | Current ETF held, momentum rankings (top 5), 30d return, positions |
| `get_portfolio_a_history(days)` | Recent rotations: which ETFs and when |

Claude can **not** modify Portfolio A. But it can:
- Avoid sector overlap (if A holds XLK, B reduces tech weight)
- Hedge against A's concentration (if A is 100% tech, B can hold defensives)
- Learn from momentum signals (if momentum says rotate to energy, B can investigate why)

Portfolio A continues running mechanically. Claude observes and coordinates, not controls.

**Phase 4 deliverables**: Tiered model usage, prompt caching, Portfolio A read-only tools, daily/monthly budget accumulators.

**Estimated effort**: 5-7 days development + tests
**Tests**: ~50-70 new tests

---

## 5. Complete Tool Registry (Target State)

### Portfolio Tools (7)
| Tool | Description |
|------|-------------|
| `get_portfolio_state()` | Full B state: positions, cash, P&L, sector exposure |
| `get_position_detail(ticker)` | Deep dive: P&L, trailing stop, entry date, thesis |
| `get_portfolio_performance(period)` | Returns, drawdown, Sharpe, alpha, win rate |
| `get_historical_trades(days)` | Past trades with outcomes and verdicts |
| `get_correlation_check(tickers?)` | Portfolio correlation matrix, diversification score |
| `get_portfolio_a_status()` | Read-only: A's current ETF, momentum rankings, return |
| `get_risk_assessment()` | Current exposure, VaR estimate, earnings risk, stop distances |

### Market Tools (4)
| Tool | Description |
|------|-------------|
| `get_batch_technicals(tickers[])` | Bulk: price, returns, RSI, MACD, SMA, volume for 5-20 tickers |
| `get_sector_heatmap()` | All sectors: 1d/5d/20d %, rotation signals, ETF RSI |
| `get_market_overview()` | Regime, VIX + trend, yield curve, SPY stats, breadth |
| `get_earnings_calendar(tickers?)` | Upcoming earnings with days until and portfolio exposure |

### News Tools (2)
| Tool | Description |
|------|-------------|
| `search_news(ticker?)` | Today's compacted headlines, filtered by ticker |
| `search_historical_news(ticker, days)` | ChromaDB semantic search, past context |

### Action Tools (5)
| Tool | Description |
|------|-------------|
| `execute_trade(ticker, side, shares, reason)` | Submit trade (RiskManager + Telegram approval if >10% of portfolio) |
| `set_trailing_stop(ticker, trail_pct, reason)` | Set/update stop with custom % |
| `update_watchlist(action, ticker, ...)` | Add/remove watchlist items |
| `declare_posture(posture, reason)` | Set session risk posture |
| `save_observation(key, content)` | Persist insight to pinned context |

**Total: 18 tools** (vs 9 today, and 2x more effective per call)

---

## 6. Budget & Compute Strategy

### Decision: API-Only (Pay-Per-Token)

**Rationale**: Full cost control, no subscription lock-in, exact billing visibility. The owner sets hard budget caps — the system never exceeds them.

### Cost Projections

| Configuration | Daily | Monthly | Annual |
|--------------|-------|---------|--------|
| **Current** (stateless, 9 tools, Sonnet) | $0.25 | $7.50 | $90 |
| **Phase 1** (persistent, existing 9 tools) | $0.35 | $10.50 | $126 |
| **Phase 2** (18 tools, batch operations) | $0.45 | $13.50 | $162 |
| **Phase 3** (+ dynamic strategy) | $0.50 | $15.00 | $180 |
| **Phase 4** (tiered models + caching) | $0.40 | $12.00 | $144 |
| **Phase 4 volatile month** | $0.80 | $24.00 | — |

### Built-In Cost Controls

The persistent agent enforces hard budget limits at three levels:

| Control | Default | Configurable? | What happens when hit |
|---------|---------|---------------|----------------------|
| **Per-session budget** | $0.75 | Yes (env var) | Graceful finalize — agent wraps up with no tools |
| **Per-day budget** | $3.00 | Yes (env var) | Skip remaining sessions, log warning |
| **Per-month budget** | $75.00 | Yes (env var) | Switch to Haiku-only mode for remaining days |
| **Per-turn cost check** | Pre-turn | Automatic | Stops tool loop before exceeding session budget |

These are **system-level enforcements** — the agent cannot override them. The `TokenTracker` (already implemented) handles per-turn accounting. New: daily and monthly accumulators stored in DB.

**Telegram cost alerts**: Daily summary includes API spend. Alert if daily spend > 2x normal.

### API Pricing Reference (Anthropic, as of Feb 2026)

| Model | Input | Output | Cached Input (90% off) |
|-------|-------|--------|------------------------|
| Sonnet 4.5 | $3.00/Mtok | $15.00/Mtok | $0.30/Mtok |
| Opus 4.6 | $5.00/Mtok | $25.00/Mtok | $0.50/Mtok |
| Haiku 4.5 | $0.80/Mtok | $4.00/Mtok | $0.08/Mtok |

> **Note**: Opus is much more affordable than initially assumed ($5/$25 not $15/$75). This makes Opus validation feasible for every morning session without significantly impacting budget.

### Token Budget Per Session (Phase 4, Sonnet with caching)

| Component | Tokens | First call | Cached calls (within session) |
|-----------|--------|------------|-------------------------------|
| System prompt + tools | 5,000 | $0.015 | $0.002 |
| Pinned context | 3,000 | $0.009 | $0.001 |
| Conversation history | 25,000 | $0.075 | $0.008 |
| New trigger + tools | 8,000 | $0.024 | $0.024 |
| Output (across turns) | 2,000 | $0.030 | $0.030 |
| **Session total** | **43,000** | **$0.153** | **~$0.065** |

Cache TTL is 5 minutes, auto-extends on use. During tool loops (calls every 10-30s), cache stays hot. Between sessions (2.5+ hours), cache expires — each session's first call pays full price, subsequent turns use cache. **Daily savings: ~30-35%** (3 sessions x ~6 cached turns each).

### ROI Framework

For a $99K portfolio:
- 0.5% annual improvement = $495 (covers Phase 4 compute 3x over)
- 1% annual improvement = $990
- Break-even: ~0.15% improvement pays for compute
- The question is not "can we afford the compute" but "does the AI make better decisions"

### Claude Code on VPS (Development Tool)

Claude Code v2.1.42 is installed on the VPS at `/opt/kukulkan-trade`. Uses the same `ANTHROPIC_API_KEY`. **Costs $0 when idle** — only pay-per-token when actively used.

Usage: SSH in → `cd /opt/kukulkan-trade && source .env && claude`

Purpose: Interactive development, debugging, monitoring. Not part of the automated trading pipeline. Can be used in a tmux session for convenience.

---

## 7. Risk Guardrails (Non-Negotiable)

These are **hardcoded in the RiskManager** and cannot be overridden by the agent, regardless of posture:

| Guardrail | Limit | Enforcement |
|-----------|-------|-------------|
| Max single position | 30% of portfolio | RiskManager blocks trade |
| Max total positions | 10 | RiskManager blocks trade |
| Daily loss circuit breaker | -5% | Halt all trading for the day |
| Weekly loss circuit breaker | -10% | Halt all trading for the week |
| Minimum cash | 2% | RiskManager blocks buy |
| No options/futures/margin | Hard block | Not in instrument universe |
| No round-tripping | 3-day cooldown per ticker | RiskManager blocks re-entry |
| Sector concentration | Per-sector caps (see risk_rules.py) | RiskManager blocks trade |
| All trades through RiskManager | No bypass | `execute_trade` tool enforces this |
| Large trade approval | >10% of portfolio | Telegram confirmation required |
| Posture cap | Max "Balanced" until proven | 50+ trades, >55% WR, positive alpha to unlock Aggressive |
| Human kill switch | Telegram `/stop` | Immediately halts agent |

**The agent can choose HOW to invest within these bounds but cannot remove the bounds.**

---

## 8. Success Metrics

### Quantitative (measured monthly)

| Metric | Current Baseline | Target (6 months) |
|--------|-----------------|-------------------|
| Win rate (Portfolio B) | TBD (new system) | >55% |
| Avg P&L per trade | TBD | >+1.0% |
| Alpha vs SPY | TBD | >0% consistently |
| Max drawdown | TBD | <8% |
| Tool utilization (calls/session) | 3-4 | 8-12 |
| Tickers analyzed per session | 3-4 | 15-20 |
| Multi-day thesis completion rate | 0% (impossible today) | >60% |

### Qualitative

- Claude references prior sessions naturally in reasoning
- Trades show regime-appropriate behavior (defensive in corrections, offensive in bulls)
- Position sizing reflects calibrated conviction (learned from track record)
- Cross-portfolio coordination visible (B avoids A's sector overweight)
- Weekly self-assessments show learning trajectory

---

## 9. Migration Plan

### How We Transition: Hybrid Wrapping

The persistent agent **wraps** existing components rather than replacing them. This is a key de-risking decision:

| Component | Change |
|-----------|--------|
| Orchestrator | Still runs 3x daily, still handles Portfolio A, still does risk checks |
| `_run_portfolio_b()` | Calls `PersistentAgent.on_session()` instead of fresh API call |
| AgentRunner | **Wrapped** (not deprecated): PersistentAgent delegates tool loops to it |
| ToolRegistry | **Wrapped**: existing 9 tools work immediately, new tools added incrementally |
| AgentMemoryManager | **Wrapped**: existing memory continues, conversation history supplements it |
| ClaudeAgent | **Wrapped**: `build_system_prompt()` reused for pinned context generation |
| Risk manager | Unchanged: all trades still pass through it |
| Alpaca executor | Unchanged: trades still executed the same way |
| Telegram | Enhanced: trade approval for >10% positions, agent-triggered alerts |
| API endpoints | Unchanged: dashboard reads from same DB |

**Why wrapping matters**: If PersistentAgent has a bug, we can fall back to the existing AgentRunner path immediately. No code was deleted, only wrapped.

### Rollback Plan

The current system (`use_agent_loop` flag) stays in the codebase. If the persistent agent underperforms:
1. Set `use_agent_loop = False` on the tenant
2. Bot reverts to stateless single-shot Claude calls (Phase 32 code untouched)
3. No data loss, no downtime

### Crash Recovery

The triggered process model provides inherent crash safety:
- Each session is atomic: load from DB, run, save to DB, exit
- If the process dies mid-tool-loop: partial tool results are lost, but submitted trades are tracked by Alpaca
- On next trigger: `sync_positions()` reconciles local state against Alpaca broker positions
- No orphaned connections, no zombie processes, no memory leaks
- Worst case: one missed session (out of 3 daily) — next session picks up naturally

---

## 10. Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Compute billing** | API-only (pay-per-token) | Full cost control, no subscription, exact billing |
| **Portfolio A** | Stays rule-based, Claude gets read-only access | Control group preserved, cross-portfolio coordination enabled |
| **Instruments** | Stocks + ETFs + crypto proxy only | No options, futures, or margin — smart within simple instruments |
| **Budget enforcement** | Hard caps per-session, per-day, per-month | Owner sets limits, system enforces, agent cannot override |
| **Claude Code on VPS** | Installed for development, uses same API key | $0 when idle, pay-per-token when used interactively |
| **Process model** | Triggered process (not daemon) | Crash-safe, no memory leaks, conversation in DB |
| **Migration approach** | Hybrid wrapping (PersistentAgent wraps AgentRunner) | De-risks transition, instant rollback to Phase 32 |
| **Trade approval** | Telegram confirmation for >10% of portfolio | Human-in-the-loop for large positions |
| **Posture cap** | Max "Balanced" until 50+ trades with >55% WR | Prevent overconfidence before calibration |
| **Tenant scope** | Multi-tenant from day 1 (tenant_id on conversation table) | Avoid migration 010 later |
| **Phase separation** | Persistence first (existing tools), then upgrade tools | Isolate bugs: persistence vs tool issues |

## 11. Resolved Questions (Analyst Feedback)

These questions were raised in v2.0 and resolved during analyst review:

| # | Question | Resolution |
|---|----------|------------|
| 1 | Trade approval threshold? | **10% of portfolio** via Telegram (approve/reject, 60s timeout, default reject) |
| 2 | Posture cap? | **Balanced max** until 50+ trades, >55% WR, positive alpha unlocks Aggressive |
| 3 | Budget caps? | **$0.75/session, $3.00/day, $75/month** — good defaults, configurable via env |
| 4 | Portfolio A evolution? | **Keep as control indefinitely**. No AI oversight planned — it serves as benchmark |
| 5 | Backtest? | **No** — simulating conversation persistence for historical data is too complex for the value. Use live paper trading as the test |
| 6 | Thesis time horizon? | **14-day default, 30-day max with reason** — tracked in pinned context, auto-evaluation on expiry |
| 7 | Crisis escalation? | **Sonnet with more turns** (12 instead of 8). Opus only for final trade validation. At corrected Opus pricing ($5/$25), Opus validation is affordable for all morning sessions |
| 8 | Tenant scope? | **Multi-tenant from day 1** — add `tenant_id` to conversation table in migration 009 |
| 9 | History retention? | **5 sessions full, then summaries** — but keep full JSON in DB for 30 days for debugging (compression replaces the context inclusion, not the storage) |

## 12. Analyst Technical Q&A

Questions raised by the analyst during review, with engineering responses:

**Q1: How does the "persistent agent" maintain state between sessions?**
Not a long-running daemon. Each scheduler trigger spawns a process that: loads conversation from SQLite, builds context, runs tool loop, saves session, exits. Conversation state lives in the DB, not in memory. See Section 3.1 for details.

**Q2: How do we validate that Haiku compression preserves critical trading context?**
Validation gate before deployment: compress 5 real sessions, present to Sonnet asking "what's missing?", adjust compression prompt until >95% of key facts (trades, positions, theses) are preserved. See Section 4, Phase 1C.

**Q3: Is the 40-60% cache savings estimate realistic?**
Revised to **30-35%**. Cache TTL is 5 minutes (not 1 hour as originally stated). Cache stays hot within a tool loop (calls every 10-30s) but expires between sessions (2.5+ hours apart). Savings come from ~6 cached turns per session x 3 sessions/day.

**Q4: How does execute_trade return fill results to the agent within the same tool call?**
The tool internally: (1) calls RiskManager, (2) requests Telegram approval if >10%, (3) submits to Alpaca, (4) polls for fill (up to 30s), (5) returns `{success, filled_shares, fill_price, cash_freed}` to the agent. The agent sees the result and can react immediately.

**Q5: How do we avoid deprecating working Phase 32 code?**
Hybrid wrapping: PersistentAgent imports and wraps AgentRunner, ToolRegistry, and AgentMemoryManager. These continue working inside the persistent layer. If PersistentAgent has a bug, set `use_agent_loop=False` to fall back to Phase 32 code instantly.

**Q6: What happens if the process crashes mid-session?**
Triggered process model provides inherent safety. Last completed session is intact in DB. Partial tool results are lost but submitted trades are tracked by Alpaca. `sync_positions()` on next trigger reconciles local state. Worst case: one missed session out of 3 daily.

---

## 13. Timeline

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | 1A-1B | ConversationStore + ContextManager + migration 009 |
| 2 | 1C-1D | SessionCompressor + PersistentAgent (wrapping existing tools) |
| 3 | 1 validation | Compression validation, integration tests, crash recovery tests |
| 4 | 2A-2B | Batch market tools + risk management tools |
| 5 | 2C | Execution tools + Telegram approval + fill results |
| 6 | 3A-3B | Dynamic posture + weekly playbooks |
| 7 | 3C | Conviction calibration + integration tests |
| 8 | 4A-4B | Tiered models + prompt caching |
| 9 | 4C + hardening | Portfolio A visibility + production hardening + monitoring |

**Total: 9 weeks** (conservative estimate, some phases may compress)

**Total estimated new tests**: ~300-370
**Total estimated new code**: ~3,000-4,000 lines (source) + ~4,000-5,000 lines (tests)
