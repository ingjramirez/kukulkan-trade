# Kukulkan Trade — State of the Art & Agentic Roadmap

> Converged plan from architectural review between the bot agent, project analyst, and owner.
> Last updated: 2026-02-13

---

## 1. Current State of the Art

### What We Have

Kukulkan is a **scheduled batch trading system** with an LLM decision step. The architecture:

```
APScheduler (3x daily)
  └─ Orchestrator (9-step pipeline, per tenant)
       ├─ Step 1: Sync portfolios with broker
       ├─ Step 2: Fetch market data (yfinance, 70 tickers, 1yr OHLCV)
       ├─ Step 3: Fetch macro (yield curve, VIX) + classify regime
       ├─ Step 4: Portfolio A — deterministic momentum (top 1 ETF)
       ├─ Step 5: Fetch news + ChromaDB historical context
       ├─ Step 6: Portfolio B — single Claude call (mega-prompt → JSON)
       ├─ Step 7: Risk filter (sector concentration, sizing, circuit breakers)
       ├─ Step 8: Execute trades (Alpaca)
       ├─ Step 9: Snapshot + reconciliation
       └─ Step 10: Telegram notifications
```

### How Claude Participates Today

Claude receives a **pre-packaged prompt** assembled from 8+ layers:

| Layer | Source | Tokens (~) |
|-------|--------|------------|
| Base identity | Hardcoded | 200 |
| Strategy directive | conservative/standard/aggressive | 400 |
| Session directive | Morning/Midday/Closing | 100 |
| Regime summary | RegimeClassifier output | 50 |
| Performance stats | PerformanceTracker + SPY benchmark | 150 |
| Memory context | 3-tier (recent decisions, weekly, notes) | 500 |
| Trailing stops + earnings + watchlist | DB lookups | 200 |
| Market data + news (user prompt) | yfinance + ChromaDB + news APIs | 1500 |

Claude responds with a single JSON object: `trades[]`, `reasoning`, `regime_assessment`, `memory_notes[]`, `suggested_tickers[]`, `watchlist_updates[]`.

The orchestrator parses the JSON, applies risk filters, executes trades, and stores the decision. **Claude has no further involvement until the next scheduled run.**

### What's Genuinely Smart

- **Risk manager is rule-based and independent** — Claude can't override position limits, sector concentration, or circuit breakers. Correct for a trading system.
- **Portfolio A is purely algorithmic** — Momentum strategy needs no AI. Clean separation.
- **Trailing stops are mechanical** — Once set, they execute without AI input.
- **Memory system gives continuity** — Claude sees its recent decisions, weekly lessons, and persistent notes.
- **Regime classifier adapts context** — Strategy directives shift based on market state.
- **Multi-tenant isolation** — Each tenant gets its own universe, allocations, and credentials.

### What's Not Actually Smart

| What it looks like | What it actually is |
|---|---|
| "AI chooses strategy based on regime" | Hardcoded rules inject different text into prompt |
| "AI learns from past decisions" | Narrative memory with no quantitative feedback |
| "AI discovers new tickers" | Claude suggests, human approves via Telegram |
| "3 daily sessions adapt to market" | Same pipeline runs 3x with different session text |
| "Conviction-based sizing" | Claude says high/medium/low, multiplied by fixed constants |
| "AI manages portfolio" | Claude fills in a JSON template within hard constraints |

### The Core Limitation

Claude is a **function in the pipeline**, not an agent. It:
- Cannot decide **when** to think
- Cannot decide **what data** to look at
- Cannot decide **whether** to act (the pipeline always asks for trades)
- Cannot iterate (no multi-step reasoning, no "let me check something else")
- Cannot learn quantitatively (no "my tech calls have a 60% hit rate in bull regimes")
- Cannot react to events between scheduled runs

---

## 2. Agentic Roadmap (Converged Plan)

### Sequencing: A → B → D → C

```
Phase A: Outcome Feedback Loop ──── 1 week dev ────────────────────┐
Phase B: Hybrid Agentic Tool Use ── 2-3 weeks dev + 6 weeks eval ─┤
Phase D: Strategy Evolution ─────── 1-2 weeks dev (after B data) ──┤
Phase C: Event Triggers ─────────── Deferred (revisit post-D) ─────┘
```

Phase C (event triggers) is intentionally deferred — we're paper trading, intraday reactivity is premature. Phase D moves ahead of C because it only requires Phase A data + a few weeks of Phase B results.

---

### Phase A: Outcome Feedback Loop

**Goal**: Claude learns whether its decisions were good — with real numbers, not vibes.

**Timeline**: ~1 week development, ~25 new tests.

**Model**: Opus 4.6 (existing, no cost change — it's the same single Claude call).

#### What Changes

**1. Trade outcome injection into prompts**

After each run, compute P&L for positions Claude entered in the last 1-7 days. On the next run, inject a "Decision Review" section:

```
## Recent Decision Outcomes
- 3 days ago: BUY NVDA @ $850 (thesis: "AI infrastructure demand")
  → Now $820 (-3.5%), sector -2.0%. Underperforming sector by -1.5%.
- 5 days ago: SELL XLP @ $78 (thesis: "rotate out of defensives in bull")
  → Now $76 (-2.6%). Good call — avoided further decline.
```

**2. Server-side win rate computation by category**

Compute on the server (not by Claude) and inject as a stats block:

```
## Your Track Record
- Tech trades: 4W-6L (40%)
- Defensive trades: 7W-2L (78%)
- High-conviction trades: 5W-3L (63%)
- BULL regime trades: 8W-4L (67%)
- CONSOLIDATION regime trades: 2W-5L (29%)
- Morning session entries: 6W-3L (67%)
- Midday session entries: 3W-5L (38%)
```

Claude draws its own conclusions from factual data. No confidence floats — those are premature without calibration data.

**3. Weekly compaction upgrade**

Change the existing weekly compaction prompt from "summarize your decisions this week" to "**evaluate** your decisions this week — what worked, what didn't, what patterns do you see in the track record data?"

**4. Decision Quality metric (new)**

For each trade, track whether the price moved favorably within 1, 3, and 5 days after entry. This is more granular than win rate (which depends on exit timing) and directly measures entry quality.

```
Decision Quality = % of BUY entries where price was higher 3 days later
                 + % of SELL exits where price was lower 3 days later
```

#### Why First

Without feedback, every other improvement is cosmetic. An agent that can't learn from outcomes is just a chatbot with a cron job.

#### Key Output

Claude should be able to say "I notice my tech calls underperform in consolidation regimes" based on **actual data**, not narrative memory.

#### Files Affected

- `src/orchestrator.py` — add outcome computation + injection
- `src/agent/claude_agent.py` — new "Decision Review" + "Track Record" prompt sections
- `src/agent/memory.py` — upgrade weekly compaction prompt
- `src/storage/database.py` — trade outcome queries (P&L by category)
- New migration — decision_quality tracking columns or table

---

### Phase B: Hybrid Agentic Tool Use

**Goal**: Claude drives deeper analysis when it wants to, while keeping the proven mega-prompt as the baseline.

**Timeline**: ~2-3 weeks development, ~60 new tests, then 6-week evaluation (B.0→B.3).

**Model**: Sonnet 4.5 (upgrade to Sonnet 5 when available — same API, drop-in). Sonnet is ~15x cheaper than Opus per token. With tool use adding 4-8 round trips per session, model cost is the primary constraint.

#### The Hybrid Architecture

Instead of replacing the mega-prompt with pure tool use, **keep the mega-prompt as the seed analysis and add an optional tool-use investigation phase**:

```
Phase 1: Mega-prompt (existing pipeline, unchanged)
  → Claude receives full context: portfolio, regime, market data, news
  → Claude drafts initial analysis + proposed trades

Phase 2: Tool use (new, optional)
  → "Based on your analysis, you have tools available.
     Is there anything you want to investigate before finalizing?"
  → Claude can:
     - Deep-dive a ticker: get_price_and_technicals("NVDA", days=60)
     - Check a hunch: search_news("semiconductor tariffs", days=3)
     - Verify exposure: get_portfolio_summary()
     - Finalize: propose_trades(trades_json)
  → OR skip Phase 2 entirely if mega-prompt analysis was sufficient
```

**Why hybrid instead of full replacement**:
- **Cheaper**: Most sessions, Claude calls 1-2 tools (or none). The mega-prompt does 80% of the work.
- **More deterministic**: Base analysis is the proven pipeline. Tools add optional depth.
- **Easier to evaluate**: Compare "mega-prompt only" vs "mega-prompt + tools" directly.
- **Graceful degradation**: If tools hit budget cap or timeout, you still have the mega-prompt analysis.

#### Tool Set (7 tools for v1)

```python
tools = [
    # Portfolio awareness
    get_current_positions(),               # "What am I holding?"
    get_position_pnl(ticker),             # "How's my NVDA doing since entry?"
    get_portfolio_summary(),              # Sector exposure, cash, utilization

    # Market data (merged — saves round trips)
    get_price_and_technicals(ticker, days=30),  # Price + RSI/MACD/momentum
    get_market_context(),                  # Regime + VIX + macro + earnings in one call

    # External data
    search_news(query, days),             # "Any news about semiconductors?"

    # Actions
    propose_trades(trades_json),          # "I want to buy NVDA and sell XLP"
]
```

**Why 7, not 12**: Each tool call costs tokens and latency. Merged tools (`get_price_and_technicals` instead of separate price + indicators) reduce round trips without losing capability. Watchlist and memory note updates remain in the response JSON (like today) — they don't need to be tools in v1.

**Add in v1.5** (after core loop is stable): `update_watchlist()`, `save_memory_note()` as tools.

#### Orchestrator Integration

The AgentRunner replaces **only Step 6** — the Claude call. Everything else is unchanged:

```python
# orchestrator.py, step 6
if tenant.use_agent_loop:
    trades = await self._run_portfolio_b_agentic(context)  # AgentRunner (hybrid)
else:
    trades = await self._run_portfolio_b(context)           # existing mega-prompt
```

```
Steps 1-5: IDENTICAL for both pipelines
  ├─ Market data, macro, regime, news — all fetched the same way
  │
Step 6: BRANCHING POINT (tenant-level flag)
  ├─ use_agent_loop=false → mega-prompt (build prompt → single call → parse JSON)
  ├─ use_agent_loop=true  → AgentRunner (mega-prompt seed → tool loop → propose trades)
  │
Steps 7-10: IDENTICAL for both pipelines
  ├─ Risk filter, execution, snapshots, notifications
```

Both pipelines coexist in the codebase. Rollback is instant: flip a boolean, next run uses mega-prompt. No code changes, no deploy.

#### Budget Controls

| Control | Value | Notes |
|---------|-------|-------|
| Max turns per session | 8 (start at 6 for B.1) | Hard stop — graceful exit with whatever trades were proposed |
| Cost cap per session | $0.30 Sonnet | Track cumulative tokens, abort if exceeded |
| Model | Sonnet 4.5 → Sonnet 5 (drop-in) | Opus only as fallback if Sonnet quality is insufficient |
| Tool call logging | Every call logged | Tool name, input, tokens consumed, influenced_decision (bool) |
| Weekly cost report | Telegram | Total API cost, cost per session, cost per trade |

#### BYOK (Bring Your Own Key)

- **Phase A**: Owner's Anthropic key (cost doesn't change)
- **Phase B onward**: Add `anthropic_api_key_enc` to TenantRow. Non-owner tenants must provide their own key to use agent loop. Tenants without their own key stay on mega-prompt (cheaper, owner subsidized).

#### Files Affected

- **Create**: `src/agent/agent_runner.py`, `src/agent/tools/` directory (portfolio.py, market.py, news.py, actions.py)
- **Modify**: `src/orchestrator.py` (step 6 branching), `src/storage/models.py` (use_agent_loop on TenantRow), new migration
- **Unchanged**: `src/agent/claude_agent.py` (mega-prompt path stays as-is)

---

### Phase B Evaluation Plan (4 Sub-Phases)

Each sub-phase is 2 weeks with a go/no-go checkpoint.

#### B.0 — Shadow Mode (Weeks 1-2)

**Setup**:
- Mega-prompt pipeline continues running and executing trades (unchanged)
- Agent loop runs in parallel after mega-prompt, same data, **trades logged only** (not executed)
- Both see the same portfolio state, market data, regime

**What we measure**:
- Decision overlap: how often do agent and mega-prompt agree?
- Decision divergence: when they disagree, who would have been right?
- Agent API cost per session
- Tool call patterns: which tools does it use? How many turns?
- Failure rate: does the agent hit max_turns without proposing trades?

**Go/No-Go**:
- Go to B.1 if: agent decisions would have matched or beaten mega-prompt in >50% of divergent cases AND cost < $0.30/session
- Stay in B.0 if: agent consistently worse or costs too high — adjust tools, turns, prompt
- Abort if: agent produces incoherent decisions or fails >20% of sessions

**Cost**: ~$3-5/day extra for 2 weeks = $42-70 total. Acceptable for confidence.

#### B.1 — Agent Live, Limited (Weeks 3-4)

**Setup**:
- Agent loop replaces mega-prompt for **owner's tenant only** (other tenants stay on mega-prompt as control)
- Max_turns=6, 5 tools (drop search_news, keep propose_trades as response JSON)
- Budget cap: $0.25/session

**What we measure**:
- Real alpha vs control tenant's mega-prompt (same market, different pipeline)
- Tool efficiency: % of tool calls that influenced final decision (target: >60%)
- Average turns used (expect 4-5 out of 6)
- Cost per session (target: <$0.20 with Sonnet)

**Go/No-Go**:
- Go to B.2 if: alpha >= mega-prompt AND cost < $0.25/session AND tool efficiency > 50%
- Rollback to mega-prompt if: alpha < mega-prompt by >1% OR cost > $0.40/session
- Adjust and retry if: mixed results

#### B.2 — Full Tools (Weeks 5-6)

**Setup**:
- All 7 tools enabled
- Max_turns=8
- Add `update_watchlist` and `save_memory_note` as tools (not JSON response)
- Budget cap: $0.35/session

**What we measure**:
- Do extra tools (news, watchlist, memory) improve decisions?
- Does increasing max_turns from 6 to 8 add value or just cost?
- Compare B.2 metrics vs B.1: did expanding scope help?

**Go/No-Go**:
- Lock in B.2 config if: measurable improvement over B.1
- Revert to B.1 config if: more tools don't improve results (simpler is better)
- Either way, this becomes the "production config" going forward

#### B.3 — Sonnet 5 Upgrade (When Available)

**Setup**:
- Drop-in model replacement (same API)
- Same tools, same max_turns, same budget cap
- Run for 2 weeks

**What we measure**:
- Quality improvement: better reasoning, fewer bad trades?
- Cost change: same price per token expected, but token usage may change
- If quality improves significantly: consider increasing max_turns to 10-12

---

### Evaluation Metrics

#### Level 1: Does it generate alpha?

| Metric | How | Target |
|--------|-----|--------|
| Return vs SPY | Weekly comparison | Positive alpha |
| Win rate | Total + by category (sector, regime, session, conviction) | > mega-prompt baseline |
| Decision Quality | % of entries where price moved favorably within 3 days | > 55% |
| Sharpe ratio | Risk-adjusted return | > 0.5 |
| Max drawdown | Worst peak-to-trough | < mega-prompt baseline |
| Cash utilization | % of capital deployed | 40-80% (not sitting idle, not over-leveraged) |

#### Level 2: Does it use tools intelligently?

| Metric | How | Target |
|--------|-----|--------|
| Tool efficiency | % of tool calls that influenced final decision | > 60% |
| Exploration depth | Average tool calls per session | 3-6 (not 0, not 15) |
| Budget utilization | % of session cost cap used | 40-80% |

Log every tool call:
```json
{
  "session_id": "...",
  "turn": 3,
  "tool": "get_price_and_technicals",
  "input": {"ticker": "NVDA", "days": 30},
  "tokens_consumed": 850,
  "influenced_decision": true
}
```

#### Level 3: Is it worth the cost?

```
Cost-Adjusted Alpha = (Agent Alpha - Mega-Prompt Alpha) - (Agent API Cost - Mega-Prompt API Cost)
```

Monthly API budget ceiling: **$100/month** across all tenants. If costs exceed this, reduce max_turns or fall back to hybrid with fewer tools.

---

### Phase D: Strategy Evolution (Moved Up)

**Goal**: Claude evolves its own trading rules based on what works.

**Timeline**: ~1-2 weeks development, ~30 new tests. Requires 4+ weeks of Phase A+B data.

**Model**: Sonnet (weekly review call).

#### What Changes

**1. Weekly strategy review** (Sunday, after memory compaction)

Claude receives its full performance history for the past 4 weeks — computed server-side, not by Claude:
- Win rate by regime, sector, session, conviction level
- Average return by trade type
- Drawdown attribution (which positions caused biggest losses)
- Comparison vs SPY benchmark
- **Sample sizes for every category** — Claude must see the n

**2. Structured adjustment proposals**

```json
{
  "adjustments": [
    {
      "rule": "Increase defensive allocation in CONSOLIDATION regime from 30% to 40%",
      "rationale": "My tech picks during consolidation: 29% win rate (n=17) vs 67% in BULL (n=24)",
      "category": "regime_allocation"
    }
  ]
}
```

**3. Minimum sample thresholds**

| Threshold | Value | Rationale |
|-----------|-------|-----------|
| Min trades per category for rule proposal | n >= 15 | Below this, noise dominates signal |
| Min data window | >= 2 weeks | Single bad week shouldn't trigger rule changes |
| Sample size always visible | Required | Claude must see "n=17" alongside "29% win rate" |

With ~2 tenants, 3 sessions/day, ~2-5 trades/session, expect ~40-60 trades/week. After 4 weeks: ~200 trades. Enough for aggregate win rates. NOT enough for hyper-specific categories like "tech calls in consolidation during midday" (might be 3 data points).

**4. Human-in-the-loop approval**

Adjustments sent via Telegram for approval (same flow as ticker discovery):
- Owner reviews proposed rule + rationale + sample size
- Approve → stored as strategy override in database
- Reject → logged, not applied
- Active overrides injected into strategy directives on each run

**5. Strategy overrides table**

```sql
CREATE TABLE strategy_overrides (
    id INTEGER PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL,
    rule TEXT NOT NULL,
    rationale TEXT NOT NULL,
    category VARCHAR(50) NOT NULL,
    sample_size INTEGER NOT NULL,
    proposed_at DATETIME NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'proposed',  -- proposed/approved/rejected/expired
    approved_at DATETIME,
    expires_at DATETIME  -- auto-expire after 90 days, force re-evaluation
);
```

#### Files Affected

- **Create**: `src/agent/strategy_evolution.py`, new migration (strategy_overrides table)
- **Modify**: `src/agent/strategy_directives.py` (override injection), `src/main.py` (Sunday review job), `src/storage/database.py` (override CRUD), `src/storage/models.py` (StrategyOverrideRow)

---

### Phase C: Event Triggers (Deferred)

**Goal**: The bot reacts to market events, not just cron schedules.

**Status**: Intentionally deferred. Revisit when:
- Phase B has been running for 4+ weeks
- There are documented cases where the bot missed an important intraday event
- The system moves toward real money trading

**Interim measure**: If intraday reactivity is wanted sooner, tighten trailing stop check frequency from 3x/day to every 30 minutes — a one-line scheduler change, not a new subsystem.

**When implemented**, the architecture would be:

```
EventMonitor (15-min polling, lightweight)
  ├─ Price alerts: held position moves > 3% intraday
  ├─ Volatility spike: VIX jumps > 15% in a day
  ├─ News trigger: high-relevance article about held ticker
  ├─ Earnings surprise: post-earnings move > 5%
  └─ Trailing stop proximity: price within 1% of stop level
       └─ if trigger fires → mini agent session (smaller context, focused question)
```

---

## 3. What Stays the Same

These are **correct design decisions** that should NOT change:

- **Risk manager is rule-based** — Never let AI override risk limits
- **Trailing stops are mechanical** — Once set, they execute deterministically
- **Portfolio A is algorithmic** — Momentum doesn't need AI
- **Multi-tenant isolation** — Each tenant's data stays separate
- **Execution via Alpaca SDK** — No need to change the execution layer
- **SQLite + async** — Storage layer is fine
- **APScheduler for scheduling** — Event triggers augment, not replace, the schedule
- **Mega-prompt pipeline** — Stays in codebase as fallback, configurable per tenant

---

## 4. Impact & Effort Matrix

| Phase | Impact | Effort | API Cost Delta | Dependencies |
|-------|--------|--------|----------------|--------------|
| **A: Feedback Loop** | High | 1 week | ~$0/day (prompt change) | None |
| **B: Hybrid Tool Use** | Very High | 2-3 weeks + 6 weeks eval | +$1-3/day (Sonnet) | Phase A |
| **D: Strategy Evolution** | High (long-term) | 1-2 weeks | +$0.10/week (review) | Phase A + 4 weeks B data |
| **C: Event Triggers** | Medium | 2 weeks | +$0.20-1/day | Phases B+D, real money |

**Monthly API budget ceiling**: $100/month across all tenants.

---

## 5. The Philosophical Shift

```
TODAY:    Orchestrator tells Claude what to think about → Claude fills in a template
PHASE A:  Orchestrator tells Claude what to think about → Claude reflects on outcomes
PHASE B:  Mega-prompt seeds analysis → Claude investigates deeper with tools
PHASE D:  Claude reviews its own performance → Claude proposes strategy changes
PHASE C:  Events wake Claude up → Claude decides if action is needed
```

The mega-prompt pipeline is the **proven baseline that never goes away**. The agentic layers add optional depth, learning, and autonomy on top — controlled by per-tenant flags, budget caps, and human-in-the-loop approval for strategy changes.

---

## 6. Rollback Plan

At ANY point, revert to mega-prompt by setting `use_agent_loop: false` per tenant. Both pipelines coexist permanently. The mega-prompt code is never deleted — it's the fallback.

```python
# TenantRow
use_agent_loop: bool = False  # default: proven mega-prompt pipeline
```
