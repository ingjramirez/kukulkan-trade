# AI Agent & Tools

Machine-readable context for Claude. Covers the agent subsystem, tool-use loop, memory, discovery, and strategy directives.

## Key Files

| File | Purpose |
|------|---------|
| `src/agent/claude_agent.py` | ClaudeAgent class, system/user prompt builders, single-shot analyze() |
| `src/agent/agent_runner.py` | AgentRunner tool-use loop (Sonnet 4.5, 8 turns, $0.50 budget) |
| `src/agent/token_tracker.py` | Per-model pricing, budget enforcement |
| `src/agent/memory.py` | AgentMemoryManager: short-term, weekly compaction, agent notes |
| `src/agent/strategy_directives.py` | CONSERVATIVE/STANDARD/AGGRESSIVE + SESSION_DIRECTIVES |
| `src/agent/complexity_detector.py` | ComplexityDetector: 6-signal scoring for model escalation |
| `src/agent/ticker_discovery.py` | TickerDiscovery: validation, propose, expire |
| `src/agent/persistent_agent.py` | PersistentAgent wrapping AgentRunner with conversation persistence |
| `src/agent/conversation_store.py` | SQLite-backed conversation save/load/compress/cleanup |
| `src/agent/context_manager.py` | Context building: system prompt, messages, trigger messages, pinned context |
| `src/agent/session_compressor.py` | Haiku compression + Sonnet validation for old sessions |
| `src/agent/session_profiles.py` | SessionProfile enum (FULL/LIGHT/CRISIS/REVIEW/BUDGET_SAVING) |
| `src/agent/haiku_scanner.py` | HaikuScanner for fast market triage (ScanResult: ROUTINE/INVESTIGATE/URGENT) |
| `src/agent/opus_validator.py` | OpusValidator for trade review (ValidationResult: approved/concerns) |
| `src/agent/tiered_runner.py` | TieredModelRunner orchestrating Haiku→Sonnet→Opus per session profile |
| `src/agent/budget_tracker.py` | BudgetTracker (daily $3 / monthly $75 caps), BudgetStatus dataclass |
| `src/agent/posture.py` | PostureLevel enum, PostureLimits, PostureManager (aggressive gate) |
| `src/agent/tools/__init__.py` | ToolRegistry + ToolDefinition dataclass |
| `src/agent/tools/portfolio.py` | 6 tools + 3 legacy aliases |
| `src/agent/tools/market.py` | 5 tools + 2 legacy aliases |
| `src/agent/tools/news.py` | 3 tools: search_news, search_historical_news, get_portfolio_a_status |
| `src/agent/tools/actions.py` | 6 tools + 2 legacy aliases + ActionState + declare_posture |
| `config/strategies.py` | PortfolioAConfig, PortfolioBConfig frozen dataclasses |

## ClaudeAgent (`src/agent/claude_agent.py`)

```python
class ClaudeAgent:
    def __init__(self, api_key: str | None = None, model: str = PORTFOLIO_B.model) -> None
    @property
    def client(self) -> anthropic.Anthropic  # lazy-initialized

    def analyze(
        self, analysis_date, cash, total_value, positions, prices, tickers,
        indicators, recent_trades, regime=None, yield_curve=None, vix=None,
        news_context="", interesting_tickers=None, closes_df=None,
        model_override=None, system_prompt=None,
    ) -> dict  # {regime_assessment, reasoning, trades, risk_notes, _raw, _tokens_used, _model}

    def generate_daily_commentary(self, analysis_date, portfolio_a_summary, portfolio_b_summary, regime=None) -> str
```

### Module-Level Functions

```python
def build_system_prompt(
    performance_stats=None, memory_context=None, strategy_mode="conservative",
    session="", regime_summary=None, portfolio_allocation=None, universe_size=None,
    trailing_stops_context=None, earnings_context=None, watchlist_context=None,
    decision_review=None, track_record=None,
) -> str
# Assembly order: Base -> Decision Framework -> Hard Rules -> Regime -> Session ->
# Strategy -> Performance -> Decision Review -> Track Record -> Memory ->
# Trailing Stops -> Earnings -> Watchlist

def build_user_message(
    analysis_date, cash, total_value, positions, prices, tickers, indicators,
    recent_trades, regime=None, yield_curve=None, vix=None, news_context="",
    interesting_tickers=None, closes_df=None,
) -> str
# Shared by BOTH single-shot and agentic paths

def build_positions_text(positions) -> str
def build_price_table(tickers, prices) -> str
def build_indicators_table(tickers, indicators) -> str
def build_macro_context(yield_curve, vix) -> str
def build_compact_price_summary(tickers, prices) -> str   # 75% smaller
def build_compact_indicators(tickers, indicators) -> str   # 75% smaller
```

### Response JSON Format

```json
{
  "regime_assessment": "string",
  "reasoning": "string",
  "trades": [{"ticker": "str", "side": "BUY|SELL", "weight": 0.0-0.30, "conviction": "high|medium|low", "reason": "str"}],
  "risk_notes": "string",
  "suggested_tickers": [{"ticker": "str", "rationale": "str"}],
  "memory_notes": [{"key": "str", "content": "str"}],
  "watchlist_updates": [{"action": "add|remove", "ticker": "str", "reason": "str", "conviction": "str", "target_entry": number}]
}
```

## AgentRunner (`src/agent/agent_runner.py`)

```python
@dataclass
class ToolCallLog:
    turn: int; tool_name: str; tool_input: dict; tool_output_preview: str; success: bool; error: str | None = None

@dataclass
class AgentRunResult:
    response: dict; tool_calls: list[ToolCallLog]; turns: int = 0
    token_tracker: TokenTracker; raw_messages: list[dict]

class AgentRunner:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6", max_turns: int = 8, max_cost_usd: float = 0.50) -> None
    @property
    def registry(self) -> ToolRegistry
    async def run(self, system_prompt: str, user_message: str, model_override: str | None = None) -> AgentRunResult
```

### Two-Phase Flow (orchestrator `_run_portfolio_b`)

1. **Phase 1 (SEED):** Single-shot `_agent.analyze()` with complexity-routed model
2. **Phase 2 (INVESTIGATE):** If `use_agent_loop=True`, AgentRunner tool-use loop (always Sonnet 4.5)
3. **Merge:** Investigation overrides seed trades; falls back to seed on timeout/error

### Stop Conditions

- `stop_reason == "end_turn"` -> model done, parse response
- `stop_reason == "tool_use"` -> execute tools, append results, loop
- Budget exceeded -> graceful finalize (force text output, no tools)
- Max turns reached -> finalize

## TokenTracker (`src/agent/token_tracker.py`)

```python
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.0, 15.0),   # (input_per_mtok, output_per_mtok)
    "claude-opus-4-6": (5.0, 25.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}

# Prompt caching economics
CACHE_WRITE_MULTIPLIER = 1.25   # Cache writes cost 1.25x base input price
CACHE_READ_MULTIPLIER = 0.10    # Cache hits cost 0.10x base input price (90% savings)

@dataclass
class TokenTracker:
    session_budget_usd: float = 0.50
    def record(self, model: str, input_tokens: int, output_tokens: int, turn: int,
               cache_creation_tokens: int = 0, cache_read_tokens: int = 0) -> None
    def summary(self) -> dict  # includes cache_savings_usd
    @property
    def budget_exceeded(self) -> bool
    @property
    def cache_savings_usd(self) -> float  # full_price - cached_price for all cache_read_tokens
```

### Prompt Caching

System prompt blocks use `cache_control: {"type": "ephemeral"}` markers. Cost formula per call:

```
cost = (input * base + cache_write * base * 1.25 + cache_read * base * 0.10 + output * output_price) / 1M
```

Cached system prompt via `build_cached_system_prompt()` in `claude_agent.py`.

## Strategy Directives (`src/agent/strategy_directives.py`)

| Strategy | Key Rules |
|----------|-----------|
| `conservative` | Min 40% cash/defensive, max 50% equities, max 10% single pos, TP +8-10%, SL -5% |
| `standard` | 20-30% cash buffer, 8-12 positions, max 15% single pos, TP +12-15%, SL -7% |
| `aggressive` | 80-95% invested, 5-6 concentrated, max 25% single pos, TP +20%, SL -10% |

```python
STRATEGY_MAP: dict[str, str]       # "conservative" -> directive text
STRATEGY_LABELS: dict[str, str]    # "conservative" -> "Conservative"
SESSION_DIRECTIVES: dict[str, str] # "Morning" -> "Post-open assessment..."
```

## Tool System

### ToolRegistry (`src/agent/tools/__init__.py`)

```python
class ToolRegistry:
    def register(self, name: str, description: str, input_schema: dict, handler: Callable) -> None
    def get_tool_definitions(self) -> list[dict]  # Anthropic API format
    async def execute(self, name: str, arguments: dict) -> Any  # raises KeyError if not found
```

**Important:** `execute()` does NOT catch errors. Errors propagate to `AgentRunner._execute_tool()` which logs them as `success=False`.

### 23 Tools (4 modules)

**Portfolio tools** (`portfolio.py`):

| Tool | Input | Returns |
|-|-|-|
| `get_portfolio_state` | `{}` | positions, cash, P&L, sector exposure, instrument_type |
| `get_position_detail` | `{ticker}` | P&L, trailing stop, days held |
| `get_portfolio_performance` | `{}` | returns, drawdown, Sharpe ratio |
| `get_historical_trades` | `{days?}` | recent trade log |
| `get_correlation_check` | `{}` | position correlation matrix |
| `get_risk_assessment` | `{}` | risk metrics, inverse_exposure |
| `get_portfolio_a_history` | `{days?}` | Portfolio A trades (read-only) |
| `list_discovered_tickers` | `{status?}` | past discoveries with status, approval rate, source |

**Market tools** (`market.py`):

| Tool | Input | Returns |
|-|-|-|
| `get_batch_technicals` | `{tickers}` | RSI, MACD, SMA for batch, instrument_type |
| `get_sector_heatmap` | `{}` | sector performance + RSI |
| `get_market_overview` | `{}` | regime, VIX, yield curve, breadth |
| `get_earnings_calendar` | `{days?}` | upcoming earnings for held tickers |
| `search_ticker_info` | `{ticker}` | yfinance lookup: price, cap, volume, RSI, sector, minimums check |

**News tools** (`news.py`):

| Tool | Input | Returns |
|-|-|-|
| `search_news` | `{ticker?}` | today's news context |
| `search_historical_news` | `{ticker, days?}` | ChromaDB vector search |
| `get_portfolio_a_status` | `{}` | Portfolio A positions (read-only) |

**Action tools** (`actions.py`):

| Tool | Input | Returns |
|-|-|-|
| `execute_trade` | `{ticker, side, weight, conviction, reason}` | accumulates |
| `set_trailing_stop` | `{ticker, trail_pct}` | override default stop |
| `get_order_status` | `{ticker?}` | current order statuses |
| `save_observation` | `{key, content}` | accumulates memory note |
| `update_watchlist` | `{updates}` | accumulates |
| `declare_posture` | `{posture, reason}` | sets session posture |
| `discover_ticker` | `{ticker, reason, conviction?, sector_rationale?}` | validate + propose for owner approval |

Legacy aliases preserved: `get_current_positions`, `get_position_pnl`, `get_portfolio_summary`, `get_price_and_technicals`, `get_market_context`, `propose_trades`, `save_memory_note`.

### ActionState (`src/agent/tools/actions.py`)

```python
@dataclass
class ActionState:
    proposed_trades: list[dict]; watchlist_updates: list[dict]; memory_notes: list[dict]
    executed_trades: list[dict]; trailing_stop_requests: list[dict]
    discovery_proposals: list[dict]  # Phase 40: tool-based ticker discoveries
    declared_posture: str | None
    def get_accumulated_state(self) -> dict
    def reset(self) -> None
```

Per-run isolation: one ActionState per agent session, accumulated across multiple tool calls.

### Registration Functions

```python
def register_portfolio_tools(registry, db, tenant_id, current_prices, closes=None, held_tickers=None) -> None
def register_market_tools(registry, closes, vix=None, yield_curve=None, regime=None, db=None, held_tickers=None, tenant_id="default") -> None
def register_news_tools(registry, news_context, news_fetcher=None, current_prices=None) -> None
def register_action_tools(registry, state, db=None, tenant_id="default", ticker_discovery=None) -> None
```

### Discovery Tool Flow (Phase 40)

```
Agent mid-session → search_ticker_info("ANET") → yfinance lookup
→ "Meets minimums, fits thesis" → discover_ticker("ANET", "reason", "high")
→ Validates + saves to DB as "proposed" → returns immediately to agent
→ After agent loop: orchestrator sends Telegram approval → approved/rejected
→ Approved tickers enter universe on next session
```

Passive `suggested_tickers` JSON path preserved as fallback. System prompt tells agent to prefer tools.

## Persistent Agent (`src/agent/persistent_agent.py`)

```python
class PersistentAgent:
    def __init__(self, api_key, db, tenant_id, model=None, max_turns=8, max_cost_usd=0.50)
    async def run_session(self, system_prompt, user_message, trigger_type="scheduled",
                          model_override=None) -> PersistentRunResult
```

Uses ConversationStore for SQLite persistence, ContextManager for prompt assembly, SessionCompressor for old session compaction (Haiku compress + Sonnet validate). Orchestrator routing: `use_persistent_agent` > `use_agent_loop` > single-shot.

## Tiered Model Runner (`src/agent/tiered_runner.py`)

```python
class TieredModelRunner:
    def __init__(self, api_key, db, tenant_id, ...)
    async def run(self, system_prompt, user_message, ...) -> TieredRunResult
```

Flow per SessionProfile:
- **FULL**: Haiku scan → Sonnet investigate → Opus validate trades
- **LIGHT + ROUTINE**: Haiku scan only ($0.002), skip investigation
- **LIGHT + INVESTIGATE/URGENT**: Haiku scan → Sonnet investigate
- **CRISIS**: Always full investigation
- **BUDGET_SAVING**: Haiku scan only

### Session Profiles (`src/agent/session_profiles.py`)

```python
class SessionProfile(Enum):
    FULL = "full"           # Morning session, complex market
    LIGHT = "light"         # Midday/Closing, routine
    CRISIS = "crisis"       # VIX > 30 or major regime change
    REVIEW = "review"       # Weekend review
    BUDGET_SAVING = "budget_saving"  # Budget > 80% spent

def get_session_profile(session, vix, regime_changed, budget_pct_used) -> SessionProfile
```

### Budget Tracker (`src/agent/budget_tracker.py`)

Daily $3 / monthly $75 caps (configurable via `AGENT_DAILY_BUDGET`, `AGENT_MONTHLY_BUDGET`). When monthly > 80%, forces `BUDGET_SAVING` profile (Haiku only).

## Posture System (`src/agent/posture.py`)

```python
class PostureLevel(Enum):
    DEFENSIVE = "defensive"; CAUTIOUS = "cautious"; NEUTRAL = "neutral"
    OPPORTUNISTIC = "opportunistic"; AGGRESSIVE = "aggressive"; CRISIS = "crisis"

class PostureManager:
    def resolve_posture(self, declared, track_record) -> tuple[PostureLevel, PostureLimits]
```

Aggressive gate: requires 50+ trades, >55% win rate, positive alpha. `posture_limits` passed to `RiskManager.check_pre_trade()` — can only tighten, never loosen.

## AgentMemoryManager (`src/agent/memory.py`)

```python
MAX_SHORT_TERM = 3; MAX_WEEKLY_SUMMARIES = 4; MAX_AGENT_NOTES = 10

class AgentMemoryManager:
    def build_memory_prompt(self, memories: dict) -> str  # keys: short_term, weekly_summary, agent_note
    async def save_short_term(self, db, analysis_date, response, tenant_id="default") -> None
    async def save_agent_notes(self, db, notes: list[dict], tenant_id="default") -> None
    async def run_weekly_compaction(self, db, agent, tenant_id="default", outcome_summary=None, track_record_text=None) -> None
```

Weekly compaction uses Haiku to evaluate past week's trades. Stores as `weekly_summary` with key like `week_2026-07`.

## ComplexityDetector (`src/agent/complexity_detector.py`)

```python
class ComplexityDetector:
    def __init__(self, threshold: int | None = None) -> None  # default: PORTFOLIO_B.escalation_threshold
    def evaluate(self, closes, positions, total_value, peak_value, regime_today, regime_yesterday, vix, indicators) -> ComplexityResult

@dataclass(frozen=True)
class ComplexityResult:
    score: int  # 0-100
    should_escalate: bool  # score >= threshold
    signals: list[str]
```

### 6 Signals (max 100 points)

1. Drawdown > 5% from peak: +20
2. Regime change (today != yesterday): +20
3. VIX > 30: +20; VIX 25-30: +15
4. >= 3 tickers moved > 5% today: +15
5. > 7 positions held: +10
6. Conflicting indicators (MACD > 0 AND RSI > 70): +15

## TickerDiscovery (`src/agent/ticker_discovery.py`)

```python
MAX_DYNAMIC_TICKERS = 10; MIN_MARKET_CAP = 1_000_000_000; MIN_AVG_VOLUME = 100_000; EXPIRY_DAYS = 30

class TickerDiscovery:
    def __init__(self, db: Database) -> None
    def validate_ticker(self, ticker: str) -> TickerValidationResult  # sync, checks cap/volume/universe
    async def propose_ticker(self, ticker, rationale, source="agent", today=None, tenant_id="default") -> DiscoveredTickerRow | None
    async def get_active_tickers(self, tenant_id="default") -> list[str]
    async def expire_old(self, today=None, tenant_id="default") -> int
```

Source values: `"agent"` (passive JSON response), `"agent_tool"` (active discover_ticker tool), `"news"`, `"screener"`.

## Config (`config/strategies.py`)

```python
@dataclass(frozen=True)
class PortfolioAConfig:
    name = "Aggressive Momentum"; allocation_usd = 33_000.0; top_n = 1
    momentum_lookback_days = 63; momentum_skip_days = 5; rebalance_frequency = "daily"

@dataclass(frozen=True)
class PortfolioBConfig:
    name = "AI Full Autonomy"; allocation_usd = 66_000.0; model = "claude-opus-4-6"
    max_positions = 20; max_single_position_pct = 0.30
    escalation_model = "claude-opus-4-6"; escalation_threshold = 50
```

## Gotchas

- `build_user_message()` is shared by both single-shot and agentic paths -- changes affect both
- `ToolRegistry.execute()` raises `KeyError` on unknown tool, does NOT catch handler errors
- `AgentSettings` env prefix is `AGENT_` -- field `agent_tool_model` becomes env var `AGENT_AGENT_TOOL_MODEL`
- `save_tool_call_logs`: `tool_input` is a dict but column is Text -- must `json.dumps()` before storing
- `_run_portfolio_b()` returns 3-tuple `(trades, reasoning, tool_summary)` -- all callers must match
- Agentic tests must patch `orch._strategy_b._agent.analyze()` -- two-phase flow calls it in Phase 1 (seed)
- `TrackRecord.format_for_prompt` uses `:.0f` for win_rate -- 66.7% rounds to "67%", not "66%"
- Weekly compaction evaluation prompt now includes outcome feedback (verdict, track record)
