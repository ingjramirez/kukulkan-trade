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
| `src/agent/tools/__init__.py` | ToolRegistry + ToolDefinition dataclass |
| `src/agent/tools/portfolio.py` | 3 tools: get_current_positions, get_position_pnl, get_portfolio_summary |
| `src/agent/tools/market.py` | 2 tools: get_price_and_technicals, get_market_context |
| `src/agent/tools/news.py` | 1 tool: search_news |
| `src/agent/tools/actions.py` | 3 tools: propose_trades, update_watchlist, save_memory_note + ActionState |
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
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5-20250929", max_turns: int = 8, max_cost_usd: float = 0.50) -> None
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
    "claude-sonnet-4-5-20250929": (3.0, 15.0),   # (input_per_mtok, output_per_mtok)
    "claude-opus-4-6": (5.0, 25.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}

@dataclass
class TokenTracker:
    session_budget_usd: float = 0.50
    def record(self, model: str, input_tokens: int, output_tokens: int, turn: int) -> None
    def summary(self) -> dict  # {total_input_tokens, total_output_tokens, total_cost_usd, budget_usd, ...}
    @property
    def budget_exceeded(self) -> bool
```

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

### 9 Tools

| Tool | File | Input | Returns |
|------|------|-------|---------|
| `get_current_positions` | portfolio.py | `{}` | list of position dicts |
| `get_position_pnl` | portfolio.py | `{ticker}` | P&L + trailing stop status |
| `get_portfolio_summary` | portfolio.py | `{}` | cash, value, sector exposure |
| `get_price_and_technicals` | market.py | `{ticker}` | price, returns, RSI, MACD, SMA |
| `get_market_context` | market.py | `{}` | regime, VIX, yield curve, sector heatmap |
| `search_news` | news.py | `{ticker?}` | filtered news articles |
| `propose_trades` | actions.py | `{trades: [...]}` | accumulates, returns count |
| `update_watchlist` | actions.py | `{updates: [...]}` | accumulates |
| `save_memory_note` | actions.py | `{key, content}` | accumulates |

### ActionState (`src/agent/tools/actions.py`)

```python
@dataclass
class ActionState:
    proposed_trades: list[dict]; watchlist_updates: list[dict]; memory_notes: list[dict]
    def get_accumulated_state(self) -> dict
    def reset(self) -> None
```

Per-run isolation: one ActionState per agent session, accumulated across multiple tool calls.

### Registration Functions

```python
def register_portfolio_tools(registry, db, tenant_id, current_prices) -> None
def register_market_tools(registry, closes, vix=None, yield_curve=None, regime=None) -> None
def register_news_tools(registry, news_context) -> None
def register_action_tools(registry, state) -> None
```

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

## Config (`config/strategies.py`)

```python
@dataclass(frozen=True)
class PortfolioAConfig:
    name = "Aggressive Momentum"; allocation_usd = 33_000.0; top_n = 1
    momentum_lookback_days = 63; momentum_skip_days = 5; rebalance_frequency = "daily"

@dataclass(frozen=True)
class PortfolioBConfig:
    name = "AI Full Autonomy"; allocation_usd = 66_000.0; model = "claude-opus-4-6"
    max_positions = 10; max_single_position_pct = 0.30
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
