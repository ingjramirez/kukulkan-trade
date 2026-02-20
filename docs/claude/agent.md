# AI Agent & Tools

Machine-readable context for Claude. Covers the agent subsystem, Claude Code CLI integration, tools, memory, discovery, and strategy directives.

## Key Files

| File | Purpose |
|-|-|
| `src/agent/claude_agent.py` | ClaudeAgent class, system/user prompt builders, single-shot analyze() |
| `src/agent/claude_invoker.py` | ClaudeInvoker (subprocess `claude -p`), `claude_cli_call`/`claude_cli_json` helpers, session state/context writers |
| `src/agent/mcp_server.py` | MCP stdio server for Claude Code — reads session-state.json, registers tools |
| `src/agent/memory.py` | AgentMemoryManager: short-term, weekly compaction, agent notes |
| `src/agent/strategy_directives.py` | CONSERVATIVE/STANDARD/AGGRESSIVE + SESSION_DIRECTIVES |
| `src/agent/ticker_discovery.py` | TickerDiscovery: validation, propose, expire |
| `src/agent/conversation_store.py` | SQLite-backed conversation save/load/compress/cleanup |
| `src/agent/posture.py` | PostureLevel enum, PostureLimits, PostureManager |
| `src/agent/sentinel.py` | SentinelRunner: intraday stop/regime/fill checks, escalation guards |
| `src/agent/tools/__init__.py` | ToolRegistry + ToolDefinition dataclass |
| `src/agent/tools/portfolio.py` | 6 tools + 3 legacy aliases |
| `src/agent/tools/market.py` | 5 tools + 2 legacy aliases |
| `src/agent/tools/news.py` | 3 tools: search_news, search_historical_news, get_portfolio_a_status |
| `src/agent/tools/actions.py` | 6 tools + 2 legacy aliases + ActionState + declare_posture |
| `config/strategies.py` | PortfolioAConfig, PortfolioBConfig frozen dataclasses |
| `data/agent-workspace/` | CLAUDE.md (agent instructions), mcp.json (MCP config), settings.json |

## Architecture: Claude Code CLI

Phase 49 replaced the Anthropic SDK runtime (AgentRunner, PersistentAgent, TieredModelRunner, TokenTracker, BudgetTracker, ComplexityDetector, HaikuScanner, OpusValidator, ContextManager, SessionCompressor, SessionProfiles) with Claude Code CLI (`claude -p`) via Claude Max subscription.

**Runtime flow:**
```
Orchestrator → write_session_state() + write_context_file()
            → ClaudeInvoker.invoke() → subprocess `claude -p`
            → Claude Code spawns MCP server (reads session-state.json)
            → Claude reads context.md, calls MCP tools, returns JSON
            → Invoker reads JSON + session-results.json → returns InvokeResult
```

**No API key needed at runtime.** Anthropic SDK is dev-only (backtest).

### ClaudeInvoker (`src/agent/claude_invoker.py`)

```python
@dataclass
class InvokeResult:
    response: dict      # Parsed trading response (trades, reasoning, etc.)
    session_id: str     # For --resume across sessions
    accumulated: dict   # ActionState from MCP server (trailing stops, posture, etc.)
    error: str | None
    # Properties: .trades, .reasoning, .posture, .trailing_stop_requests, .tool_summary

class ClaudeInvoker:
    def __init__(self, workspace=WORKSPACE, timeout=600, max_turns=25, model="claude-sonnet-4-6")
    async def invoke(self, session_type: str, today: date | None = None) -> InvokeResult
```

Session strategy: morning starts new session, midday/close resume via `--resume <session_id>`. Session ID persisted in `.session-id` file.

CLI command built by `_build_cmd()`:
```bash
claude -p "<prompt>" --output-format json --mcp-config mcp.json \
    --allowedTools "mcp__kukulkan__*" --max-turns 25 --model claude-sonnet-4-6
```

### Lightweight CLI Helpers

```python
async def claude_cli_call(prompt: str, model: str = "claude-haiku-4-5-20251001", timeout: int = 120) -> str
    # Text-in/text-out. Used by memory compaction.

async def claude_cli_json(prompt: str, model: str = "claude-sonnet-4-6", timeout: int = 120) -> dict
    # Text-in/JSON-out. Used by weekly improvement analyzer.
```

Both wrap `subprocess.run` via `asyncio.to_thread()`. No MCP, no session persistence.

### MCP Server (`src/agent/mcp_server.py`)

Stdio-based MCP server registered via `mcp.json`. On startup:
1. Reads `session-state.json` → reconstructs DataFrames, prices, regime
2. Registers all agent tools (portfolio, market, news, actions)
3. Writes accumulated ActionState to `session-results.json` on exit

### Session State Files (`data/agent-workspace/`)

| File | Written By | Read By |
|-|-|-|
| `session-state.json` | `write_session_state()` | MCP server startup |
| `context.md` | `write_context_file()` | Claude Code (prompt context) |
| `session-results.json` | MCP server on exit | `ClaudeInvoker._read_session_results()` |
| `.session-id` | `ClaudeInvoker` | `ClaudeInvoker` (resume) |
| `CLAUDE.md` | Manual | Claude Code (system instructions) |
| `mcp.json` | Manual | Claude Code (MCP config) |

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

**Note:** `ClaudeAgent` is now only used by the backtest (`src/backtest/ai_strategy.py`). Production path uses `ClaudeInvoker`.

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
# Used by backtest path only (production uses write_context_file)
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

## Strategy Directives (`src/agent/strategy_directives.py`)

| Strategy | Key Rules |
|-|-|
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

**Important:** `execute()` does NOT catch errors. In MCP server context, errors propagate to the MCP handler.

### 24 Tools (4 modules)

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
| `get_signal_rankings` | `{}` | top-ranked, biggest movers, alerts |

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
    discovery_proposals: list[dict]
    declared_posture: str | None
    def get_accumulated_state(self) -> dict
    def reset(self) -> None
```

Per-run isolation: one ActionState per MCP session, accumulated across multiple tool calls. Written to `session-results.json` on exit.

### Registration Functions

```python
def register_portfolio_tools(registry, db, tenant_id, current_prices, closes=None, held_tickers=None) -> None
def register_market_tools(registry, closes, vix=None, yield_curve=None, regime=None, db=None, held_tickers=None, tenant_id="default") -> None
def register_news_tools(registry, news_context, news_fetcher=None, current_prices=None) -> None
def register_action_tools(registry, state, db=None, tenant_id="default", ticker_discovery=None) -> None
```

### Discovery Tool Flow

```
Agent mid-session → search_ticker_info("ANET") → yfinance lookup
→ "Meets minimums, fits thesis" → discover_ticker("ANET", "reason", "high")
→ Validates + saves to DB as "proposed" → returns immediately to agent
→ After agent loop: orchestrator sends Telegram approval → approved/rejected
→ Approved tickers enter universe on next session
```

## Posture System (`src/agent/posture.py`)

```python
class PostureLevel(Enum):
    DEFENSIVE = "defensive"; CAUTIOUS = "cautious"; NEUTRAL = "neutral"
    OPPORTUNISTIC = "opportunistic"; AGGRESSIVE = "aggressive"; CRISIS = "crisis"

class PostureManager:
    def resolve_posture(self, declared, track_record) -> tuple[PostureLevel, PostureLimits]
```

All posture levels unlocked (paper trading freedom). `posture_limits` passed to `RiskManager.check_pre_trade()` — can only tighten, never loosen.

## AgentMemoryManager (`src/agent/memory.py`)

```python
MAX_SHORT_TERM = 3; MAX_WEEKLY_SUMMARIES = 4; MAX_AGENT_NOTES = 10

class AgentMemoryManager:
    def build_memory_prompt(self, memories: dict) -> str  # keys: short_term, weekly_summary, agent_note
    async def save_short_term(self, db, analysis_date, response, tenant_id="default") -> None
    async def save_agent_notes(self, db, notes: list[dict], tenant_id="default") -> None
    async def run_weekly_compaction(self, db, tenant_id="default", outcome_summary=None, track_record_text=None) -> None
```

Weekly compaction uses `claude_cli_call(model="claude-haiku-4-5-20251001")` to evaluate past week's trades. Stores as `weekly_summary` with key like `week_2026-07`.

## TickerDiscovery (`src/agent/ticker_discovery.py`)

```python
MAX_DYNAMIC_TICKERS = 10; MIN_MARKET_CAP = 1_000_000_000; MIN_AVG_VOLUME = 100_000; EXPIRY_DAYS = 30

class TickerDiscovery:
    def __init__(self, db: Database) -> None
    def validate_ticker(self, ticker: str) -> TickerValidationResult
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

- `AIAutonomyStrategy.__init__()` takes NO args — `agent=` parameter was removed in Phase 49
- `claude_cli_call` is lazily imported in `memory.py` → patch at `src.agent.claude_invoker.claude_cli_call`
- `claude_cli_json` is lazily imported in `weekly_improvement.py` → patch at `src.agent.claude_invoker.claude_cli_json`
- `build_user_message()` is only used by backtest path now — production uses `write_context_file()`
- `ToolRegistry.execute()` raises `KeyError` on unknown tool, does NOT catch handler errors
- `AgentSettings` env prefix is `AGENT_` — field `agent_tool_model` becomes env var `AGENT_AGENT_TOOL_MODEL`
- `save_tool_call_logs`: `tool_input` is a dict but column is Text — must `json.dumps()` before storing
- `_run_portfolio_b()` returns 3-tuple `(trades, reasoning, tool_summary)` — all callers must match
- Tests mocking Portfolio B should mock `orchestrator._run_portfolio_b` as `AsyncMock(return_value=([], "reasoning", "tool_summary"))` — NOT `_strategy_b._agent`
- `TrackRecord.format_for_prompt` uses `:.0f` for win_rate — 66.7% rounds to "67%"
- Weekly compaction evaluation prompt includes outcome feedback (verdict, track record)
