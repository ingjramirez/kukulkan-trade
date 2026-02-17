# Bot Pipeline & Execution

Machine-readable context for Claude. Covers the orchestrator, strategies, execution, intraday snapshots, scheduler, and notifications.

## Key Files

| File | Purpose |
|------|---------|
| `src/orchestrator.py` | Orchestrator class, 9-step daily pipeline, tenant iteration |
| `src/strategies/portfolio_a.py` | MomentumStrategy: 63-day ETF momentum, top 1 selection |
| `src/strategies/portfolio_b.py` | AIAutonomyStrategy: Claude agent context prep + response-to-trades |
| `src/execution/alpaca_executor.py` | AlpacaExecutor: market orders, fill polling, position sync |
| `src/execution/paper_trader.py` | PaperTrader: local simulation, same interface |
| `src/execution/client_factory.py` | AlpacaClientFactory: per-tenant cached TradingClient |
| `src/main.py` | APScheduler setup, job definitions (3x daily + intraday + weekly) |
| `src/intraday.py` | 15-min portfolio snapshot collector |
| `src/events/event_bus.py` | EventBus singleton: SSE pub/sub, 18 event types |
| `src/notifications/telegram_bot.py` | TelegramNotifier: daily brief, trade confirmation, large trade + inverse approvals |
| `src/notifications/telegram_factory.py` | TelegramFactory: per-tenant cached notifier |
| `src/notifications/weekly_report.py` | WeeklyReporter: Friday performance summary |

## Orchestrator (`src/orchestrator.py`)

```python
class Orchestrator:
    def __init__(self, db: Database, notifier: TelegramNotifier | None = None, executor=None) -> None

    async def run_all_tenants(self, today=None, session="") -> list[dict]
    async def run_tenant_session(self, tenant: TenantRow, today=None, session="") -> dict
    async def run_daily(
        self, today=None, session="", tenant_id="default", strategy_mode=None,
        run_portfolio_a=True, run_portfolio_b=True, allocations=None, portfolio_b_universe=None,
    ) -> dict
```

### Daily Pipeline Steps

```
run_daily() entry
|-- Guard: market_closed -> skip
|-- Step 1: initialize_portfolios()
|-- Step 1.1: sync_positions() [Alpaca only]
|-- Step 1.2: _detect_deposits()
|-- Step 1.5: expire_old() + get_dynamic_universe()
|-- Step 2: fetch_universe() -> closes, volumes DataFrames (1-year OHLCV)
|-- Step 2.1: _check_trailing_stops() -> trailing_stop_sells, alerts
|-- Step 2.3: _handle_rebalance() [if pending_rebalance=True]
|-- Step 2.5: recovery_check() -> backfill missed days
|-- Step 3: macro_data -> yield_curve, vix
|-- Step 3.1: _regime_classifier.classify() -> regime_result
|-- Step 3.5: check_circuit_breakers() -> halted_portfolios
|-- Step 4: _run_portfolio_a() -> (trades_a, reason)
|-- Step 5: fetch + compact news -> news_context
|-- Step 5.1: get_historical_context() -> append ChromaDB results
|-- Step 5.5: earnings_calendar.get_upcoming() -> earnings_context [Morning only]
|-- Step 5.6: cleanup_expired_watchlist() [Morning only]
|-- Step 6: _run_portfolio_b() -> (trades_b, reasoning, tool_summary)
|-- Step 6.5: risk_manager.check_pre_trade() -> RiskVerdict (allowed, blocked, requires_approval, requires_trade_approval)
|   |-- Merge trailing_stop_sells (bypass risk filter)
|   |-- Inverse ETF approval via Telegram (requires_approval)
|   |-- Large trade approval via Telegram (requires_trade_approval, >threshold%)
|   |-- Publishes SSE events: TRADE_REJECTED (blocked), TRADE_APPROVAL_REQUESTED/RESOLVED
|-- Execute all trades (sells first, then buys) → publishes TRADE_EXECUTED per trade
|-- Step 6.8: _process_suggested_tickers() (passive JSON path) + _process_tool_discoveries() (active tool path)
|-- Step 7.1: create_trailing_stops() + remove watchlist if traded
|-- Step 7.2: deactivate_trailing_stops() for sells
|-- Step 8: take_snapshot() [enabled portfolios, Alpaca prices preferred]
|-- Step 8.5: _reconcile_equity() [$10-$50 drift correction]
|-- Step 9: _send_notifications() -> daily brief + trade confirmation
|-- SSE events published throughout: SESSION_STARTED/COMPLETED/SKIPPED, POSITIONS_UPDATED, PORTFOLIO_SNAPSHOT, SYSTEM_ERROR
```

### Key Helper Methods

```python
async def _run_portfolio_a(self, closes, today, tenant_id="default", allocations=None) -> tuple[list, str]
async def _run_portfolio_b(self, closes, volumes, yield_curve, vix, today, news_context="", session="",
    regime_result=None, tenant_id="default", strategy_mode=None, allocations=None,
    portfolio_b_universe=None, earnings_context=None) -> tuple[list, str, dict]
async def _check_trailing_stops(self, tenant_id, closes, run_portfolio_a, run_portfolio_b) -> tuple[list[TradeSchema], list[dict]]
async def _handle_rebalance(self, tenant_id, closes, run_portfolio_a, run_portfolio_b) -> TenantAllocations | None
async def recovery_check(self, today, closes, tenant_id="default", allocations=None, run_portfolio_a=True, run_portfolio_b=True) -> list[str]
async def _detect_deposits(self, allocations, tenant_id) -> TenantAllocations
async def _reconcile_equity(self, tenant_id, run_portfolio_a, run_portfolio_b, allocations) -> float | None
```

### Multi-Tenant Iteration

```python
_INTER_TENANT_DELAY_SECONDS = 2.0

async def run_all_tenants(self, today, session):
    tenants = await db.get_active_tenants()
    for tenant in tenants:
        if not tenant_fully_configured(tenant): continue
        # Create tenant-specific executor, notifier, allocations
        await self.run_tenant_session(tenant, today, session)
        await asyncio.sleep(_INTER_TENANT_DELAY_SECONDS)
```

Each tenant session is wrapped in try-except. Failures are logged but don't block other tenants.

## Portfolio A: Momentum (`src/strategies/portfolio_a.py`)

```python
class MomentumStrategy:
    def __init__(self, lookback: int = 63, skip: int = 5, top_n: int = 1) -> None
    def rank(self, closes: pd.DataFrame) -> pd.DataFrame  # [ticker, return_63d, rank]
    def get_target_ticker(self, closes: pd.DataFrame) -> str | None
    def generate_trades(self, closes, current_positions, cash, portfolio_value=None) -> list[TradeSchema]
    def get_ranking_rows(self, closes, ranking_date) -> list
```

- Filters to `PORTFOLIO_A_UNIVERSE` (20 ETFs: sector + thematic)
- Calculates 63-day momentum, skips last 5 days (mean reversion filter)
- Holds single top-momentum ETF, rebalances daily
- Sells all non-target holdings, buys target
- Cap at `RISK_RULES.max_single_position_pct`

## Portfolio B: AI Autonomy (`src/strategies/portfolio_b.py`)

```python
class AIAutonomyStrategy:
    def __init__(self, agent: ClaudeAgent | None = None) -> None
    def prepare_context(self, closes, volumes, positions, cash, total_value, recent_trades,
        regime=None, yield_curve=None, vix=None, news_context="", system_prompt=None, universe=None) -> dict
    def agent_response_to_trades(self, response, total_value, current_positions, latest_prices,
        extra_tickers=None, universe=None) -> list[TradeSchema]
    async def save_decision(self, db, analysis_date, response, trades, tenant_id="default",
        regime=None, session_label=None) -> None
```

```python
def filter_interesting_tickers(closes, current_positions, top_movers=15, universe=None) -> list[str]
# Returns: held tickers + top movers by 1d change + RSI extremes
```

Position limits: max 30% weight, max 10 positions, conviction multipliers (high=1.0, medium=0.7, low=0.4).

## Execution

### AlpacaExecutor (`src/execution/alpaca_executor.py`)

```python
class AlpacaExecutor:
    def __init__(self, db, client: TradingClient, fill_timeout=30.0, fill_poll_interval=5.0) -> None
    async def initialize_portfolios(self, allocations=None, tenant_id="default") -> None
    async def execute_trades(self, trades: list[TradeSchema], tenant_id="default") -> list[TradeSchema]
    async def sync_positions(self) -> dict[str, list[dict]]
    async def take_snapshot(self, portfolio_name, snapshot_date, prices, allocations=None, tenant_id="default") -> None
```

Flow: Submit market order -> poll every 5s (max 30s) -> update DB positions/cash -> log trade.
Terminal states: filled, partially_filled (success), rejected/canceled/expired (fail), timeout (reconciliation queue).

### PaperTrader (`src/execution/paper_trader.py`)

Same interface as AlpacaExecutor. Simulates locally: checks cash/shares sufficiency, updates DB, no Alpaca calls.

### ClientFactory (`src/execution/client_factory.py`)

```python
class AlpacaClientFactory:
    _cache: dict[str, TradingClient]
    @classmethod def get_trading_client(cls, tenant: TenantRow) -> TradingClient
    @classmethod def invalidate(cls, tenant_id: str) -> None
    @classmethod def clear_cache(cls) -> None
```

## Scheduler (`src/main.py`)

| Job | Schedule (US/Eastern) | Function |
|-----|-----------------------|----------|
| Morning | Mon-Fri 10:00 AM | `orchestrator.run_all_tenants(session="Morning")` |
| Midday | Mon-Fri 12:30 PM | `orchestrator.run_all_tenants(session="Midday")` |
| Closing | Mon-Fri 3:45 PM | `orchestrator.run_all_tenants(session="Closing")` |
| Intraday Snapshots | Every 15 min, 9:30-16:00 ET (Mon-Fri) | `collect_intraday_snapshot()` |
| Weekly Memory Compaction | Sunday 6:00 PM | `memory_manager.run_weekly_compaction()` |
| Weekly Performance Report | Friday 5:00 PM | `WeeklyReporter.generate_and_send()` |
| Intraday Cleanup | Sunday 7:00 PM | `purge_old_intraday_snapshots(days=90)` |

```python
async def run_once() -> None    # --run-now flag, single pipeline execution
async def run_scheduled() -> None  # APScheduler, recurring jobs
```

## Intraday Snapshots (`src/intraday.py`)

```python
async def collect_intraday_snapshot(db: Database, tenant: TenantRow) -> int
```

Fetches live Alpaca positions, sums per portfolio (positions_value + cash = total_value), stores `IntradaySnapshotRow`.

## Notifications

### TelegramNotifier (`src/notifications/telegram_bot.py`)

```python
class TelegramNotifier:
    def __init__(self, bot_token=None, chat_id=None) -> None
    async def send_daily_brief(self, brief_date, regime, portfolio_a, portfolio_b, proposed_trades,
        commentary="", session="", strategy_mode="conservative", run_portfolio_a=True, run_portfolio_b=True,
        trailing_stop_alerts=None, agent_tool_summary=None) -> bool
    async def send_trade_confirmation(self, trades: list[TradeSchema]) -> bool
    async def send_error(self, error_msg: str) -> bool
    async def send_ticker_proposal(self, ticker_row, request_id) -> int | None
    async def wait_for_ticker_approval(self, request_id, timeout_seconds=300) -> str  # "approve"|"reject"
    async def send_inverse_approval(self, trade, risk_reason, request_id) -> int | None
    async def wait_for_inverse_approval(self, request_id, timeout_seconds=300) -> str  # "approve"|"reject"
    async def send_large_trade_approval(self, trade, trade_pct, approval_reason, request_id) -> int | None
    async def wait_for_large_trade_approval(self, request_id, timeout_seconds=300) -> str  # "approve"|"reject"
```

### TelegramFactory (`src/notifications/telegram_factory.py`)

```python
class TelegramFactory:
    _cache: dict[str, TelegramNotifier]
    @classmethod def get_notifier(cls, tenant: TenantRow) -> TelegramNotifier
    @classmethod def invalidate(cls, tenant_id: str) -> None
```

### WeeklyReporter (`src/notifications/weekly_report.py`)

```python
class WeeklyReporter:
    def __init__(self, db, notifier, tenant_id="default", allocations=None,
        run_portfolio_a=True, run_portfolio_b=True) -> None
    async def generate_and_send(self, report_date=None) -> str
```

Sections: Portfolio A/B value + weekly % + vs SPY, trades of the week, AI decisions, drawdown status.

## TradeSchema

```python
class TradeSchema:
    portfolio: PortfolioName  # A or B
    ticker: str
    side: OrderSide  # BUY or SELL
    shares: float
    price: float
    reason: str
    @property
    def total(self) -> float  # shares * price
```

## Gotchas

- `_run_portfolio_b()` returns 3-tuple `(trades, reasoning, tool_summary)` -- ALL callers (incl skip path) must match
- PaperTrader + AlpacaExecutor `execute_trades()` accept `tenant_id` -- mock executors in tests must accept `**kwargs`
- Alpaca client is sync; wrap with `asyncio.to_thread()` for async context
- Alpaca SDK `str(OrderStatus)` -> `"orderstatus.filled"`, use `rsplit(".", 1)[-1]`
- Lazy imports inside functions: patch at `src.execution.client_factory.AlpacaClientFactory`, NOT `src.module.AlpacaClientFactory`
- Deposit detection: never compare broker equity to tracked totals -- use Alpaca account activities API (CSD/JNLC)
- `format_daily_brief` + `WeeklyReporter` skip disabled portfolios
- Trailing stop sells bypass risk filter (stops ARE the risk mechanism)
