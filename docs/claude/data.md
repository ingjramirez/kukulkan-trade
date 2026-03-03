# Data Pipeline & Storage

Machine-readable context for Claude. Covers market data, news pipeline, database models, and ChromaDB vectors.

## Key Files

| File | Purpose |
|------|---------|
| `src/data/market_data.py` | MarketData: yfinance wrapper, OHLCV, calendar |
| `src/data/macro_data.py` | MacroData: FRED API (yield curve, VIX) |
| `src/data/news_aggregator.py` | NewsAggregator: 3-source priority pipeline with dedup |
| `src/data/news_compactor.py` | NewsCompactor: cluster + score + format for agent prompt |
| `src/data/news_fetcher.py` | NewsFetcher: orchestrator-facing, ChromaDB storage + historical context |
| `src/data/news_article.py` | NewsArticle + NewsCluster dataclasses |
| `src/data/alpaca_news.py` | AlpacaNewsFetcher: Benzinga-sourced headlines |
| `src/data/finnhub_news.py` | FinnhubNewsFetcher: company + general news |
| `src/data/earnings_calendar.py` | EarningsCalendar: yfinance earnings dates |
| `src/storage/database.py` | Database class: async SQLAlchemy (PG prod, SQLite tests), all CRUD methods |
| `src/storage/models.py` | 25 ORM models + Pydantic schemas |
| `src/storage/vector_store.py` | ChromaDB client wrapper |
| `config/universe.py` | 70 tickers, SECTOR_MAP, SECTOR_ETF_MAP, get_tenant_universe() |

## Market Data (`src/data/market_data.py`)

```python
class MarketData:
    async def fetch_universe(self, tickers: list[str], period="1y") -> tuple[pd.DataFrame, pd.DataFrame]
        # Returns: (closes, volumes) DataFrames -- 1-year daily OHLCV via yfinance

    async def fetch_closes(self, tickers: list[str], period="1y") -> pd.DataFrame

    def is_market_open(self, today: date | None = None) -> bool
        # Uses pandas_market_calendars (NYSE)

    async def get_latest_prices(self, tickers: list[str]) -> dict[str, float]
```

All yfinance calls wrapped in `asyncio.to_thread()`.

### Extended Hours Prices

```python
async def get_extended_hours_prices(tickers: list[str]) -> dict[str, float]
    # yfinance fast_info with 5-minute TTL cache, wrapped in asyncio.to_thread()
def _clear_price_cache() -> None  # for tests
```

## Macro Data (`src/data/macro_data.py`)

```python
class MacroData:
    async def get_yield_curve(self) -> float | None   # 10Y-2Y Treasury spread from FRED
    async def get_vix(self) -> float | None            # VIX from yfinance (^VIX)
```

Requires `FRED_API_KEY` env var for yield curve. VIX fetched via yfinance.

## News Pipeline

### Data Models (`src/data/news_article.py`)

```python
@dataclass
class NewsArticle:
    headline: str; summary: str; source: str  # "alpaca"|"finnhub"|"yfinance"
    publisher: str; tickers: list[str]; published_at: datetime | None
    url: str = ""; sentiment: float | None = None

@dataclass
class NewsCluster:
    representative: NewsArticle; source_count: int = 1
    signal: str = "INFO"  # POS, NEG, MACRO, EVENT, INFO
    score: int = 0; all_tickers: list[str]
```

### NewsAggregator (`src/data/news_aggregator.py`)

```python
class NewsAggregator:
    def __init__(self, alpaca_fetcher=None, finnhub_fetcher=None) -> None
    def fetch_all(self, tickers: list[str], max_articles=100) -> list[NewsArticle]
```

Priority order: Alpaca (highest quality) -> Finnhub -> yfinance (fallback if <10 articles).
Deduplicates by headline word overlap (>50% = same story via `_headlines_overlap()`).

### Individual Fetchers

```python
class AlpacaNewsFetcher:  # src/data/alpaca_news.py
    def fetch(self, tickers: list[str], limit=50) -> list[NewsArticle]
    # Uses alpaca-py NewsClient. Data: response.data.get("news", [])

class FinnhubNewsFetcher:  # src/data/finnhub_news.py
    def fetch(self, tickers: list[str], days_back=2, max_per_ticker=5) -> list[NewsArticle]
    # Company news per ticker (max 20 tickers) + general market news
```

### NewsCompactor (`src/data/news_compactor.py`)

```python
class NewsCompactor:
    def compact(self, articles: list[NewsArticle], universe: set[str]) -> str
        # Clusters articles, scores by relevance, formats for agent prompt
        # Output format: "TICKER|SIGNAL|headline|sources" lines
```

Separates into portfolio-relevant clusters + discovery clusters (not in universe).

### NewsFetcher (`src/data/news_fetcher.py`)

Orchestrator-facing class that combines aggregation + compaction + ChromaDB storage.

```python
class NewsFetcher:
    def __init__(self, db=None, chroma_client=None) -> None
    async def fetch_and_compact(self, tickers, universe) -> str  # compact news for agent prompt
    def store_articles(self, articles: list[NewsArticle]) -> None  # store in ChromaDB with published_at ISO metadata
    def get_historical_context(self, tickers: list[str], days_back=7) -> str
        # Query ChromaDB per held ticker, deduplicates against today's headlines
```

### Earnings Calendar (`src/data/earnings_calendar.py`)

```python
class EarningsCalendar:
    async def refresh(self, tickers: list[str], db: Database) -> int  # fetch from yfinance, persist
    async def get_upcoming(self, db: Database, tickers: list[str], days_ahead=14) -> list[dict]
```

`EarningsCalendarRow` has no `tenant_id` (earnings dates are global) -- filter by tickers at query time.

## Database (`src/storage/database.py`)

```python
class Database:
    def __init__(self, url: str = "sqlite+aiosqlite:///data/kukulkan.db") -> None  # prod overrides via DATABASE_URL env
    async def init_db(self) -> None   # create_all tables
    async def close(self) -> None
    def session(self) -> AsyncSession  # context manager
```

### CRUD Methods (all accept `tenant_id: str = "default"` unless noted)

**Portfolios:**
- `get_portfolio(name, tenant_id)`, `save_portfolio(row)`, `update_portfolio_cash(name, cash, tenant_id)`

**Positions:**
- `get_positions(portfolio, tenant_id)`, `upsert_position(row)`, `delete_position(portfolio, ticker, tenant_id)`
- `get_all_positions(tenant_id)` -- both portfolios

**Trades:**
- `save_trade(row)`, `get_recent_trades(portfolio, limit, tenant_id)`, `get_trades(portfolio, side, limit, tenant_id)`

**Snapshots:**
- `save_snapshot(row)`, `get_snapshots(portfolio, since, tenant_id)`, `get_latest_snapshot(portfolio, tenant_id)`

**Intraday:**
- `save_intraday_snapshot(row, is_extended_hours, market_phase)`, `get_intraday_snapshots(portfolio, since, tenant_id)`
- `purge_old_intraday_snapshots(days, tenant_id)`
- `get_last_market_hours_snapshot(tenant_id)` -- latest snapshot where `is_extended_hours=False`

**Sentinel Actions:**
- `save_sentinel_action(row)`, `get_pending_sentinel_actions(tenant_id)`
- `resolve_sentinel_action(action_id, status, resolved_by)`

**Momentum:**
- `save_momentum_rankings(rows)`, `get_latest_rankings(tenant_id)`

**Agent Decisions:**
- `save_agent_decision(row)`, `get_recent_decisions(limit, tenant_id)`

**Agent Memory:**
- `get_memories_by_category(category, tenant_id)`, `save_memory(row)`, `delete_old_memories(category, keep, tenant_id)`

**Trailing Stops:**
- `get_trailing_stops(portfolio, tenant_id)`, `save_trailing_stop(row)`, `update_trailing_stop_peak(id, peak, stop)`
- `deactivate_trailing_stop(portfolio, ticker, tenant_id)`

**Watchlist:**
- `get_active_watchlist(portfolio, tenant_id)`, `add_watchlist_item(row)`, `remove_watchlist_item(portfolio, ticker, tenant_id)`
- `cleanup_expired_watchlist(tenant_id)`

**Earnings:**
- `save_earnings(row)`, `get_upcoming_earnings(tickers, days_ahead)`

**Discovered Tickers:**
- `get_discovered_ticker(ticker, tenant_id)`, `save_discovered_ticker(row)`
- `update_discovered_ticker_status(ticker, status, tenant_id)`, `expire_old_tickers(today, tenant_id)`
- `get_all_discovered_tickers(tenant_id, status)`, `get_all_approved_tickers_all_tenants()`

**Tool Call Logs:**
- `save_tool_call_logs(logs, session_date, session_label, tenant_id)`, `get_tool_call_logs(session_date, limit, tenant_id)`

**Tenants:**
- `create_tenant(row)`, `get_tenant(tenant_id)`, `get_active_tenants()`, `get_all_tenants()`
- `get_tenant_by_username(username)`, `update_tenant(tenant_id, updates: dict)`, `deactivate_tenant(tenant_id)`

## ORM Models (`src/storage/models.py`)

25 models total:

| Model | Key Columns | tenant_id? |
|-------|-------------|------------|
| `PortfolioRow` | name, cash, total_value, updated_at | yes |
| `PositionRow` | portfolio, ticker, shares, avg_price | yes |
| `TradeRow` | portfolio, ticker, side, shares, price, reason, executed_at | yes |
| `SnapshotRow` | portfolio, date, total_value, cash, daily_return_pct, cumulative_return_pct | yes |
| `IntradaySnapshotRow` | portfolio, timestamp, total_value, cash, positions_value, is_extended_hours, market_phase | yes |
| `SentinelActionRow` | action_type, ticker, reason, source, alert_level, status, resolved_at, resolved_by | yes |
| `MomentumRankingRow` | date, ticker, return_63d, rank | no |
| `AgentDecisionRow` | date, prompt_summary, response_summary, proposed_trades, reasoning, model_used, tokens_used, regime, session_label | yes |
| `AgentMemoryRow` | category (short_term/weekly_summary/agent_note), memory_key, content, expires_at | yes |
| `TrailingStopRow` | portfolio, ticker, entry_price, peak_price, trail_pct, stop_price, is_active | yes |
| `EarningsCalendarRow` | ticker, earnings_date, source | no (global) |
| `WatchlistRow` | portfolio, ticker, reason, conviction, target_entry, added_date, expires_at | yes |
| `DiscoveredTickerRow` | ticker, source, rationale, status, proposed_at, expires_at, sector, market_cap | yes (composite unique tenant_id+ticker) |
| `ToolCallLogRow` | session_date, session_label, turn, tool_name, tool_input, tool_output_preview, success, error, influenced_decision | yes |
| `TenantRow` | id, name, is_active, credentials (enc), config, allocations, quiet_hours_start/end/timezone | -- |

## ChromaDB (`src/storage/vector_store.py`)

- Docker container on port 8000
- Collection: `news_articles`
- Documents: `"{title}. {summary}"` (falls back to title-only if no summary)
- Metadata: ticker, source, published_at (date-only ISO `YYYY-MM-DD`), signal, region
- `search_similar(query, n_results, ticker, days_back)` — date-range filtering via `$and`/`$gte` where clauses
- `cleanup_old(days=180)` — batch-deletes articles older than 6 months (Sunday 7PM cleanup job)
- Used for: historical context retrieval per ticker (default 30-day lookback, max 180)

## Universe (`config/universe.py`)

```python
PORTFOLIO_A_UNIVERSE = SECTOR_ETFS + THEMATIC_ETFS  # 20 ETFs
PORTFOLIO_B_UNIVERSE = ...  # 70 tickers (ETFs + stocks + crypto + fixed income + international)
FULL_UNIVERSE = sorted(set(PORTFOLIO_B_UNIVERSE))
BENCHMARK_TICKERS = ["SPY"]

SECTOR_MAP: dict[str, str]     # ticker -> sector name (19 sectors)
SECTOR_ETF_MAP: dict[str, str] # sector -> benchmark ETF

async def get_dynamic_universe(db) -> list[str]  # FULL_UNIVERSE + approved discovered tickers
```

### Per-Tenant Universe (`config/universe.py`)

```python
async def get_tenant_universe(
    tenant: TenantRow,
    discovered_tickers: list[str] | None = None,
) -> list[str]
```

Applies tenant's `ticker_whitelist` (if set, replaces base), `ticker_additions`, `ticker_exclusions`.

## Gotchas

- PG columns are `TIMESTAMP WITHOUT TIME ZONE` — use `datetime.utcnow()`, NOT `datetime.now(timezone.utc)` (asyncpg rejects tz-aware)
- SQLite returns naive datetimes -- when comparing with `datetime.now(timezone.utc)`, strip tzinfo first
- `alpaca-py` `NewsSet.data` is `Dict[str, List[News]]` (key "news"), NOT a flat list
- `TradingClient` has no `get_account_activities` -- use raw `client.get("/v2/account/activities", params)`
- `pandas-ta` unavailable for Python 3.11 -- using `ta` library instead
- EarningsCalendarRow has no `tenant_id` -- earnings dates are global, filter by tickers at query time
- `save_tool_call_logs`: `tool_input` is a dict but column is Text -- must `json.dumps()` before storing
- SQLAlchemy `create_all` doesn't ALTER existing tables -- need SQL migrations for schema changes
- SQLAlchemy column defaults (e.g. `default=33.33`) only apply on INSERT -- in-memory TenantRow without DB round-trip has `None`
