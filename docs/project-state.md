# Kukulkan Trade — Project State

**Date:** 2026-02-16
**Commit:** `0bc1b11` (main)
**Phase:** 36 complete

---

## 1. What Is This?

An educational algorithmic trading bot running on Alpaca paper trading. Two portfolios:

| Portfolio | Strategy | Target Allocation | Description |
|-----------|----------|-------------------|-------------|
| **A** | Aggressive Momentum | 33% of equity | Rules-based: 63-day momentum ranking of sector/thematic ETFs, monthly rebalance |
| **B** | AI Full Autonomy | 67% of equity | Claude AI agent makes all decisions: stock selection, sizing, timing, hedging |

The system runs 3x daily (morning, midday, close) via APScheduler, with a full pipeline: market data fetch, regime classification, strategy execution, risk filtering, order execution, snapshot recording, and Telegram notifications.

---

## 2. Architecture Overview

```
                    +-----------------+
                    |   Scheduler     |  APScheduler (3x daily)
                    +--------+--------+
                             |
                    +--------v--------+
                    |  Orchestrator   |  2,910 lines — daily pipeline
                    +--------+--------+
                             |
          +------------------+------------------+
          |                                     |
+---------v---------+             +-------------v-------------+
|  Portfolio A      |             |  Portfolio B              |
|  (Momentum)       |             |  (AI Agent)               |
|  Rules-based ETF  |             |  3-level fallback:        |
|  rotation         |             |  Persistent → Agentic →   |
+-------------------+             |  Single-shot              |
                                  +---------------------------+
                                            |
                         +------------------+------------------+
                         |                  |                  |
                  +------v------+  +--------v-------+  +------v------+
                  | Haiku 4.5   |  | Sonnet 4.5     |  | Opus 4.6    |
                  | Scanner     |  | Investigator   |  | Validator   |
                  | ($0.002/run)|  | ($0.10-0.50)   |  | ($0.50-2.00)|
                  +-------------+  +----------------+  +-------------+

+-----------+  +-----------+  +-----------+  +-----------+
| Market    |  | News      |  | Risk      |  | Execution |
| Data      |  | Pipeline  |  | Manager   |  | Layer     |
| (yfinance)|  | (Alpaca,  |  | (inverse  |  | (Alpaca   |
|           |  |  Finnhub, |  |  guards,  |  |  paper +  |
|           |  |  yfinance,|  |  posture,  |  |  Paper    |
|           |  |  ChromaDB)|  |  circuit   |  |  trader)  |
|           |  |           |  |  breakers) |  |           |
+-----------+  +-----------+  +-----------+  +-----------+

+-----------+  +-----------+  +-----------+
| FastAPI   |  | Next.js   |  | Telegram  |
| REST API  |  | Frontend  |  | Bot       |
| :8001     |  | :3000     |  | Alerts    |
+-----------+  +-----------+  +-----------+
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| Database | SQLite (async via aiosqlite + SQLAlchemy 2.0) |
| API | FastAPI + uvicorn (port 8001) |
| Frontend | Next.js (port 3000, separate repo) |
| AI | Anthropic Claude (Haiku 4.5 / Sonnet 4.5 / Opus 4.6) |
| Broker | Alpaca (paper trading) |
| Market Data | yfinance (primary), Alpaca (news only) |
| News Vectors | ChromaDB (Docker, port 8000) |
| Notifications | Telegram bot |
| Scheduling | APScheduler |
| Auth | JWT (python-jose, 2h expiry) + bcrypt passwords |
| Encryption | Fernet (tenant credentials) |
| CI/CD | GitHub Actions (test → lint → rsync → migrate → restart) |
| Hosting | Hetzner VPS (128.140.102.191) |
| Linting | ruff (line-length=120, target py311) |
| Testing | pytest-asyncio (auto mode), in-memory SQLite |
| Retries | tenacity (transient network failures) |

---

## 3. Codebase Metrics

| Metric | Value |
|--------|-------|
| Source files (src/) | 97 |
| Test files (tests/) | 106 |
| Total Python lines | ~47,000 |
| Source lines (src/ + config/) | ~16,000 |
| Test functions | 1,317 |
| Passing tests | 1,320 (including parametrized) |
| SQL migrations | 12 |
| ORM models | 24 tables |
| API endpoints | ~40 |
| Agent tools | 20 |

---

## 4. Database Schema (24 Tables)

### Core Trading (6 tables)
| Table | Scope | Purpose |
|-------|-------|---------|
| `portfolios` | per-tenant | Current cash + total_value for A and B |
| `positions` | per-tenant | Open positions (ticker, shares, avg_price) |
| `trades` | per-tenant | Executed trade log |
| `daily_snapshots` | per-tenant | End-of-day portfolio values |
| `momentum_rankings` | global | Daily 63-day return rankings for Portfolio A |
| `market_data` | global | OHLCV price cache |

### AI Agent (7 tables)
| Table | Scope | Purpose |
|-------|-------|---------|
| `agent_decisions` | per-tenant | Claude's daily trade proposals + reasoning |
| `agent_memory` | per-tenant | Short-term, weekly summaries, agent notes |
| `agent_conversations` | per-tenant | Persistent conversation sessions (messages JSON) |
| `tool_call_logs` | per-tenant | Per-turn tool usage in agentic sessions |
| `posture_history` | per-tenant | Declared vs effective posture per session |
| `playbook_snapshots` | per-tenant | Regime x sector win rate matrix |
| `conviction_calibration` | per-tenant | Per-conviction-level accuracy stats |

### Market Intelligence (5 tables)
| Table | Scope | Purpose |
|-------|-------|---------|
| `technical_indicators` | global | RSI, MACD, SMAs, Bollinger Bands |
| `macro_data` | global | FRED economic indicators |
| `news_log` | global | Fetched articles + ChromaDB embedding IDs |
| `earnings_calendar` | global | Upcoming earnings dates (yfinance) |
| `discovered_tickers` | per-tenant | AI-proposed tickers with approval workflow |

### Portfolio Management (4 tables)
| Table | Scope | Purpose |
|-------|-------|---------|
| `trailing_stops` | per-tenant | Active trailing stops with peak tracking |
| `watchlist` | per-tenant | AI-managed watchlist (conviction, target entry) |
| `intraday_snapshots` | per-tenant | 15-min portfolio values during market hours |
| `agent_budget_log` | per-tenant | Per-session AI cost tracking |

### Infrastructure (2 tables)
| Table | Scope | Purpose |
|-------|-------|---------|
| `tenants` | -- | Multi-tenant config (encrypted creds, strategy, flags) |
| `weekly_reports` | global | Weekly performance summaries |

### Data Integrity
- **Foreign keys** on all 16 `tenant_id` columns → `tenants(id) ON DELETE CASCADE`
- `PRAGMA foreign_keys=ON` enforced via SQLAlchemy event listener
- Default tenant auto-seeded by `init_db()`

---

## 5. AI Agent System (Portfolio B)

### Three-Level Fallback
1. **Persistent Agent** — Conversation persistence across sessions (if `use_persistent_agent=True`)
2. **Agentic Loop** — Tool-use loop with 8-turn max, $0.50 budget (if `use_agent_loop=True`)
3. **Single-Shot** — One Claude call, parse JSON response (default fallback)

### Tiered Model Routing (if `use_tiered_models=True`)
```
HaikuScanner ($0.002) → ScanResult: ROUTINE | INVESTIGATE | URGENT
    ├── ROUTINE → skip investigation (use scan result directly)
    └── INVESTIGATE/URGENT → Sonnet 4.5 full analysis ($0.10-0.50)
                                 └── if trades proposed → OpusValidator ($0.50-2.00)
```

### Session Profiles
| Profile | When | Behavior |
|---------|------|----------|
| FULL | Morning session | All tools, full investigation |
| LIGHT | Midday check | Reduced tool set, scan-only if routine |
| CRISIS | VIX spike / regime shift | Emergency analysis |
| REVIEW | Post-close | Outcome review, memory consolidation |
| BUDGET_SAVING | Near daily/monthly cap | Haiku only |

### Budget Caps
- Daily: $3.00 (configurable per tenant)
- Monthly: $75.00 (configurable per tenant)
- Per-session: $0.50 (agentic loop hard limit)

### 20 Agent Tools (4 modules)

**Portfolio Tools (6):** get_portfolio_state, get_position_detail, get_portfolio_performance, get_historical_trades, get_correlation_check, get_risk_assessment

**Market Tools (4+1):** get_batch_technicals, get_sector_heatmap, get_market_overview, get_earnings_calendar, get_portfolio_a_history (read-only)

**News Tools (3):** search_historical_news (ChromaDB vectors), get_portfolio_a_status (read-only cross-portfolio visibility)

**Action Tools (6):** execute_trade, set_trailing_stop, get_order_status, save_observation, update_watchlist, declare_posture

### Self-Improvement Loop
- **Playbook Generator**: Weekly regime x sector win rate matrix (sweet_spot / solid / avoid / neutral)
- **Conviction Calibrator**: Per-conviction-level accuracy (validated / overconfident / underconfident)
- **Posture Manager**: Agent declares posture (balanced/defensive/crisis/aggressive), gated by track record (50+ trades, >55% WR, positive alpha for aggressive)
- **Outcome Feedback**: Decision review + track record injected into system prompt each session

---

## 6. Risk Management

### Pre-Trade Checks
- Position size limits (single position %, total portfolio %)
- Sector concentration limits (with per-sector overrides)
- Correlation checks (max portfolio beta)
- Circuit breakers (consecutive loss days)

### Inverse ETF Guardrails
| Rule | Limit |
|------|-------|
| Regime gate | Equity hedges (SH/PSQ/RWM) only in CORRECTION/CRISIS regimes |
| Posture gate | Only defensive/crisis posture |
| Single position | Max 10% of portfolio |
| Total inverse exposure | Max 15% |
| Max inverse positions | 2 simultaneous |
| TBF (rate hedge) | Exempt from regime/posture gates |
| Hold time monitoring | Warning at 3-4 days, review at 5+ days |
| Approval flow | Telegram approval required for inverse BUYs |

### Posture Limits (tighten-only)
Posture limits from the agent can only tighten the hardcoded risk rules, never loosen them. Example: if posture says max 8% position size and rules say 10%, the effective limit is 8%.

---

## 7. Ticker Universe (70+ tickers)

| Category | Count | Examples |
|----------|-------|---------|
| Sector ETFs | 11 | XLK, XLF, XLE, XLV, XLI, XLC, XLY, XLP, XLU, XLRE, XLB |
| Thematic ETFs | 10 | QQQ, SMH, ARKK, TAN, HACK, IBB, KWEB, SOXX, IGV, VNQ |
| Individual Stocks | 40 | AAPL, MSFT, NVDA, TSLA, AMZN, GOOG, META, JPM, ... |
| Defensive | 4 | GLD, TLT, SHY, UUP |
| Inverse ETFs | 4 | SH, PSQ, RWM, TBF |

Portfolio A trades sector + thematic ETFs only. Portfolio B has access to the full universe plus AI-discovered tickers.

### Per-Tenant Customization
- `ticker_whitelist`: Override entire universe
- `ticker_additions`: Add tickers to base universe
- `ticker_exclusions`: Remove tickers from base universe

---

## 8. REST API (FastAPI)

### Authentication
- `POST /api/auth/login` — JWT token (2h expiry, bcrypt password verification)
- `POST /api/auth/logout` — Token revocation (in-memory blacklist)
- Rate limiting: 60 req/min general, 5 req/min login
- CORS: `app.kukulkan.trade` + `localhost:3000`

### Endpoints by Domain

**Account & Portfolios**
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/account` | Alpaca account summary |
| GET | `/api/account/history` | Portfolio history (Alpaca passthrough, 30s cache) |
| GET | `/api/portfolios/{name}` | Portfolio state (cash, total_value) |
| GET | `/api/portfolios/{name}/positions` | Open positions |
| GET | `/api/portfolios/{name}/trailing-stops` | Active trailing stops |
| GET | `/api/portfolios/{name}/watchlist` | AI watchlist items |
| GET | `/api/snapshots` | Daily snapshots |
| GET | `/api/snapshots/intraday` | 15-min intraday snapshots |
| GET | `/api/trades` | Trade history |

**AI Agent**
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/agent/decisions` | Agent decision log |
| GET | `/api/agent/outcomes` | Trade outcomes with P&L |
| GET | `/api/agent/track-record` | Win rate by sector/conviction |
| GET | `/api/agent/decision-quality` | Forward-return accuracy |
| GET | `/api/agent/tool-logs` | Tool call history |
| GET | `/api/agent/conversations` | Persistent conversation list |
| GET | `/api/agent/conversations/{id}` | Single conversation detail |
| GET | `/api/agent/posture` | Current posture + history |
| GET | `/api/agent/playbook` | Regime x sector win rates |
| GET | `/api/agent/calibration` | Conviction accuracy stats |
| GET | `/api/agent/budget` | Daily/monthly cost tracking |
| GET | `/api/agent/inverse-exposure` | Current inverse ETF positions |

**Market Data**
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/momentum/rankings` | Latest momentum rankings |
| GET | `/api/earnings/upcoming` | Upcoming earnings dates |
| GET | `/api/universe/base` | Base ticker universe by sector |

**Discovered Tickers**
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/discovered` | AI-proposed tickers (filterable) |
| PATCH | `/api/discovered/{ticker}` | Approve/reject discovered ticker |

**Tenant Management**
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/tenants` | List all tenants (admin) |
| POST | `/api/tenants` | Create tenant (admin) |
| GET | `/api/tenants/{id}` | Get tenant (admin) |
| PATCH | `/api/tenants/{id}` | Update tenant (admin) |
| DELETE | `/api/tenants/{id}` | Delete tenant (admin) |
| POST | `/api/tenants/{id}/test-alpaca` | Test Alpaca connection (admin) |
| POST | `/api/tenants/{id}/test-telegram` | Test Telegram connection (admin) |
| GET | `/api/tenants/me` | Self-service tenant info |
| PATCH | `/api/tenants/me` | Self-service update (creds, tickers) |
| POST | `/api/tenants/me/test-alpaca` | Self-service Alpaca test |
| POST | `/api/tenants/me/test-telegram` | Self-service Telegram test |

**Operations**
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/run` | Trigger manual pipeline run |

---

## 9. Daily Pipeline (Orchestrator)

The orchestrator runs the full pipeline per tenant per session:

```
Step 1    ─ Fetch market data (yfinance: closes, volumes for universe)
Step 1.5  ─ Fetch macro data (FRED: yield curve, VIX)
Step 2    ─ Classify market regime (BULL / ROTATION / NEUTRAL / BEAR)
Step 2.1  ─ Check trailing stops → generate SELL signals
Step 3    ─ Check circuit breakers (consecutive loss days)
Step 4    ─ Run Portfolio A momentum strategy
Step 5    ─ Run Portfolio B AI agent (persistent → agentic → single-shot)
Step 5.1  ─ Fetch news context (Alpaca → Finnhub → yfinance → ChromaDB historical)
Step 5.5  ─ Refresh earnings calendar
Step 5.6  ─ Clean up expired watchlist items
Step 6    ─ Merge proposed trades from both portfolios
Step 6.5  ─ Resolve posture + apply risk filter (sector limits, posture, inverse guards)
Step 6.6  ─ Check inverse ETF hold times
Step 7    ─ Execute approved trades via Alpaca (with fill verification)
Step 7.1  ─ Create trailing stops for new positions
Step 7.2  ─ Process watchlist updates from AI
Step 8    ─ Record daily snapshots (Alpaca prices preferred, yfinance fallback)
Step 8.5  ─ Reconcile equity drift ($10-$50 threshold)
Step 9    ─ Send Telegram daily brief
Step 10   ─ Process AI-discovered tickers (approval workflow)
```

### Orchestrator Refactoring (Phase 36)
The `run_daily()` method was refactored from ~618 lines to ~120 lines using 4 extracted sub-methods and 2 dataclasses (`MarketContext`, `NewsContext`). Similarly, `_run_portfolio_b()` went from ~500 to ~150 lines using 4 sub-methods and 2 dataclasses (`PortfolioBContext`, `DynamicContext`).

---

## 10. Multi-Tenant System

### Tenant Model
- Each tenant has encrypted Alpaca + Telegram credentials (Fernet)
- Passwords stored as bcrypt hashes
- JWT claims include `tenant_id` for scoping
- 16 child tables have `ForeignKey("tenants.id", ondelete="CASCADE")`
- Orchestrator iterates active tenants, skips those with incomplete credentials

### Tenant Flags
| Flag | Default | Purpose |
|------|---------|---------|
| `run_portfolio_a` | false | Enable momentum portfolio |
| `run_portfolio_b` | true | Enable AI portfolio |
| `use_agent_loop` | false | Agentic tool-use loop |
| `use_persistent_agent` | false | Conversation persistence |
| `use_tiered_models` | false | Haiku/Sonnet/Opus routing |
| `pending_rebalance` | false | Liquidate + redistribute on next run |

### Dynamic Allocations
- Initial equity captured from Alpaca on first run
- Percentage-based splits (`portfolio_a_pct` / `portfolio_b_pct`)
- Deposit detection via Alpaca account activities API (CSD/JNLC)
- Equity reconciliation corrects $10-$50 drift

---

## 11. Reliability & Error Handling (Phase 36)

### Retry Decorators (tenacity)
All retry only on `TRANSIENT_EXCEPTIONS = (ConnectionError, TimeoutError, IOError, OSError)`. Non-transient errors (ValueError, KeyError) propagate immediately.

| Decorator | Attempts | Backoff | Applied To |
|-----------|----------|---------|-----------|
| `retry_market_data` | 3 | exp 1-10s | yfinance fetch_ticker, fetch_universe, get_latest_price |
| `retry_news_api` | 2 | exp 1-5s | Alpaca/Finnhub/yfinance news fetchers |
| `retry_macro_data` | 2 | exp 2-10s | FRED API |
| `retry_broker_read` | 2 | exp 1-5s | Alpaca order/position reads (NOT order submission) |

### Exception Handling
- 15 previously-silent `except: pass` blocks now log via structlog
- Broad `except Exception` narrowed to specific types:
  - Indicator calculations: `(ValueError, KeyError, IndexError)`
  - Notifications: `(ConnectionError, TimeoutError, OSError)`
  - JWT operations: `(JWTError, ValueError)`
  - News/ChromaDB: `(ValueError, KeyError, AttributeError, IOError)`

---

## 12. Deployment

### Infrastructure
- **Server**: Hetzner VPS (128.140.102.191), Ubuntu
- **Services**: 3 systemd units (`kukulkan-bot`, `kukulkan-api`, `kukulkan-fe`) running as `kukulkan` user
- **Reverse proxy**: nginx with Cloudflare Full SSL
- **Domain**: `kukulkan.trade` (landing) / `app.kukulkan.trade` (app)

### CI/CD Pipeline
```
Push to main → GitHub Actions:
  1. pytest tests/ -x -q
  2. ruff check .
  3. rsync to server
  4. python scripts/migrate.py (SQL migrations)
  5. systemctl restart kukulkan-{bot,api}
```

### SQL Migration System
12 migrations tracked in `schema_migrations` table. Supports `--dry-run`. SQLite limitation: no `ALTER COLUMN` — must recreate tables.

---

## 13. Completed Phases

| Phase | Description | Date |
|-------|-------------|------|
| 1-5 | Scaffolding, strategies, Claude agent, orchestrator, Telegram | -- |
| 6-8 | Dashboard, news pipeline + ChromaDB, backtest runner | -- |
| 9-10 | Smart model routing, ticker discovery pipeline | -- |
| 11 | Portfolio restructure (2 portfolios) + multi-executor | -- |
| 12 | GitHub Actions CI/CD + ruff lint | -- |
| 13 | Production hardening (risk, fills, recovery, AI backtest) | -- |
| 14 | Web infra (landing page, Cloudflare DNS/SSL, CI/CD deploy) | -- |
| 15-17 | FastAPI REST API, Next.js frontend, IBKR removal | -- |
| 18-19 | Strategy directives, agent memory, universe expansion (70 tickers) | -- |
| 20 | Bot intelligence (regime, sessions, SPY benchmark, correlation, conviction) | -- |
| 21 | Production bugfixes (fills, news, cash, momentum sizing) | -- |
| 22 | Security hardening (JWT 2h, rate limiting, CORS, audit, non-root) | -- |
| 23-23.2 | Multi-tenant system (encrypted creds, scoping, factories, admin API) | -- |
| 24 | Security audit (IDOR, bcrypt, startup validation, nginx hardening) | -- |
| 25 | Dynamic allocations (equity-based, deposits, per-tenant universe) | -- |
| 26 | Portfolio toggle lifecycle (rebalance flag, liquidation, self-service) | -- |
| 27 | Trailing stops + earnings calendar + dynamic watchlist | -- |
| 28 | Tenant-scoped discovered tickers | -- |
| 29 | ChromaDB historical context in agent pipeline | -- |
| 30-30.1 | Equity reconciliation + deposit detection fix | -- |
| 31 | Intraday portfolio data (15-min snapshots, Alpaca history) | -- |
| 32-32.1 | Agentic Portfolio B (tool-use loop, outcome tracking, 2-phase flow) | -- |
| 33-33.2 | Persistent agent, upgraded toolkit (20 tools), posture/playbook/calibration | -- |
| 34 | Compute optimization (tiered models, prompt caching, budget caps) | -- |
| 35 | Inverse ETF / market hedging (regime/posture gates, approval flow) | 2026-02-16 |
| 36 | Code quality (retry decorators, exception handling, orchestrator refactor, FK constraints) | 2026-02-16 |

---

## 14. Key File Map

### Core Pipeline
| File | Lines | Purpose |
|------|-------|---------|
| `src/orchestrator.py` | 2,910 | Daily pipeline, all steps, tenant iteration |
| `config/universe.py` | 303 | 70+ tickers, sector maps, instrument classification |
| `config/settings.py` | ~120 | Pydantic v2 settings, env-based |
| `config/risk_rules.py` | ~50 | RiskRules dataclass, sector overrides |

### AI Agent
| File | Lines | Purpose |
|------|-------|---------|
| `src/agent/claude_agent.py` | 741 | Claude API calls, system prompt building |
| `src/agent/agent_runner.py` | 339 | Agentic tool-use loop (8 turns, $0.50 budget) |
| `src/agent/persistent_agent.py` | 259 | Conversation persistence wrapper |
| `src/agent/tiered_runner.py` | 298 | Haiku→Sonnet→Opus flow orchestration |
| `src/agent/haiku_scanner.py` | ~100 | Fast market triage ($0.002) |
| `src/agent/opus_validator.py` | ~100 | Trade validation review |
| `src/agent/budget_tracker.py` | ~120 | Daily/monthly cost enforcement |
| `src/agent/posture.py` | ~80 | Posture enum, limits, aggressive gate |
| `src/agent/token_tracker.py` | ~100 | Per-session token/cost tracking |
| `src/agent/tools/` | 4 modules | 20 tools across portfolio/market/news/actions |

### Storage
| File | Lines | Purpose |
|------|-------|---------|
| `src/storage/models.py` | 632 | 24 ORM models + Pydantic schemas |
| `src/storage/database.py` | 1,470 | Async CRUD, FK pragma, ensure_tenant() |

### API
| File | Lines | Purpose |
|------|-------|---------|
| `src/api/main.py` | 120 | FastAPI app, middleware, lifespan |
| `src/api/auth.py` | ~100 | JWT create/decode/revoke, login/logout |
| `src/api/deps.py` | ~60 | Auth dependencies, tenant scoping |
| `src/api/routes/` | 15 files | All REST endpoints |

### Analysis
| File | Purpose |
|------|---------|
| `src/analysis/risk_manager.py` | Pre-trade risk checks, inverse guardrails, circuit breakers |
| `src/analysis/regime_classifier.py` | Market regime detection |
| `src/analysis/outcome_tracker.py` | Trade P&L + alpha vs sector ETF/SPY |
| `src/analysis/track_record.py` | Win rate by sector/conviction/regime |
| `src/analysis/decision_quality.py` | Forward-return accuracy (1d/3d/5d) |
| `src/analysis/playbook_generator.py` | Regime x sector win rate matrix |
| `src/analysis/conviction_calibrator.py` | Per-conviction accuracy stats |

### Data
| File | Purpose |
|------|---------|
| `src/data/market_data.py` | yfinance OHLCV fetch (with retry) |
| `src/data/news_fetcher.py` | News pipeline + ChromaDB storage |
| `src/data/news_aggregator.py` | Multi-source deduplication |
| `src/data/alpaca_news.py` | Alpaca news API |
| `src/data/finnhub_news.py` | Finnhub news API |
| `src/data/macro_data.py` | FRED economic indicators |
| `src/data/earnings_calendar.py` | Upcoming earnings via yfinance |

---

## 15. Known Limitations / Technical Debt

1. **SQLite single-writer**: No concurrent write support. Fine for single-bot, would need PostgreSQL for horizontal scaling.
2. **In-memory token revocation**: JWT blacklist lost on API restart. Acceptable with 2h expiry.
3. **yfinance rate limits**: No official API key, relies on web scraping. Retry decorators help but extended outages not handled.
4. **ChromaDB single instance**: No replication. Docker container on same server.
5. **No WebSocket push**: Frontend polls API. Real-time updates would need WebSocket or SSE.
6. **Test fixtures use `Database.__new__()` bypass**: Some API tests skip FK enforcement for simplicity.
7. **No automated integration tests against live Alpaca paper**: All broker tests are mocked.
