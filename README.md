# Kukulkan Trade

Educational trading bot running 2 portfolio strategies on Alpaca paper trading.
Built with Python 3.11, async SQLAlchemy, Claude AI, and a full notification + dashboard stack.

## Portfolios

| Portfolio | Strategy | Allocation | Rebalance |
|-----------|----------|------------|-----------|
| A | Aggressive Momentum (top 1 ETF) | $33,000 | Daily |
| B | AI Full Autonomy (Claude decides) | $66,000 | Daily |

## Architecture

```
                          ┌──────────────┐
                          │  Orchestrator │
                          └──────┬───────┘
                   ┌─────────────┼─────────────┐
                   ▼             ▼              ▼
            ┌────────────┐ ┌──────────┐ ┌─────────────┐
            │ Portfolio A │ │Portfolio B│ │ News Fetcher│
            │  Momentum   │ │ AI Agent │ │  + ChromaDB │
            └─────┬──────┘ └────┬─────┘ └─────────────┘
                  │              │
                  ▼              ▼
            ┌─────────────────────────┐
            │  Risk Manager + Filter  │
            └────────────┬────────────┘
                         ▼
            ┌─────────────────────────┐
            │   Executor (Alpaca)     │
            └────────────┬────────────┘
                         ▼
          ┌──────────────────────────────┐
          │  FastAPI  │  Next.js  │  TG  │
          └──────────────────────────────┘
```

## Tech Stack

Python 3.11 | FastAPI | SQLAlchemy + SQLite | ChromaDB | yfinance | `ta` | Anthropic Claude | Alpaca | Telegram | Next.js

## Features

- **Market Regime Classifier** — 5 regimes (bull, bear, correction, crisis, consolidation) with adaptive allocation rules
- **Session-Aware Prompts** — Morning/Midday/Closing directives injected into AI context
- **SPY Benchmarking** — Alpha tracking vs S&P 500, included in weekly reports
- **Conviction-Based Sizing** — AI sets high/medium/low conviction per trade (multipliers: 1.0/0.7/0.4)
- **Risk Management** — Pre-trade filtering, sector concentration limits, circuit breakers, correlation monitoring
- **Agent Memory** — 3-tier system: short-term decisions, weekly summaries, persistent notes
- **Multi-Tenant** — Fernet-encrypted credentials, per-tenant data isolation, admin API + CLI
- **Security** — JWT auth (2h expiry + revocation), rate limiting, timing-safe auth, audit logging
- **SQL Migrations** — Automated schema migrations in CI/CD with manual trigger option
- **70-Ticker Universe** — ETFs + stocks across sectors, fixed income, international, thematic

## Setup

```bash
# Clone and enter the project
cd kukulkan-trade

# Create virtualenv (Python 3.11)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Copy and fill environment variables
cp .env.example .env
# Edit .env — required keys:
#   ANTHROPIC_API_KEY      — Claude API
#   ALPACA_API_KEY         — Alpaca paper trading
#   ALPACA_SECRET_KEY      — Alpaca secret
#   EXECUTOR               — alpaca | paper
#   TELEGRAM_BOT_TOKEN     — Telegram notifications
#   TELEGRAM_CHAT_ID       — your chat ID
#   JWT_SECRET             — API authentication
#   TENANT_ENCRYPTION_KEY  — Fernet key for credential encryption
#   FRED_API_KEY           — macro data (optional)

# Start ChromaDB (Docker)
docker compose up -d

# Run tests
pytest tests/ -v

# Start the API
python -m src.api.main
```

## Running

```bash
# Single run (execute pipeline now and exit)
python -m src.main --run-now

# Scheduler mode (runs daily at market open)
python -m src.main

# Choose executor
EXECUTOR=alpaca python -m src.main --run-now   # Alpaca paper trading (default)
EXECUTOR=paper  python -m src.main --run-now   # Local simulation
```

## API

FastAPI REST API on port 8001 with JWT authentication.

```
POST /api/auth/login          — Get access token
POST /api/auth/logout         — Revoke token
GET  /api/account             — Alpaca account + positions
GET  /api/portfolios          — Portfolio summaries (A & B)
GET  /api/portfolios/{name}   — Portfolio detail with positions
GET  /api/snapshots           — Daily performance snapshots
GET  /api/trades              — Trade history
GET  /api/momentum            — Momentum rankings
GET  /api/decisions           — AI agent decisions
POST /api/tenants             — Create tenant (admin)
GET  /api/tenants             — List tenants (admin)
PATCH /api/tenants/{id}       — Update tenant config (admin)
```

## Tenant Management

```bash
# CLI commands
python -m src.cli.tenant_cli add-tenant --name "User" --username user1 --password pass123
python -m src.cli.tenant_cli list-tenants
python -m src.cli.tenant_cli seed-default  # Create default tenant from .env

# Tenants can be created with just login credentials.
# Alpaca/Telegram are configured later via PATCH.
# Bot skips tenants without complete credentials.
```

## Backtesting

```bash
# Standard backtest
python -m src.backtest.runner --start 2024-01-01 --end 2024-12-31

# AI backtest (uses Claude, costs apply)
python -m src.backtest.runner --use-ai --ai-strategy conservative --ai-budget 5.00

# Dry run (no API calls)
python -m src.backtest.runner --use-ai --dry-run
```

## Database Migrations

```bash
# Run pending migrations
python scripts/migrate.py --db data/kukulkan.db

# Preview without applying
python scripts/migrate.py --dry-run

# Migrations run automatically during deploy (between pip install and service restart)
# Manual trigger: GitHub Actions → "Run DB Migrations" → "Run workflow"
```

## Deployment

The bot runs on a Hetzner VPS as systemd services. Pushing to `main` triggers GitHub Actions to lint, test, and auto-deploy.

```bash
# Service management
sudo systemctl start kukulkan-bot
sudo systemctl start kukulkan-api
sudo journalctl -u kukulkan-bot -f
```

**GitHub Actions** (`.github/workflows/deploy.yml`):
1. **Test** — lint with `ruff`, run `pytest`
2. **Deploy** — rsync to server, install deps, run migrations, restart services

Required secrets: `DEPLOY_SSH_KEY`, `DEPLOY_HOST`, `DEPLOY_USER`

## Project Structure

```
kukulkan-trade/
├── config/
│   ├── settings.py            # Pydantic Settings (env-based config)
│   ├── strategies.py          # Strategy parameter dataclasses
│   ├── universe.py            # 70-ticker universe with sector mapping
│   └── risk_rules.py          # Position size & risk limits
├── src/
│   ├── agent/
│   │   ├── claude_agent.py        # Claude AI analysis & trade decisions
│   │   ├── complexity_detector.py # Smart model routing (Haiku/Opus)
│   │   ├── memory.py             # 3-tier agent memory system
│   │   ├── strategy_directives.py # Strategy + session + regime prompts
│   │   └── ticker_discovery.py    # AI-suggested ticker additions
│   ├── analysis/
│   │   ├── momentum.py            # 63-day momentum with 5-day skip
│   │   ├── performance.py         # Portfolio stats + SPY benchmarking
│   │   ├── regime.py             # Market regime classifier (5 regimes)
│   │   ├── risk_manager.py       # Pre-trade filtering + circuit breakers
│   │   └── technical.py           # RSI, MACD, SMA, Bollinger Bands
│   ├── api/
│   │   ├── main.py               # FastAPI app + security middleware
│   │   ├── auth.py               # JWT auth + tenant login
│   │   ├── deps.py               # Dependency injection (auth, db)
│   │   ├── rate_limit.py         # Sliding-window rate limiter
│   │   ├── schemas.py            # Pydantic request/response models
│   │   └── routes/               # 7 route modules
│   ├── backtest/
│   │   ├── runner.py              # Historical strategy backtesting
│   │   └── ai_strategy.py        # AI backtest with budget tracking
│   ├── cli/
│   │   └── tenant_cli.py         # Tenant management CLI
│   ├── data/
│   │   ├── market_data.py         # yfinance price fetcher
│   │   ├── macro_data.py          # FRED yield curve & VIX
│   │   ├── news_fetcher.py        # News + ChromaDB vector search
│   │   ├── news_aggregator.py     # Multi-source news collection
│   │   └── news_compactor.py      # Token-efficient news summarization
│   ├── execution/
│   │   ├── alpaca_executor.py     # Alpaca API executor
│   │   ├── client_factory.py     # Per-tenant Alpaca client cache
│   │   └── paper_trader.py        # Local paper trading simulation
│   ├── notifications/
│   │   ├── telegram_bot.py        # Daily briefs & trade alerts
│   │   ├── telegram_factory.py   # Per-tenant Telegram cache
│   │   └── weekly_report.py       # Friday performance report
│   ├── storage/
│   │   ├── models.py              # 16 SQLAlchemy tables + Pydantic schemas
│   │   ├── database.py            # Async CRUD operations
│   │   └── vector_store.py        # ChromaDB client
│   ├── strategies/
│   │   ├── portfolio_a.py         # Momentum strategy
│   │   └── portfolio_b.py         # AI Autonomy strategy
│   ├── utils/
│   │   ├── crypto.py             # Fernet encryption for credentials
│   │   ├── market_calendar.py    # NYSE trading calendar
│   │   └── tenant_universe.py    # Per-tenant ticker resolution
│   ├── orchestrator.py            # Daily pipeline coordinator
│   └── main.py                    # Entry point + APScheduler
├── migrations/                    # SQL migration files
├── scripts/
│   └── migrate.py                 # Migration runner
├── tests/                         # 581 tests
├── deploy/
│   ├── kukulkan-bot.service       # Bot systemd unit
│   ├── kukulkan-api.service       # API systemd unit
│   └── nginx/kukulkan.trade       # Nginx config
├── .github/workflows/
│   ├── deploy.yml                 # CI/CD: test → deploy → migrate → restart
│   └── migrate.yml                # Standalone migration workflow
├── docker-compose.yml             # ChromaDB service
├── pyproject.toml
└── CLAUDE.md                      # Development rules
```
