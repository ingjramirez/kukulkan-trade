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
            │  Executor (Alpaca/IBKR) │
            └────────────┬────────────┘
                         ▼
            ┌─────────────────────────┐
            │  Telegram + Dashboard   │
            └─────────────────────────┘
```

## Tech Stack

Python 3.11 | SQLAlchemy + SQLite | ChromaDB | yfinance | `ta` | Anthropic Claude | Alpaca | Telegram | Streamlit | Plotly

## Setup

```bash
# Clone and enter the project
cd kukulkan-trade

# Create virtualenv (Python 3.11)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
make install

# Copy and fill environment variables
cp .env.example .env
# Edit .env — required keys:
#   ANTHROPIC_API_KEY   — Claude API
#   ALPACA_API_KEY      — Alpaca paper trading
#   ALPACA_SECRET_KEY   — Alpaca secret
#   EXECUTOR            — alpaca | ibkr | paper
#   TELEGRAM_BOT_TOKEN  — Telegram notifications
#   TELEGRAM_CHAT_ID    — your chat ID
#   FRED_API_KEY        — macro data (optional)

# Start ChromaDB (Docker)
make infra-up

# Run tests
make test

# Launch dashboard
make dashboard
```

## Running

```bash
# Single run (execute pipeline now and exit)
python -m src.main --run-now

# Scheduler mode (runs daily at market open)
python -m src.main

# Choose executor
EXECUTOR=alpaca python -m src.main --run-now   # Alpaca paper trading (default)
EXECUTOR=ibkr   python -m src.main --run-now   # Interactive Brokers
EXECUTOR=paper  python -m src.main --run-now   # Local simulation
```

## Deployment

The bot runs on a Hetzner VPS as a systemd service. Pushing to `main` triggers GitHub Actions to lint, test, and auto-deploy.

```bash
# First-time server setup
sudo bash deploy/setup.sh

# Service management
sudo systemctl start kukulkan-bot
sudo systemctl status kukulkan-bot
sudo journalctl -u kukulkan-bot -f
```

**GitHub Actions** (`.github/workflows/deploy.yml`):
1. **Test** — lint with `ruff`, run `pytest`
2. **Deploy** — rsync to server, `pip install`, restart systemd service

Required secrets: `DEPLOY_SSH_KEY`, `DEPLOY_HOST`, `DEPLOY_USER`

## Project Structure

```
kukulkan-trade/
├── config/
│   ├── settings.py          # Pydantic Settings (env-based config)
│   ├── strategies.py        # Strategy parameter dataclasses
│   ├── universe.py          # Ticker universe (~47 symbols)
│   └── risk_rules.py        # Position size & risk limits
├── src/
│   ├── agent/
│   │   ├── claude_agent.py      # Claude AI analysis & trade decisions
│   │   ├── complexity_detector.py # Smart model routing (Haiku/Opus)
│   │   └── ticker_discovery.py  # AI-suggested ticker additions
│   ├── analysis/
│   │   ├── momentum.py          # 63-day momentum with 5-day skip
│   │   └── technical.py         # RSI, MACD, SMA, Bollinger Bands
│   ├── backtest/
│   │   └── runner.py            # Historical strategy backtesting
│   ├── dashboard/
│   │   └── app.py               # Streamlit UI (5 pages + Plotly)
│   ├── data/
│   │   ├── market_data.py       # yfinance price fetcher
│   │   ├── macro_data.py        # FRED yield curve & VIX
│   │   └── news_fetcher.py      # News + ChromaDB vector search
│   ├── execution/
│   │   ├── alpaca_executor.py   # Alpaca API executor
│   │   ├── ibkr_executor.py     # IBKR TWS executor
│   │   └── paper_trader.py      # Local paper trading simulation
│   ├── notifications/
│   │   └── telegram_bot.py      # Daily briefs & trade alerts
│   ├── storage/
│   │   ├── models.py            # 12 SQLAlchemy tables
│   │   ├── database.py          # Async CRUD operations
│   │   └── vector_store.py      # ChromaDB client
│   ├── strategies/
│   │   ├── portfolio_a.py       # Momentum strategy
│   │   └── portfolio_b.py       # AI Autonomy strategy
│   ├── orchestrator.py          # Daily pipeline coordinator
│   └── main.py                  # Entry point + APScheduler
├── tests/                       # 287 tests
├── deploy/
│   ├── kukulkan-bot.service      # systemd unit file
│   └── setup.sh                 # Server provisioning script
├── docker-compose.yml           # ChromaDB service
├── pyproject.toml
└── Makefile
```
