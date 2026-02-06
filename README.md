# Atlas Trading Bot

Educational trading bot running 3 parallel strategies on IBKR paper trading.

## Portfolios

| Portfolio | Strategy | Allocation | Rebalance |
|-----------|----------|------------|-----------|
| A | Aggressive Momentum (top 1 ETF) | $33,333 | Daily |
| B | Sector Rotation + Macro + Contrarian (7-factor) | $33,333 | Weekly |
| C | AI Full Autonomy (Claude decides) | $33,333 | Daily |

## Setup

```bash
# 1. Clone and enter the project
cd atlas-trading-bot

# 2. Create virtualenv (pyenv + Python 3.11)
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
make install

# 4. Copy and fill environment variables
cp .env.example .env
# Edit .env with your API keys

# 5. Start infrastructure (ChromaDB)
make infra-up

# 6. Run tests
make test

# 7. Launch dashboard
make dashboard
```

## Project Structure

```
atlas-trading-bot/
├── config/              # Settings, universe, strategy params, risk rules
├── src/
│   ├── agent/           # Claude AI integration
│   ├── analysis/        # Momentum, technical indicators, macro
│   ├── dashboard/       # Streamlit UI
│   ├── data/            # Market data fetching (yfinance / IBKR)
│   ├── execution/       # Paper trader and IBKR executor
│   ├── notifications/   # Telegram bot
│   ├── storage/         # SQLite models, database, ChromaDB vector store
│   └── strategies/      # Portfolio A, B, C implementations
├── tests/
├── data/                # SQLite database (gitignored)
└── docker-compose.yml   # ChromaDB service
```

## Tech Stack

Python 3.11 | SQLAlchemy + SQLite | ChromaDB | yfinance | pandas-ta | Anthropic Claude API | Telegram | Streamlit
