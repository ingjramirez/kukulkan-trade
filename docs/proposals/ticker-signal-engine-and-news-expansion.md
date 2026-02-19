# Proposal: Local Signal Engine + Global News Expansion

**Author**: Claude (Phase 45 follow-up)
**Date**: 2026-02-19
**Status**: Draft for PM/Analyst review

---

## Problem Statement

The agent only sees 22% of the universe (16/72 tickers in 30 days). Phase 45 added passive `build_universe_opportunities()` but it's a static snapshot — just 20d momentum + RSI at pipeline time. Meanwhile:

- **No continuous ranking**: The system only evaluates tickers 3x/day during pipeline runs
- **No cross-signal synthesis**: Momentum, technicals, and news are siloed — no combined score
- **News blind spots**: Only US sources (Alpaca/Benzinga + Finnhub). No European, Asian, or sentiment indicators
- **Agent can't discover what it doesn't see**: The agent's tools only work on tickers it already knows about

## Proposal: Two Complementary Systems

### System A: Local Ticker Signal Engine (no Anthropic dependency)

A lightweight Python batch job using **only numpy/pandas/ta-lib** (already installed) that runs every 10 minutes during market hours. No ML model training needed — this is a **rule-based scoring engine** using proven quantitative signals.

### System B: Global News Sources + Sentiment Indicators

Expand the news pipeline with 3-5 new sources focusing on international coverage and composite sentiment, feeding into both the Signal Engine and the agent's news tools.

---

## System A: Ticker Signal Engine

### Architecture

```
┌─────────────────────────────────────────────────┐
│  SignalEngine (runs every 10 min, market hours)  │
│                                                   │
│  Input: yfinance 6mo closes (cached in SQLite)   │
│                                                   │
│  Signals per ticker (all pure pandas/numpy):     │
│  ┌───────────┐ ┌──────────┐ ┌───────────────┐   │
│  │ Momentum  │ │Technical │ │  Volatility   │   │
│  │ Score     │ │ Score    │ │  Score        │   │
│  └─────┬─────┘ └────┬─────┘ └──────┬────────┘   │
│        └────────────┼───────────────┘            │
│                     ▼                             │
│            Composite Rank (0-100)                │
│                     │                             │
│  ┌──────────────────┼──────────────────────────┐ │
│  │ Change Detection │                          │ │
│  │ "NVDA jumped from rank 45→8 in 2 hours"    │ │
│  │ "XLE RSI crossed below 30"                  │ │
│  │ "GLD golden cross (SMA20 > SMA50)"         │ │
│  └──────────────────┼──────────────────────────┘ │
│                     ▼                             │
│  Output: SQLite table + SSE event + agent context│
└─────────────────────────────────────────────────┘
```

### Signal Components (all computable from price data we already have)

| Signal | Weight | Computation | Library |
|-|-|-|-|
| Momentum 20d | 20% | 20-day return | pandas |
| Momentum 63d | 15% | 63-day return (skip 5) | pandas |
| RSI position | 15% | Distance from 50 (extremes score higher) | ta |
| MACD histogram | 10% | Positive = bullish, negative = bearish | ta |
| SMA trend | 15% | Price vs SMA20/50/200 (above = bullish) | ta |
| Bollinger %B | 10% | Position within bands | ta |
| Volume surge | 15% | Today's volume / 20d avg volume | pandas |

**Composite score**: Weighted sum normalized to 0-100. Each signal is z-scored across the universe first so no single ticker dominates.

### Change Detection (the real value)

Static rankings aren't enough — **rank velocity** is the signal:

```python
@dataclass
class TickerSignal:
    ticker: str
    composite_score: float      # 0-100
    rank: int                   # 1-72
    prev_rank: int              # rank 10 min ago
    rank_velocity: float        # rank change per hour
    momentum_20d: float
    rsi: float
    volume_ratio: float         # vs 20d avg
    triggered_alerts: list[str] # ["golden_cross", "rsi_oversold", "volume_spike"]
    updated_at: datetime
```

Alert rules (configurable):
- **Rank jump**: Moved 10+ ranks in 1 hour
- **RSI cross**: Crossed 30 (oversold entry) or 70 (overbought exit)
- **Golden/death cross**: SMA20 crosses SMA50
- **Volume spike**: Volume > 2x 20-day average
- **Bollinger breakout**: Price breaks above upper / below lower band
- **New high/low**: 20-day high or low

### Implementation

**New file**: `src/analysis/signal_engine.py`

```python
class SignalEngine:
    """Batch ticker ranking engine — runs every 10 min, pure pandas/numpy."""

    def __init__(self, db: Database) -> None:
        self._db = db
        self._prev_ranks: dict[str, int] = {}  # in-memory rank history

    async def run(self, closes: pd.DataFrame, volumes: pd.DataFrame) -> list[TickerSignal]:
        """Score and rank all tickers. Detect changes from previous run."""

    def _compute_momentum_score(self, closes: pd.DataFrame) -> pd.Series: ...
    def _compute_technical_score(self, closes: pd.DataFrame) -> pd.Series: ...
    def _compute_volatility_score(self, closes: pd.DataFrame, volumes: pd.DataFrame) -> pd.Series: ...
    def _detect_alerts(self, ticker: str, indicators: dict) -> list[str]: ...
```

**New SQLite model**: `TickerSignalRow`

```python
class TickerSignalRow(Base):
    __tablename__ = "ticker_signals"
    id: int (PK)
    tenant_id: str (FK)
    ticker: str
    composite_score: float
    rank: int
    rank_velocity: float
    momentum_20d: float
    rsi: float
    volume_ratio: float
    alerts: str  # JSON list
    scored_at: datetime
```

**Scheduler job** in `main.py`:

```python
scheduler.add_job(
    signal_engine_job,
    CronTrigger(minute="*/10", hour="9-16", day_of_week="mon-fri", timezone="US/Eastern"),
    id="signal_engine",
)
```

**Agent integration**: Replace `build_universe_opportunities()` output with signal engine data:

```
## Universe Signal Rankings (updated 5 min ago)
Biggest movers (rank velocity):
  NVDA: rank 3 (was 28, +25 in 2h) — volume 3.2x avg, RSI 62, MACD bullish
  XLE: rank 12 (was 41, +29 in 4h) — golden cross, volume 1.8x avg

Top 5 non-held by composite score:
  1. META (score 87) — momentum +8.2%, RSI 58, above all SMAs
  2. JPM (score 81) — momentum +5.1%, volume surge 2.1x
  ...

Alerts triggered since last session:
  - AAPL: RSI crossed below 30 (oversold)
  - GLD: Bollinger upper band breakout
  - XLU: 20-day low
```

**API endpoint**: `GET /api/signals/rankings` — for the frontend dashboard.

### Cost: $0 per run

No API calls. No model inference. Pure numpy/pandas on data we already cache in SQLite. The `ta` library is already installed. yfinance data is already fetched 3x/day; we'd add a lightweight intraday price refresh (yfinance `fast_info` — same pattern as extended hours snapshots).

### Data Freshness

The engine uses **cached closes from SQLite** (fetched at pipeline time, 3x daily) plus **intraday fast_info** (already fetched every 15 min for snapshots). No additional API calls needed for the base case. For the 10-min ranking refresh, we recompute scores using the latest intraday snapshot data.

---

## System B: Global News Sources + Sentiment

### Current State (from audit)

| Source | Type | Coverage | Rate Limit |
|-|-|-|-|
| Alpaca (Benzinga) | API | US financial news | Alpaca SDK limits |
| Finnhub | API | US financial + general | 60 req/min free |
| yfinance | Fallback | US ticker news | Fallback only |

Storage: SQLite (`NewsLogRow`) + ChromaDB (semantic search). Articles normalized via `NewsArticle` dataclass. No formal fetcher ABC — each is independent.

### Proposed New Sources (prioritized by value/effort)

#### Priority 1: Fear & Greed Index (highest value, lowest effort)

**Why**: Single number (0-100) that captures 7 market indicators. The agent currently has VIX only for sentiment. Fear & Greed adds: market momentum, stock price strength/breadth, put/call ratio, junk bond demand, safe haven demand.

**Implementation**: Not a news source — a **sentiment indicator**. Scrape from CNN or use Alternative.me API (free, no auth):
- `https://api.alternative.me/fng/` — returns current value + historical
- Response: `{"value": "25", "value_classification": "Extreme Fear"}`

**Storage**: New model `SentimentIndicatorRow` (name, value, classification, timestamp, sub_indicators JSON).

**Agent integration**: Inject into macro context alongside VIX:
```
Macro: VIX 18.3, Fear & Greed 25 (Extreme Fear), Yield Curve +0.4%
```

**Effort**: ~2 hours. No auth, simple JSON API, tiny storage.

#### Priority 2: Reddit Sentiment Scanner

**Why**: Retail sentiment is a contrarian indicator. WSB momentum predicts short-term vol spikes. Reddit API is free (100 req/min, OAuth2 app-only).

**Implementation**:
- New fetcher: `src/data/reddit_news.py`
- Subreddits: r/wallstreetbets, r/stocks, r/investing
- Extract ticker mentions via regex (`$NVDA`, `NVDA` in flair)
- Filter: score > 100 upvotes, extract top 10 per subreddit
- **Feed into Signal Engine**: `reddit_mentions` as a signal component (not a weight — an alert flag)

**Storage**: `NewsArticle` with `source="reddit"`, sentiment from upvote ratio.

**Schedule**: Every 2 hours during US market hours.

**Effort**: ~4 hours. Reddit OAuth is straightforward. Main work is ticker extraction regex + noise filtering.

#### Priority 3: CNN/Reuters RSS (international macro)

**Why**: Free RSS feeds with no auth required. Reuters covers global macro events that move US markets.

**Implementation**:
- RSS parser using `feedparser` (well-maintained, BSD license)
- Feeds: Reuters Business, Reuters Markets, CNN Business
- No auth, no rate limits (standard RSS etiquette: fetch every 30 min)

**Storage**: Same `NewsArticle` pipeline. ChromaDB for semantic search.

**Effort**: ~3 hours. RSS is the simplest integration pattern.

#### Priority 4: Nikkei Asia / SCMP (Asian market context)

**Why**: Asian market moves overnight predict US open. Semiconductor supply chain news (TSMC, Samsung) drives NVDA/AMD/SMH.

**Implementation**:
- RSS feeds (both have them)
- Schedule: 7 PM - 9 AM ET (Asian market hours)
- Tag with `region: "asia"` for agent context filtering

**Effort**: ~3 hours each. Same RSS pattern as Priority 3.

#### Priority 5: Der Aktionär (European perspective)

**Why**: European perspective on US stocks. Lowest priority because overlap with Reuters is high.

**Implementation**: RSS if available, otherwise skip. German language is fine — agent processes it natively.

**Effort**: ~2 hours if RSS exists, ~6 hours if web scraping needed.

### Fetcher Standardization

Before adding sources, create a lightweight ABC to standardize the interface:

```python
class BaseNewsFetcher(ABC):
    source_name: str
    rate_limit_per_min: int

    @abstractmethod
    async def fetch(self, tickers: list[str] | None = None) -> list[NewsArticle]: ...
```

This costs ~1 hour but pays for itself immediately with 3-5 new sources.

---

## Combined Architecture

```
                    ┌─────────────────────────────┐
                    │   Data Layer (existing)      │
                    │   yfinance → SQLite cache    │
                    │   + 15-min intraday refresh  │
                    └──────────┬──────────────────┘
                               │
          ┌────────────────────┼────────────────────────┐
          │                    │                         │
          ▼                    ▼                         ▼
 ┌─────────────────┐ ┌────────────────┐ ┌──────────────────────┐
 │ Signal Engine    │ │ News Pipeline  │ │ Sentiment Indicators │
 │ (every 10 min)  │ │ (every 30 min) │ │ (2x daily)           │
 │ Pure pandas/ta  │ │ 5+ sources     │ │ Fear&Greed, Reddit   │
 │ 72 tickers      │ │ RSS + APIs     │ │ sentiment score      │
 └────────┬────────┘ └───────┬────────┘ └──────────┬───────────┘
          │                   │                      │
          └───────────────────┼──────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Agent Context Builder         │
              │  (morning/midday/close)        │
              │                                │
              │  Macro: VIX 18, F&G 25 (Fear) │
              │  Signals: NVDA rank 3 (+25),  │
              │    volume 3.2x, golden cross   │
              │  News: TSMC earnings beat,     │
              │    WSB hot: NVDA (45 posts)    │
              └───────────────┬───────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Claude Agent   │
                    │  (Portfolio B)  │
                    │  Full context   │
                    │  → better       │
                    │    decisions    │
                    └─────────────────┘
```

---

## Implementation Phases

### Phase 46: Signal Engine Core (~1 day)

- `src/analysis/signal_engine.py` — scoring, ranking, change detection
- `TickerSignalRow` model + migration
- Scheduler job (every 10 min, market hours)
- Replace `build_universe_opportunities()` output with signal data in trigger messages
- `GET /api/signals/rankings` endpoint
- ~40 tests

### Phase 47: News Infrastructure + Fear & Greed (~0.5 day)

- `BaseNewsFetcher` ABC
- Refactor Alpaca/Finnhub fetchers to extend ABC
- `SentimentIndicatorRow` model + migration
- `src/data/fear_greed.py` — Alternative.me API fetcher
- Inject into macro context
- ~20 tests

### Phase 48: Reddit + RSS Sources (~1 day)

- `src/data/reddit_news.py` — OAuth2, ticker extraction, noise filtering
- `src/data/rss_news.py` — generic RSS fetcher (Reuters, CNN, Nikkei, SCMP)
- Register all sources in news aggregator
- Update compactor to handle multi-region articles
- Feed Reddit mention counts into Signal Engine as alert flag
- ~30 tests

### Phase 49: Agent Tool Enhancements (~0.5 day)

- New tool: `get_signal_rankings` — top movers, alerts, rank velocity
- Enhance `search_news` to filter by region
- Update agent system prompt to reference signal data
- ~15 tests

**Total**: ~3 days, ~105 tests, 8-10 new files, 5-6 modified files.

---

## Dependency Summary

| Dependency | Already Installed | License | Purpose |
|-|-|-|-|
| pandas, numpy | Yes | BSD | Signal computation |
| ta | Yes | MIT | RSI, MACD, BB, SMA |
| yfinance | Yes | Apache 2.0 | Price data (existing) |
| feedparser | **No** | BSD | RSS parsing |
| httpx | Yes | BSD | Reddit API, Fear&Greed API |

Only **one new dependency** (`feedparser`) — well-maintained, BSD license, widely used.

---

## Key Design Decisions for PM/Analyst

1. **Signal Engine runs locally (no API cost)** — all computation uses data we already have. The value is in the *synthesis* (combining 7 signals into one rank) and *change detection* (rank velocity alerts).

2. **Not ML, but quantitative scoring** — ML models need training data and can overfit. A rule-based scoring engine is transparent, explainable, and the agent can reason about the components. If we want ML later, the Signal Engine's historical scores become perfect training features.

3. **News sources are additive, not replacement** — Alpaca/Finnhub remain primary for US ticker-specific news. New sources add geographic and sentiment coverage.

4. **Fear & Greed is the highest-ROI addition** — one number that captures 7 market indicators, free API, 2 hours to implement. The agent currently has no composite sentiment beyond VIX.

5. **Reddit is a contrarian indicator, not a signal** — WSB consensus is often wrong, but WSB *volume of attention* predicts volatility. We track mention counts, not sentiment direction.

6. **10-minute refresh is sufficient** — the agent runs 3x/day, not HFT. 10-minute ranking updates ensure the agent sees intraday momentum shifts when it wakes up, without overwhelming SQLite.

---

## Risk / Gotchas

- **yfinance rate limits**: The 10-min signal engine reuses cached data + 15-min intraday snapshots. No additional yfinance calls needed unless we want more granular intraday data.
- **Reddit API changes**: Reddit has tightened API access (2023 pricing change). Free tier (100 req/min) still works for our volume. If they restrict further, Reddit becomes the first source we'd cut.
- **ChromaDB must be running**: New sources store in ChromaDB. If ChromaDB is down, articles still go to SQLite (existing fallback pattern).
- **RSS feeds can change URLs**: Need periodic URL validation. `feedparser` handles most format variations gracefully.
- **Signal Engine ≠ trading signals**: This is an *attention routing* system, not an alpha generator. It tells the agent "look here" — the agent still makes all trade decisions.
