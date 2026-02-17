# Analysis & Strategies

Machine-readable context for Claude. Covers regime classification, momentum, technical indicators, risk management, performance metrics, outcome tracking, track record, and decision quality.

## Key Files

| File | Purpose |
|------|---------|
| `src/analysis/regime.py` | RegimeClassifier: 5-point market state from SPY/VIX/breadth |
| `src/analysis/technical.py` | Technical indicators: RSI, MACD, SMA, Bollinger via `ta` library |
| `src/analysis/risk_manager.py` | RiskManager: pre/post trade enforcement, sector concentration |
| `src/analysis/performance.py` | Performance metrics: inception return, drawdown, Sharpe, SPY alpha |
| `src/analysis/risk_manager.py` | Also contains correlation monitoring functions |
| `src/analysis/outcome_tracker.py` | OutcomeTracker: P&L, alpha vs sector ETF/SPY, verdict |
| `src/analysis/track_record.py` | TrackRecord: win rate by sector/conviction/regime/session |
| `src/analysis/decision_quality.py` | DecisionQualityTracker: forward-return accuracy (1d/3d/5d) |
| `config/risk_rules.py` | RiskRules dataclass: all limits and thresholds |
| `config/strategies.py` | PortfolioAConfig, PortfolioBConfig |
| `config/universe.py` | SECTOR_MAP, SECTOR_ETF_MAP, ticker lists |

## Regime Classifier (`src/analysis/regime.py`)

```python
class RegimeClassifier:
    def classify(self, closes: pd.DataFrame, vix: float | None = None) -> RegimeResult
```

### 5 Regimes

| Regime | Conditions |
|--------|------------|
| `bullish` | SPY above 50-day SMA, breadth positive, VIX < 20 |
| `correction` | SPY -5% to -10% from peak, elevated VIX |
| `bearish` | SPY below 50-day SMA, breadth negative, VIX > 25 |
| `crisis` | SPY -10%+ from peak, VIX > 30 |
| `consolidation` | Default / sideways |

```python
@dataclass
class RegimeResult:
    regime: str        # one of the 5 above
    confidence: float  # 0-1
    details: dict      # spy_trend, vix_level, breadth
```

## Technical Indicators (`src/analysis/technical.py`)

```python
def compute_indicators(closes: pd.DataFrame, volumes: pd.DataFrame | None = None) -> dict[str, dict]
    # Returns: {ticker: {rsi_14, macd, macd_signal, sma_20, sma_50, bollinger_upper, bollinger_lower}}
```

Uses the `ta` library (not `pandas-ta`, which is unavailable for Python 3.11).

## Risk Manager (`src/analysis/risk_manager.py`)

```python
class RiskManager:
    def __init__(self, rules: RiskRules = RISK_RULES, db: Database | None = None) -> None

    async def check_pre_trade(
        self, trades: list[TradeSchema], positions: dict, portfolio_value: float,
        closes: pd.DataFrame | None = None, tenant_id: str = "default",
    ) -> TradeVerdict
        # Returns: RiskVerdict(allowed, blocked, requires_approval, requires_trade_approval, warnings)

    async def check_circuit_breakers(
        self, db: Database, portfolio_name: str, tenant_id: str = "default",
    ) -> bool
        # Returns True if daily loss exceeds threshold -> halt trading
```

### Pre-Trade Checks

1. **Single position limit:** Max % of portfolio in one ticker
2. **Sector concentration:** Max % in one sector (with overrides per sector)
3. **Inverse ETF rules:** Regime/posture gating, hold time warnings, Telegram approval
4. **Large trade approval:** Non-inverse BUYs > `trade_approval_threshold_pct` (default 10%) flagged for Telegram approval
5. **Correlation check:** Warn if adding correlated position
6. **Position count:** Max total positions

```python
@dataclass
class RiskVerdict:
    allowed: list[TradeSchema]
    blocked: list[TradeSchema]
    requires_approval: list[tuple[TradeSchema, str]]       # inverse ETF approval
    requires_trade_approval: list[tuple[TradeSchema, str]]  # large trade approval (>threshold%)
    warnings: list[str]
```

## Risk Rules (`config/risk_rules.py`)

```python
@dataclass(frozen=True)
class RiskRules:
    max_single_position_pct: float = 0.30       # 30% max single position
    max_sector_concentration_pct: float = 0.40   # 40% max in one sector
    max_positions: int = 10
    daily_loss_halt_pct: float = 0.03            # 3% daily loss -> halt
    sector_concentration_overrides: dict = ...    # per-sector limits
    defensive_tickers: frozenset = frozenset({"GLD", "TLT", "SH", "PSQ", "BIL", "SHY", "VTIP"})
```

## Performance Metrics (`src/analysis/performance.py`)

```python
class PerformanceTracker:
    async def compute_stats(self, db: Database, portfolio_name: str, tenant_id="default") -> PerformanceStats
        # inception_return_pct, max_drawdown_pct, sharpe_ratio, alpha_vs_spy

    def format_for_prompt(self, stats: PerformanceStats) -> str
```

SPY benchmarking uses yfinance to fetch SPY returns for the same period.

## Correlation Monitor (in `src/analysis/risk_manager.py`)

```python
def compute_correlation_matrix(closes: pd.DataFrame, tickers: list[str], window: int = 60) -> pd.DataFrame
def find_high_correlations(matrix: pd.DataFrame, threshold: float = 0.80) -> list[tuple[str, str, float]]
```

Computes pairwise correlation on 60-day rolling window. Warns on pairs > 0.80.

## Outcome Tracker (`src/analysis/outcome_tracker.py`)

```python
class OutcomeTracker:
    def __init__(self, db: Database) -> None
    async def get_recent_outcomes(self, days=30, tenant_id="default") -> list[TradeOutcome]
    async def get_open_position_outcomes(self, tenant_id="default") -> list[TradeOutcome]

@dataclass(frozen=True)
class TradeOutcome:
    ticker: str; side: str; entry_price: float; current_price: float
    exit_price: float | None; pnl_pct: float; hold_days: int; sector: str
    sector_etf_pct: float | None; spy_pct: float | None
    alpha_vs_sector: float | None; alpha_vs_spy: float | None
    conviction: str; reasoning: str
    regime_at_entry: str | None; session_at_entry: str | None
    verdict: str | None  # "OUTPERFORMED" | "UNDERPERFORMED" | "MATCHED"
```

Alpha calculation: `pnl_pct - benchmark_pct` for both sector ETF and SPY over the same holding period.
Uses `SECTOR_ETF_MAP` from `config/universe.py` to find benchmark ETF per sector.

## Track Record (`src/analysis/track_record.py`)

```python
WIN_THRESHOLD = 0.5    # > 0.5% P&L = win
LOSS_THRESHOLD = -0.5  # < -0.5% P&L = loss
                        # Between = scratch (neutral)

class TrackRecord:
    def compute(self, outcomes: list[TradeOutcome], min_trades: int = 5) -> TrackRecordStats
    @staticmethod
    def format_for_prompt(stats: TrackRecordStats) -> str

@dataclass(frozen=True)
class TrackRecordStats:
    total_trades: int; wins: int; losses: int; scratches: int
    win_rate_pct: float; avg_pnl_pct: float; avg_alpha_vs_spy: float | None
    by_sector: list[CategoryWinRate]
    by_conviction: list[CategoryWinRate]
    by_regime: list[CategoryWinRate]
    by_session: list[CategoryWinRate]
    best_sector: str | None; worst_sector: str | None

@dataclass(frozen=True)
class CategoryWinRate:
    category: str; value: str; total: int; wins: int; losses: int
    win_rate_pct: float; avg_pnl_pct: float; avg_alpha_vs_spy: float | None
```

`min_trades` filter: categories with fewer trades than threshold are excluded from breakdown.

## Decision Quality (`src/analysis/decision_quality.py`)

```python
class DecisionQualityTracker:
    def __init__(self, db: Database) -> None
    async def analyze_recent(self, days=30, tenant_id="default") -> list[DecisionQuality]
    @staticmethod
    def summarize(qualities: list[DecisionQuality]) -> DecisionQualitySummary
    @staticmethod
    def format_for_prompt(summary: DecisionQualitySummary) -> str

@dataclass(frozen=True)
class DecisionQuality:
    date: date; ticker: str; side: str  # "BUY"|"SELL"
    fwd_1d: float | None; fwd_3d: float | None; fwd_5d: float | None
    favorable_1d: bool; favorable_3d: bool; favorable_5d: bool

@dataclass(frozen=True)
class DecisionQualitySummary:
    total_decisions: int
    favorable_1d_pct: float; favorable_3d_pct: float; favorable_5d_pct: float
```

"Favorable" means the price moved in the trade direction (up for BUY, down for SELL).
Uses yfinance to fetch forward prices for 1/3/5 business days after decision.

## Momentum Strategy (`src/strategies/portfolio_a.py`)

```python
class MomentumStrategy:
    lookback = 63 days; skip = 5 days; top_n = 1
    def rank(self, closes) -> pd.DataFrame  # [ticker, return_63d, rank]
    def generate_trades(self, closes, current_positions, cash, portfolio_value) -> list[TradeSchema]
```

Universe: `PORTFOLIO_A_UNIVERSE` (20 ETFs: 10 sector + 10 thematic).
Calculates 63-day return, skips last 5 days (mean reversion filter), selects top 1 ETF.

## Data Flow: Outcome Feedback into Agent

1. `OutcomeTracker.get_recent_outcomes()` -> recent trade P&L + alpha
2. Formatted as "Decision Review" section in system prompt
3. `TrackRecord.compute()` -> win rate stats by sector/conviction/regime/session
4. Formatted as "Track Record" section in system prompt
5. `DecisionQualityTracker.analyze_recent()` -> forward-return accuracy
6. Available via API (`/api/portfolios/B/decision-quality`)
7. Agent sees its own track record and adjusts strategy accordingly

## Gotchas

- `pandas-ta` unavailable for Python 3.11 -- using `ta` library instead
- `TrackRecord.format_for_prompt` uses `:.0f` for win_rate -- 66.7% rounds to "67%"
- Outcome alpha calculation depends on `SECTOR_ETF_MAP` -- if sector not mapped, alpha is None
- SPY benchmark fetched via yfinance -- may fail if yfinance is down
- Risk manager accepts `tenant_id` -- sector concentration limits are global, not per-tenant
