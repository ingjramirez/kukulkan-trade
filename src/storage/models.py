"""Pydantic schemas and SQLAlchemy ORM models for Kukulkan."""

from datetime import date, datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase

# ── SQLAlchemy Base ──────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    pass


# ── Enums ────────────────────────────────────────────────────────────────────


class PortfolioName(str, Enum):
    A = "A"
    B = "B"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class Regime(str, Enum):
    BULL = "BULL"
    ROTATION = "ROTATION"
    NEUTRAL = "NEUTRAL"
    BEAR = "BEAR"


# ── SQLAlchemy ORM Models ────────────────────────────────────────────────────


class PortfolioRow(Base):
    """Current state of each portfolio."""

    __tablename__ = "portfolios"
    __table_args__ = (UniqueConstraint("tenant_id", "name"),)

    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, default="default")
    name = Column(String(1), nullable=False)  # A, B
    cash = Column(Float, nullable=False, default=33_000.0)
    total_value = Column(Float, nullable=False, default=33_000.0)
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class PositionRow(Base):
    """Open positions per portfolio."""

    __tablename__ = "positions"
    __table_args__ = (UniqueConstraint("tenant_id", "portfolio", "ticker"),)

    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, default="default")
    portfolio = Column(String(1), nullable=False)
    ticker = Column(String(10), nullable=False)
    shares = Column(Float, nullable=False)
    avg_price = Column(Float, nullable=False)
    current_price = Column(Float)
    market_value = Column(Float)
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class TradeRow(Base):
    """Executed trade log."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, default="default")
    portfolio = Column(String(1), nullable=False)
    ticker = Column(String(10), nullable=False)
    side = Column(String(4), nullable=False)  # BUY / SELL
    shares = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    total = Column(Float, nullable=False)
    reason = Column(Text)
    executed_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class DailySnapshotRow(Base):
    """End-of-day portfolio snapshot for performance tracking."""

    __tablename__ = "daily_snapshots"
    __table_args__ = (UniqueConstraint("tenant_id", "portfolio", "date"),)

    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, default="default")
    portfolio = Column(String(1), nullable=False)
    date = Column(Date, nullable=False)
    total_value = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    positions_value = Column(Float, nullable=False)
    daily_return_pct = Column(Float)
    cumulative_return_pct = Column(Float)


class MomentumRankingRow(Base):
    """Daily momentum rankings for Portfolio A."""

    __tablename__ = "momentum_rankings"
    __table_args__ = (UniqueConstraint("date", "ticker"),)

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    ticker = Column(String(10), nullable=False)
    return_63d = Column(Float, nullable=False)
    rank = Column(Integer, nullable=False)


class AgentDecisionRow(Base):
    """Claude's trade decisions for Portfolio B."""

    __tablename__ = "agent_decisions"

    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, default="default")
    date = Column(Date, nullable=False)
    prompt_summary = Column(Text)
    response_summary = Column(Text)
    proposed_trades = Column(Text)  # JSON string
    reasoning = Column(Text)
    model_used = Column(String(50))
    tokens_used = Column(Integer)
    regime = Column(String(30), nullable=True)
    session_label = Column(String(20), nullable=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class MarketDataRow(Base):
    """OHLCV market data cache."""

    __tablename__ = "market_data"
    __table_args__ = (UniqueConstraint("ticker", "date"),)

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)


class TechnicalIndicatorRow(Base):
    """Computed technical indicators."""

    __tablename__ = "technical_indicators"
    __table_args__ = (UniqueConstraint("ticker", "date"),)

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False)
    date = Column(Date, nullable=False)
    rsi_14 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_hist = Column(Float)
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    bb_upper = Column(Float)
    bb_middle = Column(Float)
    bb_lower = Column(Float)


class MacroDataRow(Base):
    """Macro economic indicators from FRED."""

    __tablename__ = "macro_data"
    __table_args__ = (UniqueConstraint("indicator", "date"),)

    id = Column(Integer, primary_key=True)
    indicator = Column(String(50), nullable=False)
    date = Column(Date, nullable=False)
    value = Column(Float, nullable=False)


class NewsLogRow(Base):
    """Log of news articles fetched and embedded."""

    __tablename__ = "news_log"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10))
    headline = Column(Text, nullable=False)
    source = Column(String(100))
    url = Column(Text)
    published_at = Column(DateTime)
    sentiment = Column(Float)  # -1.0 to 1.0
    embedding_id = Column(String(100))  # ChromaDB document ID
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class DiscoveredTickerRow(Base):
    """Dynamically discovered tickers for Portfolio B (tenant-scoped)."""

    __tablename__ = "discovered_tickers"
    __table_args__ = (UniqueConstraint("tenant_id", "ticker"),)

    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, default="default")
    ticker = Column(String(10), nullable=False)
    source = Column(String(20), nullable=False)  # "agent", "news", "screener"
    rationale = Column(Text)
    # proposed/approved/rejected/expired
    status = Column(String(10), nullable=False, default="proposed")
    proposed_at = Column(Date, nullable=False)
    expires_at = Column(Date, nullable=False)
    sector = Column(String(50))
    market_cap = Column(Float)


class AgentMemoryRow(Base):
    """Persistent memory for Portfolio B AI agent."""

    __tablename__ = "agent_memory"
    __table_args__ = (UniqueConstraint("tenant_id", "category", "key"),)

    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, default="default")
    category = Column(String(20), nullable=False)  # short_term, weekly_summary, agent_note
    key = Column(String(100), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime, nullable=True)


class WeeklyReportRow(Base):
    """Weekly performance reports."""

    __tablename__ = "weekly_reports"

    id = Column(Integer, primary_key=True)
    week_start = Column(Date, nullable=False)
    week_end = Column(Date, nullable=False)
    portfolio_a_return = Column(Float)
    portfolio_b_return = Column(Float)
    benchmark_return = Column(Float)  # SPY
    report_text = Column(Text)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class TrailingStopRow(Base):
    """Active trailing stop for a position."""

    __tablename__ = "trailing_stops"
    __table_args__ = (UniqueConstraint("tenant_id", "portfolio", "ticker"),)

    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, default="default")
    portfolio = Column(String(1), nullable=False)
    ticker = Column(String(10), nullable=False)
    entry_price = Column(Float, nullable=False)
    peak_price = Column(Float, nullable=False)
    trail_pct = Column(Float, nullable=False)  # 0.05 = 5%
    stop_price = Column(Float, nullable=False)  # peak * (1 - trail_pct)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class EarningsCalendarRow(Base):
    """Cached upcoming earnings dates from yfinance."""

    __tablename__ = "earnings_calendar"
    __table_args__ = (UniqueConstraint("ticker", "earnings_date"),)

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False)
    earnings_date = Column(Date, nullable=False)
    source = Column(String(20), default="yfinance")
    fetched_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class WatchlistRow(Base):
    """AI-managed watchlist for Portfolio B."""

    __tablename__ = "watchlist"
    __table_args__ = (UniqueConstraint("tenant_id", "ticker"),)

    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, default="default")
    portfolio = Column(String(1), nullable=False, default="B")
    ticker = Column(String(10), nullable=False)
    reason = Column(Text)
    conviction = Column(String(10), nullable=False, default="medium")
    target_entry = Column(Float, nullable=True)
    added_date = Column(Date, nullable=False)
    expires_at = Column(Date, nullable=False)


class IntradaySnapshotRow(Base):
    """Intraday portfolio snapshot (every 15 min during market hours)."""

    __tablename__ = "intraday_snapshots"
    __table_args__ = (UniqueConstraint("tenant_id", "portfolio", "timestamp"),)

    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, default="default")
    portfolio = Column(String(1), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    total_value = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    positions_value = Column(Float, nullable=False)
    is_extended_hours = Column(Boolean, nullable=False, default=False)
    market_phase = Column(String(20), nullable=False, default="market")


class SentinelActionRow(Base):
    """Queued sentinel actions for after-hours and quiet hours."""

    __tablename__ = "sentinel_actions"

    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    action_type = Column(String(20), nullable=False)  # sell | reduce | hedge | review
    ticker = Column(String(10), nullable=False)
    reason = Column(Text, nullable=False)
    source = Column(String(30), nullable=False)  # afterhours_sentinel | premarket_sentinel | quiet_hours | gap_risk
    alert_level = Column(String(10), nullable=False)  # warning | critical
    status = Column(String(20), nullable=False, default="pending")  # pending | executed | cancelled
    resolved_at = Column(DateTime, nullable=True)
    resolved_by = Column(String(20), nullable=True)  # agent | owner_telegram | auto_expired


class ToolCallLogRow(Base):
    """Log of tool calls during agentic Portfolio B sessions."""

    __tablename__ = "tool_call_logs"

    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, default="default")
    session_date = Column(Date, nullable=False)
    session_label = Column(String(20), nullable=True)
    turn = Column(Integer, nullable=False)
    tool_name = Column(String(50), nullable=False)
    tool_input = Column(Text, nullable=True)  # JSON
    tool_output_preview = Column(Text, nullable=True)  # First 500 chars
    success = Column(Boolean, nullable=False, default=True)
    error = Column(Text, nullable=True)
    influenced_decision = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class AgentConversationRow(Base):
    """Persistent agent conversation sessions."""

    __tablename__ = "agent_conversations"

    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    session_id = Column(Text, nullable=False, unique=True)
    trigger_type = Column(String(20), nullable=False)  # morning/midday/close/event/weekly_review
    messages_json = Column(Text, nullable=False)  # Full Anthropic messages array (JSON)
    summary = Column(Text, nullable=True)  # Haiku-compressed summary (NULL if recent)
    token_count = Column(Integer, nullable=False, default=0)
    cost_usd = Column(Float, nullable=False, default=0.0)
    session_status = Column(String(20), nullable=False, default="completed")  # started/completed/crashed
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class PostureHistoryRow(Base):
    """Posture declarations from the AI agent per session."""

    __tablename__ = "posture_history"

    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, default="default")
    session_date = Column(Date, nullable=False)
    session_label = Column(String(20), nullable=True)
    posture = Column(String(20), nullable=False)  # balanced/defensive/crisis/aggressive
    effective_posture = Column(String(20), nullable=False)  # after gate checks
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class PlaybookSnapshotRow(Base):
    """Empirical playbook snapshots — regime×sector win rates."""

    __tablename__ = "playbook_snapshots"

    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, default="default")
    generated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    regime = Column(String(30), nullable=False)
    sector = Column(String(50), nullable=False)
    total_trades = Column(Integer, nullable=False)
    wins = Column(Integer, nullable=False)
    losses = Column(Integer, nullable=False)
    win_rate_pct = Column(Float, nullable=False)
    avg_pnl_pct = Column(Float, nullable=False)
    recommendation = Column(String(30), nullable=False)


class ConvictionCalibrationRow(Base):
    """Conviction calibration snapshots — per-level accuracy stats."""

    __tablename__ = "conviction_calibration"

    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, default="default")
    generated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    conviction_level = Column(String(10), nullable=False)
    total_trades = Column(Integer, nullable=False)
    wins = Column(Integer, nullable=False)
    losses = Column(Integer, nullable=False)
    win_rate_pct = Column(Float, nullable=False)
    avg_pnl_pct = Column(Float, nullable=False)
    assessment = Column(String(30), nullable=False)
    suggested_multiplier = Column(Float, nullable=False, default=1.0)


class AgentBudgetLogRow(Base):
    """Per-session agent cost log for daily/monthly budget tracking."""

    __tablename__ = "agent_budget_log"

    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, default="default")
    session_date = Column(Date, nullable=False)
    session_label = Column(String(50), nullable=False)
    session_id = Column(Text, nullable=True)
    input_tokens = Column(Integer, nullable=False, default=0)
    output_tokens = Column(Integer, nullable=False, default=0)
    cache_read_tokens = Column(Integer, nullable=False, default=0)
    cache_creation_tokens = Column(Integer, nullable=False, default=0)
    cost_usd = Column(Float, nullable=False, default=0.0)
    session_profile = Column(String(20), nullable=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class ImprovementSnapshotRow(Base):
    """Weekly self-improvement snapshot — performance data + proposal + applied changes."""

    __tablename__ = "improvement_snapshots"

    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, default="default")
    week_start = Column(Date, nullable=False)
    week_end = Column(Date, nullable=False)
    total_trades = Column(Integer, nullable=False, default=0)
    win_rate_pct = Column(Float, nullable=True)
    avg_pnl_pct = Column(Float, nullable=True)
    avg_alpha_vs_spy = Column(Float, nullable=True)
    total_cost_usd = Column(Float, nullable=True, default=0.0)
    strategy_mode = Column(String(20), nullable=True)
    trailing_stop_multiplier = Column(Float, nullable=True, default=1.0)
    proposal_json = Column(Text, nullable=True)  # Full Sonnet proposal JSON
    applied_changes = Column(Text, nullable=True)  # JSON list of applied changes
    report_text = Column(Text, nullable=True)  # Plain-text summary
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class ParameterChangelogRow(Base):
    """Audit log of auto-applied parameter changes."""

    __tablename__ = "parameter_changelog"

    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, default="default")
    snapshot_id = Column(Integer, ForeignKey("improvement_snapshots.id", ondelete="SET NULL"), nullable=True)
    parameter = Column(String(50), nullable=False)
    old_value = Column(Text, nullable=True)
    new_value = Column(Text, nullable=True)
    reason = Column(Text, nullable=True)
    applied_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class TenantRow(Base):
    """Multi-tenant configuration: credentials, strategy, and universe."""

    __tablename__ = "tenants"

    id = Column(String(36), primary_key=True)  # UUID
    name = Column(String(100), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)

    # Alpaca credentials (Fernet-encrypted, nullable = not yet configured)
    alpaca_api_key_enc = Column(Text, nullable=True)
    alpaca_api_secret_enc = Column(Text, nullable=True)
    alpaca_base_url = Column(String(200), nullable=False, default="https://paper-api.alpaca.markets")

    # Telegram credentials (Fernet-encrypted, nullable = not yet configured)
    telegram_bot_token_enc = Column(Text, nullable=True)
    telegram_chat_id_enc = Column(Text, nullable=True)

    # Dashboard credentials (for per-tenant login)
    dashboard_user = Column(String(100), nullable=True)
    dashboard_password_enc = Column(Text, nullable=True)  # Fernet-encrypted

    # Claude API key (encrypted, nullable = use system default)
    claude_api_key_enc = Column(Text, nullable=True)

    # Strategy
    strategy_mode = Column(String(20), nullable=False, default="conservative")

    # Portfolio config
    run_portfolio_a = Column(Boolean, nullable=False, default=False)
    run_portfolio_b = Column(Boolean, nullable=False, default=True)
    portfolio_a_cash = Column(Float, nullable=False, default=33_000.0)
    portfolio_b_cash = Column(Float, nullable=False, default=66_000.0)

    # Equity-based allocation (captured from Alpaca on first run)
    initial_equity = Column(Float, nullable=True)
    portfolio_a_pct = Column(Float, nullable=False, default=33.33)
    portfolio_b_pct = Column(Float, nullable=False, default=66.67)

    # Rebalance flag (set by API on toggle change, cleared by orchestrator)
    pending_rebalance = Column(Boolean, nullable=False, default=False)

    # Agent loop (agentic mode for Portfolio B)
    use_agent_loop = Column(Boolean, nullable=False, default=False)

    # Persistent agent (conversation persistence for Portfolio B)
    use_persistent_agent = Column(Boolean, nullable=False, default=False)

    # Tiered model runner (Haiku scan → Sonnet investigate → Opus validate)
    use_tiered_models = Column(Boolean, nullable=False, default=False)

    # Claude Code CLI (Max subscription — replaces AgentRunner + PersistentAgent)
    use_claude_code = Column(Boolean, nullable=False, default=False)

    # Trailing stop multiplier (0.5-2.0, scales TRAIL_PCT matrix)
    trailing_stop_multiplier = Column(Float, nullable=False, default=1.0)

    # Quiet hours (no Telegram during sleep window)
    quiet_hours_start = Column(String(5), nullable=False, default="21:00")
    quiet_hours_end = Column(String(5), nullable=False, default="07:00")
    quiet_hours_timezone = Column(String(40), nullable=False, default="America/Mexico_City")

    # Ticker customization (JSON arrays, nullable = use defaults)
    ticker_whitelist = Column(Text, nullable=True)  # JSON: ["AAPL","TSLA"]
    ticker_additions = Column(Text, nullable=True)  # JSON: ["COIN","MSTR"]
    ticker_exclusions = Column(Text, nullable=True)  # JSON: ["META","GOOGL"]

    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        """Mask sensitive fields in repr to prevent credential leaks."""
        return f"<Tenant id={self.id!r} name={self.name!r} active={self.is_active} strategy={self.strategy_mode!r}>"


class SentimentIndicatorRow(Base):
    """External sentiment indicators (Fear & Greed, put/call ratio, etc.)."""

    __tablename__ = "sentiment_indicators"
    __table_args__ = (Index("idx_sentiment_tenant_name", "tenant_id", "name", "fetched_at"),)

    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, default="default")
    name = Column(String(50), nullable=False)  # "fear_greed_index", "put_call_ratio", etc.
    value = Column(Float, nullable=False)  # Numeric value (0-100 for F&G)
    classification = Column(String(30), nullable=False)  # "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
    sub_indicators = Column(Text, nullable=True)  # JSON: component breakdown
    fetched_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class TickerSignalRow(Base):
    """Ticker signal rankings computed by the local SignalEngine (every 10 min)."""

    __tablename__ = "ticker_signals"
    __table_args__ = (
        Index("idx_signals_tenant_scored", "tenant_id", "scored_at"),
        Index("idx_signals_tenant_ticker", "tenant_id", "ticker", "scored_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    ticker = Column(String(10), nullable=False)
    composite_score = Column(Float, nullable=False)
    rank = Column(Integer, nullable=False)
    prev_rank = Column(Integer)
    rank_velocity = Column(Float, nullable=False, default=0)
    momentum_20d = Column(Float)
    momentum_63d = Column(Float)
    rsi = Column(Float)
    macd_histogram = Column(Float)
    sma_trend_score = Column(Float)
    bollinger_pct_b = Column(Float)
    volume_ratio = Column(Float)
    alerts = Column(String, default="[]")
    scored_at = Column(DateTime, nullable=False)


# ── Pydantic Schemas (for API / validation layer) ───────────────────────────


class TradeSchema(BaseModel):
    """Validated trade request."""

    portfolio: PortfolioName
    ticker: str = Field(max_length=10)
    side: OrderSide
    shares: float = Field(gt=0)
    price: float = Field(gt=0)
    reason: str = ""

    @property
    def total(self) -> float:
        return self.shares * self.price


class PositionSchema(BaseModel):
    """Current position snapshot."""

    portfolio: PortfolioName
    ticker: str
    shares: float
    avg_price: float
    current_price: float | None = None
    market_value: float | None = None


class PortfolioSnapshot(BaseModel):
    """Full portfolio state at a point in time."""

    name: PortfolioName
    cash: float
    positions: list[PositionSchema]
    total_value: float
    date: date


class MomentumRanking(BaseModel):
    """Single ticker momentum result."""

    ticker: str
    return_63d: float
    rank: int


class DailyBrief(BaseModel):
    """Daily summary sent via Telegram."""

    date: date
    regime: Regime | None = None
    portfolio_a: PortfolioSnapshot
    portfolio_b: PortfolioSnapshot
    proposed_trades: list[TradeSchema] = []
    commentary: str = ""


# ── Tenant Pydantic Schemas ──────────────────────────────────────────────────


class TenantCreate(BaseModel):
    """Schema for creating a new tenant."""

    name: str = Field(max_length=100)
    alpaca_api_key: str = Field(min_length=1)
    alpaca_api_secret: str = Field(min_length=1)
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    telegram_bot_token: str = Field(min_length=1)
    telegram_chat_id: str = Field(min_length=1)
    strategy_mode: str = Field(
        default="conservative",
        pattern=r"^(conservative|standard|aggressive)$",
    )
    run_portfolio_a: bool = False
    run_portfolio_b: bool = True
    portfolio_a_cash: float = Field(default=33_000.0, gt=0)
    portfolio_b_cash: float = Field(default=66_000.0, gt=0)
    portfolio_a_pct: float = Field(default=33.33, gt=0, le=100)
    portfolio_b_pct: float = Field(default=66.67, gt=0, le=100)
    ticker_whitelist: list[str] | None = None
    ticker_additions: list[str] | None = None
    ticker_exclusions: list[str] | None = None
    dashboard_user: str | None = None
    dashboard_password: str | None = None


class TenantUpdate(BaseModel):
    """Schema for updating a tenant (all fields optional)."""

    name: str | None = Field(default=None, max_length=100)
    alpaca_api_key: str | None = None
    alpaca_api_secret: str | None = None
    alpaca_base_url: str | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    strategy_mode: str | None = Field(default=None, pattern=r"^(conservative|standard|aggressive)$")
    run_portfolio_a: bool | None = None
    run_portfolio_b: bool | None = None
    portfolio_a_cash: float | None = Field(default=None, gt=0)
    portfolio_b_cash: float | None = Field(default=None, gt=0)
    portfolio_a_pct: float | None = Field(default=None, gt=0, le=100)
    portfolio_b_pct: float | None = Field(default=None, gt=0, le=100)
    ticker_whitelist: list[str] | None = None
    ticker_additions: list[str] | None = None
    ticker_exclusions: list[str] | None = None
    is_active: bool | None = None
    dashboard_user: str | None = None
    dashboard_password: str | None = None
    use_agent_loop: bool | None = None
    use_claude_code: bool | None = None
    quiet_hours_start: str | None = None
    quiet_hours_end: str | None = None
    quiet_hours_timezone: str | None = None


class TenantRead(BaseModel):
    """Schema for tenant API responses — credentials are always masked."""

    id: str
    name: str
    is_active: bool
    alpaca_api_key_masked: str
    alpaca_base_url: str
    telegram_chat_id_masked: str
    strategy_mode: str
    run_portfolio_a: bool
    run_portfolio_b: bool
    portfolio_a_cash: float
    portfolio_b_cash: float
    initial_equity: float | None = None
    portfolio_a_pct: float
    portfolio_b_pct: float
    pending_rebalance: bool = False
    ticker_whitelist: list[str] | None = None
    ticker_additions: list[str] | None = None
    ticker_exclusions: list[str] | None = None
    use_agent_loop: bool = False
    use_claude_code: bool = False
    quiet_hours_start: str = "21:00"
    quiet_hours_end: str = "07:00"
    quiet_hours_timezone: str = "America/Mexico_City"
    dashboard_user: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
