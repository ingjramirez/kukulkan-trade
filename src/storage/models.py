"""Pydantic schemas and SQLAlchemy ORM models for Atlas."""

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
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

    id = Column(Integer, primary_key=True)
    name = Column(String(1), unique=True, nullable=False)  # A, B
    cash = Column(Float, nullable=False, default=33_000.0)
    total_value = Column(Float, nullable=False, default=33_000.0)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class PositionRow(Base):
    """Open positions per portfolio."""

    __tablename__ = "positions"
    __table_args__ = (UniqueConstraint("portfolio", "ticker"),)

    id = Column(Integer, primary_key=True)
    portfolio = Column(String(1), nullable=False)
    ticker = Column(String(10), nullable=False)
    shares = Column(Float, nullable=False)
    avg_price = Column(Float, nullable=False)
    current_price = Column(Float)
    market_value = Column(Float)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class TradeRow(Base):
    """Executed trade log."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    portfolio = Column(String(1), nullable=False)
    ticker = Column(String(10), nullable=False)
    side = Column(String(4), nullable=False)  # BUY / SELL
    shares = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    total = Column(Float, nullable=False)
    reason = Column(Text)
    executed_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class DailySnapshotRow(Base):
    """End-of-day portfolio snapshot for performance tracking."""

    __tablename__ = "daily_snapshots"
    __table_args__ = (UniqueConstraint("portfolio", "date"),)

    id = Column(Integer, primary_key=True)
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
    date = Column(Date, nullable=False)
    prompt_summary = Column(Text)
    response_summary = Column(Text)
    proposed_trades = Column(Text)  # JSON string
    reasoning = Column(Text)
    model_used = Column(String(50))
    tokens_used = Column(Integer)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


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
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class DiscoveredTickerRow(Base):
    """Dynamically discovered tickers for Portfolio B."""

    __tablename__ = "discovered_tickers"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False, unique=True)
    source = Column(String(20), nullable=False)  # "agent", "news", "screener"
    rationale = Column(Text)
    # proposed/approved/rejected/expired
    status = Column(String(10), nullable=False, default="proposed")
    proposed_at = Column(Date, nullable=False)
    expires_at = Column(Date, nullable=False)
    sector = Column(String(50))
    market_cap = Column(Float)


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
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


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
