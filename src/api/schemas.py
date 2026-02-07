"""Pydantic response models for the REST API."""

from datetime import date, datetime

from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class AccountResponse(BaseModel):
    equity: float
    last_equity: float
    daily_pl: float
    daily_pl_pct: float
    cash: float
    buying_power: float
    positions: list["PositionResponse"]


class PortfolioSummary(BaseModel):
    name: str
    cash: float
    total_value: float
    updated_at: datetime | None = None


class PortfolioDetail(BaseModel):
    name: str
    cash: float
    total_value: float
    updated_at: datetime | None = None
    positions: list["PositionResponse"]


class PositionResponse(BaseModel):
    portfolio: str | None = None
    ticker: str
    shares: float
    avg_price: float
    current_price: float | None = None
    market_value: float | None = None
    unrealized_pl: float | None = None
    unrealized_plpc: float | None = None


class SnapshotResponse(BaseModel):
    portfolio: str
    date: date
    total_value: float
    cash: float
    positions_value: float
    daily_return_pct: float | None = None
    cumulative_return_pct: float | None = None


class TradeResponse(BaseModel):
    id: int
    portfolio: str
    ticker: str
    side: str
    shares: float
    price: float
    total: float
    reason: str | None = None
    executed_at: datetime


class MomentumRankingResponse(BaseModel):
    date: date
    ticker: str
    return_63d: float
    rank: int


class AgentDecisionResponse(BaseModel):
    id: int
    date: date
    prompt_summary: str | None = None
    response_summary: str | None = None
    proposed_trades: list | None = None
    reasoning: str | None = None
    model_used: str | None = None
    tokens_used: int | None = None
    created_at: datetime
