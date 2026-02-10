"""Pydantic response models for the REST API."""

from datetime import date, datetime

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(max_length=100)
    password: str = Field(max_length=200)


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


# ── Tenant Schemas ────────────────────────────────────────────────────


class TenantCreateRequest(BaseModel):
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
    ticker_whitelist: list[str] | None = None
    ticker_additions: list[str] | None = None
    ticker_exclusions: list[str] | None = None


class TenantUpdateRequest(BaseModel):
    name: str | None = Field(default=None, max_length=100)
    alpaca_api_key: str | None = None
    alpaca_api_secret: str | None = None
    alpaca_base_url: str | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    strategy_mode: str | None = Field(
        default=None, pattern=r"^(conservative|standard|aggressive)$",
    )
    run_portfolio_a: bool | None = None
    run_portfolio_b: bool | None = None
    portfolio_a_cash: float | None = Field(default=None, gt=0)
    portfolio_b_cash: float | None = Field(default=None, gt=0)
    ticker_whitelist: list[str] | None = None
    ticker_additions: list[str] | None = None
    ticker_exclusions: list[str] | None = None
    is_active: bool | None = None


class TenantReadResponse(BaseModel):
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
    ticker_whitelist: list[str] | None = None
    ticker_additions: list[str] | None = None
    ticker_exclusions: list[str] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
