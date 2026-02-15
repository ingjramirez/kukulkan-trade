"""Pydantic response models for the REST API."""

from datetime import date, datetime, timezone
from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field


def _ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


UTCDatetime = Annotated[datetime, AfterValidator(_ensure_utc)]


class LoginRequest(BaseModel):
    username: str = Field(max_length=100)
    password: str = Field(max_length=200)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    tenant_id: str | None = None


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
    updated_at: UTCDatetime | None = None


class PortfolioDetail(BaseModel):
    name: str
    cash: float
    total_value: float
    updated_at: UTCDatetime | None = None
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
    executed_at: UTCDatetime


class MomentumRankingResponse(BaseModel):
    date: date
    ticker: str
    return_63d: float
    rank: int


class IntradaySnapshotResponse(BaseModel):
    portfolio: str
    timestamp: UTCDatetime
    total_value: float
    cash: float
    positions_value: float


class PortfolioHistoryResponse(BaseModel):
    timestamps: list[int]
    equity: list[float | None]
    profit_loss: list[float | None]
    profit_loss_pct: list[float | None]
    base_value: float
    timeframe: str


class AgentDecisionResponse(BaseModel):
    id: int
    date: date
    prompt_summary: str | None = None
    response_summary: str | None = None
    proposed_trades: list | None = None
    reasoning: str | None = None
    model_used: str | None = None
    tokens_used: int | None = None
    regime: str | None = None
    session_label: str | None = None
    created_at: UTCDatetime


# ── Tenant Schemas ────────────────────────────────────────────────────


class TenantCreateRequest(BaseModel):
    name: str = Field(max_length=100)
    alpaca_api_key: str | None = None
    alpaca_api_secret: str | None = None
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
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
    username: str | None = None
    password: str | None = None


class TenantUpdateRequest(BaseModel):
    name: str | None = Field(default=None, max_length=100)
    alpaca_api_key: str | None = None
    alpaca_api_secret: str | None = None
    alpaca_base_url: str | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    strategy_mode: str | None = Field(
        default=None,
        pattern=r"^(conservative|standard|aggressive)$",
    )
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
    username: str | None = None
    password: str | None = None
    use_agent_loop: bool | None = None


class TenantSelfUpdateRequest(BaseModel):
    """Fields a tenant user can update on their own account."""

    alpaca_api_key: str | None = None
    alpaca_api_secret: str | None = None
    alpaca_base_url: str | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    strategy_mode: str | None = Field(
        default=None,
        pattern=r"^(conservative|standard|aggressive)$",
    )
    run_portfolio_a: bool | None = None
    run_portfolio_b: bool | None = None
    ticker_whitelist: list[str] | None = None
    ticker_additions: list[str] | None = None
    ticker_exclusions: list[str] | None = None
    use_agent_loop: bool | None = None


# ── Outcome & Track Record Schemas ─────────────────────────────────


class TradeOutcomeResponse(BaseModel):
    ticker: str
    side: str
    entry_price: float
    current_price: float
    exit_price: float | None = None
    pnl_pct: float
    hold_days: int
    sector: str
    sector_etf_pct: float | None = None
    spy_pct: float | None = None
    alpha_vs_sector: float | None = None
    alpha_vs_spy: float | None = None
    conviction: str
    reasoning: str
    regime_at_entry: str | None = None
    session_at_entry: str | None = None
    verdict: str | None = None


class CategoryWinRateResponse(BaseModel):
    category: str
    value: str
    total: int
    wins: int
    losses: int
    win_rate_pct: float
    avg_pnl_pct: float
    avg_alpha_vs_spy: float | None = None


class TrackRecordResponse(BaseModel):
    total_trades: int
    wins: int
    losses: int
    scratches: int
    win_rate_pct: float
    avg_pnl_pct: float
    avg_alpha_vs_spy: float | None = None
    by_sector: list[CategoryWinRateResponse] = []
    by_conviction: list[CategoryWinRateResponse] = []
    by_regime: list[CategoryWinRateResponse] = []
    by_session: list[CategoryWinRateResponse] = []
    best_sector: str | None = None
    worst_sector: str | None = None


class DecisionQualityResponse(BaseModel):
    total_decisions: int
    favorable_1d_pct: float
    favorable_3d_pct: float
    favorable_5d_pct: float


# ── Tool Call Log Schemas ──────────────────────────────────────────


class ToolCallLogResponse(BaseModel):
    id: int
    session_date: date
    session_label: str | None = None
    turn: int
    tool_name: str
    tool_input: str | None = None
    tool_output_preview: str | None = None
    success: bool
    error: str | None = None
    influenced_decision: bool = False
    created_at: UTCDatetime


class ConversationSessionResponse(BaseModel):
    session_id: str
    trigger_type: str
    summary: str | None = None
    token_count: int = 0
    cost_usd: float = 0.0
    session_status: str = "completed"
    created_at: UTCDatetime


class TenantReadResponse(BaseModel):
    id: str
    name: str
    is_active: bool
    alpaca_api_key_masked: str | None = None
    alpaca_base_url: str
    telegram_chat_id_masked: str | None = None
    strategy_mode: str
    run_portfolio_a: bool
    run_portfolio_b: bool
    portfolio_a_cash: float
    portfolio_b_cash: float
    initial_equity: float | None = None
    portfolio_a_pct: float
    portfolio_b_pct: float
    pending_rebalance: bool = False
    use_agent_loop: bool = False
    ticker_whitelist: list[str] | None = None
    ticker_additions: list[str] | None = None
    ticker_exclusions: list[str] | None = None
    dashboard_user: str | None = None
    created_at: UTCDatetime | None = None
    updated_at: UTCDatetime | None = None
