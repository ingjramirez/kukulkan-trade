"""Application settings loaded from environment variables."""

import os
from pathlib import Path

from pydantic_settings import BaseSettings


class TelegramSettings(BaseSettings):
    """Telegram bot settings."""

    bot_token: str = ""
    chat_id: str = ""

    model_config = {"env_prefix": "TELEGRAM_", "env_file": ".env", "extra": "ignore"}


class AlpacaSettings(BaseSettings):
    """Alpaca brokerage settings."""

    api_key: str = ""
    secret_key: str = ""
    paper: bool = True  # True = paper trading, False = live

    model_config = {"env_prefix": "ALPACA_", "env_file": ".env", "extra": "ignore"}


class FinnhubSettings(BaseSettings):
    """Finnhub API settings (free tier: 60 calls/min)."""

    api_key: str = ""

    model_config = {"env_prefix": "FINNHUB_", "env_file": ".env", "extra": "ignore"}


class DashboardSettings(BaseSettings):
    """Dashboard authentication settings."""

    user: str = "admin"
    password: str = ""

    model_config = {"env_prefix": "DASHBOARD_", "env_file": ".env", "extra": "ignore"}


class AgentSettings(BaseSettings):
    """AI agent strategy settings."""

    strategy_mode: str = "conservative"  # conservative | standard | aggressive
    agent_tool_model: str = "claude-sonnet-4-6"
    agent_max_turns: int = 8
    agent_session_budget: float = 0.50
    daily_budget: float = 3.0  # env: AGENT_DAILY_BUDGET
    monthly_budget: float = 75.0  # env: AGENT_MONTHLY_BUDGET
    scan_model: str = "claude-haiku-4-5-20251001"  # env: AGENT_SCAN_MODEL
    validate_model: str = "claude-opus-4-6"  # env: AGENT_VALIDATE_MODEL
    enable_tiered: bool = False  # env: AGENT_ENABLE_TIERED
    enable_cache: bool = True  # env: AGENT_ENABLE_CACHE
    max_retries: int = 2  # env: AGENT_MAX_RETRIES — pacer prevents most 429s, fail fast
    fallback_model: str = "claude-sonnet-4-6"  # env: AGENT_FALLBACK_MODEL
    agent_history_recent_n: int = 2  # env: AGENT_AGENT_HISTORY_RECENT_N — recent sessions to replay
    agent_history_summaries_n: int = 10  # env: AGENT_AGENT_HISTORY_SUMMARIES_N — compressed summaries
    agent_skip_history_triggers: str = "manual,event"  # CSV — these triggers get no recent history
    agent_routine_max_turns: int = 3  # env: AGENT_AGENT_ROUTINE_MAX_TURNS — mini investigation on ROUTINE
    agent_event_history_recent_n: int = 1  # env: AGENT_AGENT_EVENT_HISTORY_RECENT_N — recent sessions for manual/event
    agent_tool_result_max_chars: int = 1500  # env: AGENT_AGENT_TOOL_RESULT_MAX_CHARS — tool result truncation
    agent_tpm_limit: int = 25500  # env: AGENT_AGENT_TPM_LIMIT — 85% of Tier 1's 30K TPM

    model_config = {"env_prefix": "AGENT_", "env_file": ".env", "extra": "ignore"}


class ChromaSettings(BaseSettings):
    """ChromaDB connection settings."""

    host: str = "localhost"
    port: int = 8000

    model_config = {"env_prefix": "CHROMA_", "env_file": ".env", "extra": "ignore"}


class Settings(BaseSettings):
    """Root application settings."""

    # API keys
    anthropic_api_key: str = ""
    fred_api_key: str = ""

    # Executor: "alpaca" or "paper"
    executor: str = "paper"

    # Database
    database_url: str = "sqlite+aiosqlite:///data/kukulkan.db"

    # Logging
    log_level: str = "INFO"

    # JWT
    jwt_secret: str = "change-me-in-production"

    # Tenant credential encryption (Fernet key)
    tenant_encryption_key: str = ""

    # Sub-settings
    alpaca: AlpacaSettings = AlpacaSettings()
    telegram: TelegramSettings = TelegramSettings()
    finnhub: FinnhubSettings = FinnhubSettings()
    chroma: ChromaSettings = ChromaSettings()
    dashboard: DashboardSettings = DashboardSettings()
    agent: AgentSettings = AgentSettings()

    # Trade approval (disabled for paper trading — flip to True for live)
    trade_approval_enabled: bool = False  # Master switch for all Telegram trade approvals
    trade_approval_threshold_pct: float = 10.0  # Trades > this % of portfolio require Telegram approval
    trade_approval_timeout_s: int = 300  # Seconds to wait for Telegram response (5 min)

    # SSE (Server-Sent Events)
    sse_heartbeat_s: float = 30.0  # heartbeat interval in seconds
    sse_max_queue: int = 64  # max queued events per subscriber
    sse_history_size: int = 100  # recent events buffer for catch-up

    # Sentinel (intraday monitoring)
    sentinel_enabled: bool = True  # Enable sentinel checks
    sentinel_interval_min: int = 30  # Check interval in minutes
    sentinel_max_escalations_per_day: int = 2  # Max crisis sessions triggered per day
    sentinel_escalation_cooldown_s: int = 1800  # Seconds after a scheduled session before escalation is allowed

    # Pipeline tuning
    inter_tenant_delay: float = 2.0  # seconds between tenant runs (avoids Alpaca rate limits)

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root / "data"
    logs_dir: Path = project_root / "logs"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


def _is_dev_or_test() -> bool:
    """Check if running in dev/test mode (pytest or explicit env)."""
    import sys

    env = os.environ.get("ENV", os.environ.get("ENVIRONMENT", "")).lower()
    if env in ("dev", "development", "test", "testing"):
        return True
    # Detect pytest execution
    if "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ:
        return True
    return "pytest" in os.environ.get("_", "")


settings = Settings()

if not _is_dev_or_test():
    if settings.jwt_secret == "change-me-in-production":
        raise ValueError(
            "JWT_SECRET must be changed from the default value in production. "
            "Set a random 64+ character string in your .env file."
        )
    if not settings.tenant_encryption_key:
        raise ValueError(
            "TENANT_ENCRYPTION_KEY must be set in production. "
            "Generate one with: python -c "
            "'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
        )
