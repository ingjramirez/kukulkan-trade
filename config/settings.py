"""Application settings loaded from environment variables."""

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

    # Sub-settings
    alpaca: AlpacaSettings = AlpacaSettings()
    telegram: TelegramSettings = TelegramSettings()
    finnhub: FinnhubSettings = FinnhubSettings()
    chroma: ChromaSettings = ChromaSettings()
    dashboard: DashboardSettings = DashboardSettings()
    agent: AgentSettings = AgentSettings()

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root / "data"
    logs_dir: Path = project_root / "logs"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
