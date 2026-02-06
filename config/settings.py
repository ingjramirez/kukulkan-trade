"""Application settings loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings


class IBKRSettings(BaseSettings):
    """Interactive Brokers connection settings."""

    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1

    model_config = {"env_prefix": "IBKR_"}


class TelegramSettings(BaseSettings):
    """Telegram bot settings."""

    bot_token: str = ""
    chat_id: str = ""

    model_config = {"env_prefix": "TELEGRAM_"}


class ChromaSettings(BaseSettings):
    """ChromaDB connection settings."""

    host: str = "localhost"
    port: int = 8000

    model_config = {"env_prefix": "CHROMA_"}


class Settings(BaseSettings):
    """Root application settings."""

    # API keys
    anthropic_api_key: str = ""
    fred_api_key: str = ""

    # Database
    database_url: str = "sqlite+aiosqlite:///data/atlas.db"

    # Logging
    log_level: str = "INFO"

    # Sub-settings
    ibkr: IBKRSettings = IBKRSettings()
    telegram: TelegramSettings = TelegramSettings()
    chroma: ChromaSettings = ChromaSettings()

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root / "data"
    logs_dir: Path = project_root / "logs"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
