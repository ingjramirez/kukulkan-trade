"""Tests for SentimentIndicatorRow model and CRUD."""

import pytest

from src.storage.database import Database
from src.storage.models import SentimentIndicatorRow


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


async def test_save_and_get_sentiment_indicator(db: Database):
    await db.save_sentiment_indicator(
        tenant_id="default",
        name="fear_greed_index",
        value=25.0,
        classification="Extreme Fear",
        sub_indicators='{"momentum": 30, "volatility": 20}',
    )
    row = await db.get_latest_sentiment("default", "fear_greed_index")
    assert row is not None
    assert row.value == 25.0
    assert row.classification == "Extreme Fear"
    assert row.name == "fear_greed_index"


async def test_get_latest_returns_most_recent(db: Database):
    await db.save_sentiment_indicator("default", "fear_greed_index", 25.0, "Extreme Fear")
    await db.save_sentiment_indicator("default", "fear_greed_index", 75.0, "Greed")

    row = await db.get_latest_sentiment("default", "fear_greed_index")
    assert row is not None
    assert row.value == 75.0
    assert row.classification == "Greed"


async def test_get_latest_returns_none_when_empty(db: Database):
    row = await db.get_latest_sentiment("default", "fear_greed_index")
    assert row is None


async def test_sentiment_model_fields():
    row = SentimentIndicatorRow(
        tenant_id="default",
        name="fear_greed_index",
        value=50.0,
        classification="Neutral",
    )
    assert row.tenant_id == "default"
    assert row.sub_indicators is None
