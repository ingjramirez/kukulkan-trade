"""Tests for Fear & Greed Index fetcher."""

from unittest.mock import AsyncMock, MagicMock, patch

from src.data.fear_greed import (
    _classify,
    fetch_and_save,
    fetch_fear_greed,
    format_for_context,
)


def test_classify_extreme_fear():
    assert _classify(10) == "Extreme Fear"
    assert _classify(20) == "Extreme Fear"


def test_classify_fear():
    assert _classify(25) == "Fear"
    assert _classify(40) == "Fear"


def test_classify_neutral():
    assert _classify(50) == "Neutral"
    assert _classify(60) == "Neutral"


def test_classify_greed():
    assert _classify(70) == "Greed"
    assert _classify(80) == "Greed"


def test_classify_extreme_greed():
    assert _classify(90) == "Extreme Greed"
    assert _classify(100) == "Extreme Greed"


@patch("src.data.fear_greed.httpx.get")
def test_fetch_fear_greed_success(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "data": [{"value": "25", "value_classification": "Extreme Fear", "timestamp": "1708300800"}]
    }
    mock_get.return_value = mock_resp

    result = fetch_fear_greed()
    assert result is not None
    assert result["value"] == 25
    assert result["classification"] == "Extreme Fear"


@patch("src.data.fear_greed.httpx.get")
def test_fetch_fear_greed_empty_data(mock_get):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"data": []}
    mock_get.return_value = mock_resp

    result = fetch_fear_greed()
    assert result is None


@patch("src.data.fear_greed.httpx.get")
def test_fetch_fear_greed_network_error(mock_get):
    import httpx

    mock_get.side_effect = httpx.ConnectError("Connection refused")

    result = fetch_fear_greed()
    assert result is None


@patch("src.data.fear_greed.fetch_fear_greed")
async def test_fetch_and_save_success(mock_fetch):
    mock_fetch.return_value = {"value": 75, "classification": "Greed", "timestamp": "1708300800"}

    db = MagicMock()
    db.save_sentiment_indicator = AsyncMock()

    result = await fetch_and_save(db, "default")
    assert result is not None
    assert result["value"] == 75
    db.save_sentiment_indicator.assert_called_once()


@patch("src.data.fear_greed.fetch_fear_greed")
async def test_fetch_and_save_returns_none_on_failure(mock_fetch):
    mock_fetch.return_value = None

    db = MagicMock()
    db.save_sentiment_indicator = AsyncMock()

    result = await fetch_and_save(db, "default")
    assert result is None
    db.save_sentiment_indicator.assert_not_called()


def test_format_for_context():
    result = format_for_context(25.0, "Extreme Fear")
    assert "25/100" in result
    assert "Extreme Fear" in result
    assert "Fear & Greed" in result
