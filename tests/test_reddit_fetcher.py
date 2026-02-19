"""Tests for Reddit sentiment scanner."""

from unittest.mock import MagicMock, patch

from src.data.base_fetcher import BaseNewsFetcher
from src.data.reddit_news import (
    MIN_SCORE,
    RedditNewsFetcher,
    _infer_sentiment_from_flair,
)


def test_inherits_base_fetcher():
    fetcher = RedditNewsFetcher(client_id="id", client_secret="secret")
    assert isinstance(fetcher, BaseNewsFetcher)
    assert fetcher.source_name == "reddit"
    assert fetcher.region == "us"


def test_skips_without_credentials():
    fetcher = RedditNewsFetcher()
    articles = fetcher.fetch(["AAPL"])
    assert articles == []


def test_infer_sentiment_bullish():
    assert _infer_sentiment_from_flair("YOLO") == 0.5
    assert _infer_sentiment_from_flair("DD") == 0.5
    assert _infer_sentiment_from_flair("Gain") == 0.5


def test_infer_sentiment_bearish():
    assert _infer_sentiment_from_flair("Loss") == -0.5
    assert _infer_sentiment_from_flair("Puts") == -0.5


def test_infer_sentiment_neutral():
    assert _infer_sentiment_from_flair("Discussion") is None
    assert _infer_sentiment_from_flair(None) is None
    assert _infer_sentiment_from_flair("") is None


def _make_reddit_response(title: str, score: int, flair: str | None = None, selftext: str = "") -> dict:
    return {
        "data": {
            "children": [
                {
                    "data": {
                        "title": title,
                        "score": score,
                        "link_flair_text": flair,
                        "selftext": selftext,
                        "permalink": "/r/wallstreetbets/comments/abc123/test",
                        "created_utc": 1708300800,
                    }
                }
            ]
        }
    }


@patch("src.data.reddit_news.httpx.get")
@patch("src.data.reddit_news.httpx.post")
def test_fetch_returns_articles(mock_post, mock_get):
    # Mock auth
    auth_resp = MagicMock()
    auth_resp.raise_for_status = MagicMock()
    auth_resp.json.return_value = {"access_token": "test_token"}
    mock_post.return_value = auth_resp

    # Mock Reddit API
    api_resp = MagicMock()
    api_resp.status_code = 200
    api_resp.raise_for_status = MagicMock()
    api_resp.json.return_value = _make_reddit_response(
        "NVDA to the moon! $AAPL looking strong too", score=500, flair="YOLO"
    )
    mock_get.return_value = api_resp

    fetcher = RedditNewsFetcher(client_id="id", client_secret="secret")
    articles = fetcher.fetch()

    assert len(articles) > 0
    assert articles[0].source == "reddit"
    assert articles[0].sentiment == 0.5  # YOLO flair → bullish


@patch("src.data.reddit_news.httpx.get")
@patch("src.data.reddit_news.httpx.post")
def test_filters_low_score_posts(mock_post, mock_get):
    auth_resp = MagicMock()
    auth_resp.raise_for_status = MagicMock()
    auth_resp.json.return_value = {"access_token": "test_token"}
    mock_post.return_value = auth_resp

    api_resp = MagicMock()
    api_resp.status_code = 200
    api_resp.raise_for_status = MagicMock()
    api_resp.json.return_value = _make_reddit_response("Low engagement post", score=MIN_SCORE - 1)
    mock_get.return_value = api_resp

    fetcher = RedditNewsFetcher(client_id="id", client_secret="secret")
    articles = fetcher.fetch()
    assert len(articles) == 0


@patch("src.data.reddit_news.httpx.get")
@patch("src.data.reddit_news.httpx.post")
def test_extracts_tickers_from_text(mock_post, mock_get):
    auth_resp = MagicMock()
    auth_resp.raise_for_status = MagicMock()
    auth_resp.json.return_value = {"access_token": "test_token"}
    mock_post.return_value = auth_resp

    api_resp = MagicMock()
    api_resp.status_code = 200
    api_resp.raise_for_status = MagicMock()
    api_resp.json.return_value = _make_reddit_response(
        "$AAPL is undervalued compared to $MSFT",
        score=200,
        selftext="Looking at NVDA too",
    )
    mock_get.return_value = api_resp

    fetcher = RedditNewsFetcher(client_id="id", client_secret="secret")
    articles = fetcher.fetch()
    assert len(articles) > 0
    tickers = articles[0].tickers
    assert "AAPL" in tickers
    assert "MSFT" in tickers


@patch("src.data.reddit_news.httpx.post")
def test_auth_failure_returns_empty(mock_post):
    import httpx

    mock_post.side_effect = httpx.ConnectError("Connection refused")

    fetcher = RedditNewsFetcher(client_id="id", client_secret="secret")
    articles = fetcher.fetch()
    assert articles == []


@patch("src.data.reddit_news.httpx.get")
@patch("src.data.reddit_news.httpx.post")
def test_metadata_includes_score_and_subreddit(mock_post, mock_get):
    auth_resp = MagicMock()
    auth_resp.raise_for_status = MagicMock()
    auth_resp.json.return_value = {"access_token": "test_token"}
    mock_post.return_value = auth_resp

    api_resp = MagicMock()
    api_resp.status_code = 200
    api_resp.raise_for_status = MagicMock()
    api_resp.json.return_value = _make_reddit_response("Big gains today", score=1000, flair="Gain")
    mock_get.return_value = api_resp

    fetcher = RedditNewsFetcher(client_id="id", client_secret="secret")
    articles = fetcher.fetch()
    assert len(articles) > 0
    assert articles[0].metadata["score"] == 1000
    assert articles[0].metadata["subreddit"] == "wallstreetbets"
