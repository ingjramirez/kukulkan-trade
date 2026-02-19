"""Reddit sentiment scanner for wallstreetbets, stocks, investing.

Uses OAuth2 app-only (client_credentials) flow. Requires REDDIT_CLIENT_ID
and REDDIT_CLIENT_SECRET in .env. Skips gracefully if not configured.
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import structlog

from config.universe import FULL_UNIVERSE
from src.data.base_fetcher import BaseNewsFetcher
from src.data.news_article import NewsArticle

log = structlog.get_logger()

SUBREDDITS = ["wallstreetbets", "stocks", "investing"]
MIN_SCORE = 100  # Minimum upvotes to consider
MAX_POSTS_PER_SUB = 10
TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_API = "https://oauth.reddit.com"
USER_AGENT = "kukulkan-trade/1.0 (by /u/kukulkan-bot)"

# Flair-based sentiment hints
_BULLISH_FLAIRS = frozenset({"gain", "gains", "yolo", "bullish", "dd", "due diligence"})
_BEARISH_FLAIRS = frozenset({"loss", "losses", "bearish", "puts", "short"})


def _infer_sentiment_from_flair(flair: str | None) -> float | None:
    """Infer sentiment from Reddit post flair.

    Args:
        flair: Post flair text (lowercased for matching).

    Returns:
        Sentiment float or None if no signal.
    """
    if not flair:
        return None
    lower = flair.lower()
    if lower in _BULLISH_FLAIRS:
        return 0.5
    if lower in _BEARISH_FLAIRS:
        return -0.5
    return None


class RedditNewsFetcher(BaseNewsFetcher):
    """Fetches high-engagement posts from finance subreddits."""

    source_name = "reddit"
    region = "us"

    def __init__(self, client_id: str = "", client_secret: str = "") -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._access_token: str | None = None
        self._universe = set(FULL_UNIVERSE)

    def _authenticate(self) -> bool:
        """Obtain OAuth2 app-only token. Returns True on success."""
        if not self._client_id or not self._client_secret:
            return False

        try:
            resp = httpx.post(
                TOKEN_URL,
                data={"grant_type": "client_credentials"},
                auth=(self._client_id, self._client_secret),
                headers={"User-Agent": USER_AGENT},
                timeout=10,
            )
            resp.raise_for_status()
            self._access_token = resp.json().get("access_token")
            return bool(self._access_token)
        except (httpx.HTTPError, ValueError, KeyError) as e:
            log.warning("reddit_auth_failed", error=str(e))
            return False

    def fetch(self, tickers: list[str] | None = None) -> list[NewsArticle]:
        """Fetch hot posts from finance subreddits.

        Args:
            tickers: Optional ticker filter (unused — we scan all and extract).

        Returns:
            List of NewsArticle objects from high-engagement Reddit posts.
        """
        if not self._client_id or not self._client_secret:
            log.debug("reddit_fetcher_skipped_no_credentials")
            return []

        if not self._access_token and not self._authenticate():
            return []

        articles: list[NewsArticle] = []
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "User-Agent": USER_AGENT,
        }

        for sub in SUBREDDITS:
            try:
                resp = httpx.get(
                    f"{REDDIT_API}/r/{sub}/hot",
                    params={"limit": str(MAX_POSTS_PER_SUB)},
                    headers=headers,
                    timeout=10,
                )
                if resp.status_code == 401:
                    # Token expired — re-auth once
                    if self._authenticate():
                        headers["Authorization"] = f"Bearer {self._access_token}"
                        resp = httpx.get(
                            f"{REDDIT_API}/r/{sub}/hot",
                            params={"limit": str(MAX_POSTS_PER_SUB)},
                            headers=headers,
                            timeout=10,
                        )
                    else:
                        continue

                resp.raise_for_status()
                data = resp.json()

                for post in data.get("data", {}).get("children", []):
                    post_data = post.get("data", {})
                    score = post_data.get("score", 0)
                    if score < MIN_SCORE:
                        continue

                    title = post_data.get("title", "")
                    if not title:
                        continue

                    # Extract tickers from title + selftext
                    selftext = post_data.get("selftext", "")[:500]
                    found_tickers = self._extract_tickers(f"{title} {selftext}", self._universe)

                    # Infer sentiment from flair
                    flair = post_data.get("link_flair_text")
                    sentiment = _infer_sentiment_from_flair(flair)

                    # Published timestamp
                    created_utc = post_data.get("created_utc")
                    published = datetime.fromtimestamp(created_utc, tz=timezone.utc) if created_utc else None

                    articles.append(
                        NewsArticle(
                            headline=title,
                            summary=selftext[:200] if selftext else "",
                            source="reddit",
                            publisher=f"r/{sub}",
                            tickers=found_tickers,
                            published_at=published,
                            url=f"https://reddit.com{post_data.get('permalink', '')}",
                            sentiment=sentiment,
                            region="us",
                            metadata={"score": score, "subreddit": sub, "flair": flair or ""},
                        )
                    )

            except (httpx.HTTPError, ValueError, KeyError) as e:
                log.warning("reddit_fetch_failed", subreddit=sub, error=str(e))

        log.info("reddit_news_fetched", count=len(articles))
        return articles
