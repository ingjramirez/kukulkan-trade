"""News fetcher using yfinance and ChromaDB for semantic storage.

Pulls recent news headlines for universe tickers, deduplicates,
stores embeddings in ChromaDB, and logs to the news_log table.
"""

import hashlib
from datetime import datetime
from typing import Any

import structlog
import yfinance as yf

from config.settings import settings
from src.storage.models import NewsLogRow
from src.storage.vector_store import VectorStore

log = structlog.get_logger()


class NewsFetcher:
    """Fetches ticker news from yfinance and stores in ChromaDB."""

    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self._vector_store = vector_store

    @property
    def vector_store(self) -> VectorStore:
        """Lazy-init vector store."""
        if self._vector_store is None:
            self._vector_store = VectorStore(
                host=settings.chroma.host,
                port=settings.chroma.port,
            )
        return self._vector_store

    def fetch_news(self, tickers: list[str], max_per_ticker: int = 5) -> list[dict[str, Any]]:
        """Fetch news headlines from yfinance for a list of tickers.

        Args:
            tickers: List of ticker symbols.
            max_per_ticker: Maximum articles per ticker.

        Returns:
            List of article dicts with keys: ticker, title, link, publisher, published.
        """
        all_articles: list[dict[str, Any]] = []
        seen_titles: set[str] = set()

        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                news = t.news or []

                for article in news[:max_per_ticker]:
                    title = article.get("title", "")
                    if not title or title in seen_titles:
                        continue
                    seen_titles.add(title)

                    published_ts = article.get("providerPublishTime")
                    published = (
                        datetime.fromtimestamp(published_ts)
                        if published_ts
                        else None
                    )

                    all_articles.append({
                        "ticker": ticker,
                        "title": title,
                        "link": article.get("link", ""),
                        "publisher": article.get("publisher", ""),
                        "published": published,
                    })
            except Exception as e:
                log.warning("news_fetch_failed", ticker=ticker, error=str(e))

        log.info("news_fetched", total=len(all_articles), tickers=len(tickers))
        return all_articles

    def store_articles(self, articles: list[dict[str, Any]]) -> list[NewsLogRow]:
        """Store articles in ChromaDB and return NewsLogRow objects for DB persistence.

        Args:
            articles: List of article dicts from fetch_news().

        Returns:
            List of NewsLogRow objects ready to be saved.
        """
        rows: list[NewsLogRow] = []

        for article in articles:
            title = article["title"]
            doc_id = _article_id(article)

            # Store in ChromaDB
            try:
                self.vector_store.add_news(
                    doc_id=doc_id,
                    text=title,
                    metadata={
                        "ticker": article["ticker"],
                        "publisher": article.get("publisher", ""),
                        "link": article.get("link", ""),
                    },
                )
            except Exception as e:
                log.warning("chromadb_store_failed", doc_id=doc_id, error=str(e))

            # Build DB row
            rows.append(NewsLogRow(
                ticker=article["ticker"],
                headline=title,
                source=article.get("publisher", ""),
                url=article.get("link", ""),
                published_at=article.get("published"),
                embedding_id=doc_id,
            ))

        log.info("news_stored", count=len(rows))
        return rows

    def search_relevant(self, query: str, n_results: int = 10) -> list[dict[str, Any]]:
        """Search ChromaDB for news relevant to a query.

        Args:
            query: Search query (e.g., market theme or ticker).
            n_results: Max results to return.

        Returns:
            List of dicts with title, ticker, distance.
        """
        results = self.vector_store.search_similar(query, n_results=n_results)

        articles = []
        if results and results.get("documents"):
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, dists):
                articles.append({
                    "title": doc,
                    "ticker": meta.get("ticker", ""),
                    "publisher": meta.get("publisher", ""),
                    "distance": dist,
                })

        return articles

    def get_news_context(self, tickers: list[str], n_results: int = 8) -> str:
        """Build a news context string for the Claude agent prompt.

        Searches ChromaDB for the most relevant recent headlines
        across the given tickers. Deduplicates by ticker to maximize
        coverage across different names.

        Args:
            tickers: Tickers to search news for.
            n_results: Max headlines to include (default 8).

        Returns:
            Formatted text block for the agent prompt.
        """
        # Search with a broad market query — fetch extra for dedup
        query = f"market news for {', '.join(tickers[:10])}"
        articles = self.search_relevant(query, n_results=n_results * 2)

        if not articles:
            return "  (no recent news available)"

        # Deduplicate: prefer unique tickers first
        seen_tickers: set[str] = set()
        unique_first: list[dict] = []
        duplicates: list[dict] = []

        for a in articles:
            ticker = a.get("ticker", "")
            if ticker not in seen_tickers:
                seen_tickers.add(ticker)
                unique_first.append(a)
            else:
                duplicates.append(a)

        # Fill up to n_results, unique tickers first
        selected = (unique_first + duplicates)[:n_results]

        lines = []
        for a in selected:
            ticker = a.get("ticker", "")
            title = a.get("title", "")
            publisher = a.get("publisher", "")
            source_str = f" ({publisher})" if publisher else ""
            lines.append(f"  [{ticker}] {title}{source_str}")

        return "\n".join(lines)


def _article_id(article: dict) -> str:
    """Generate a deterministic ID for an article based on title + ticker."""
    raw = f"{article.get('ticker', '')}:{article.get('title', '')}"
    return hashlib.md5(raw.encode()).hexdigest()
