"""ChromaDB vector store client for news embeddings.

Connects to ChromaDB running in Docker on port 8000.
"""

from typing import Any

import chromadb
import structlog

log = structlog.get_logger()

_COLLECTION_NAME = "kukulkan_news"


class VectorStore:
    """Client for ChromaDB news embedding storage."""

    def __init__(self, host: str = "localhost", port: int = 8000) -> None:
        self._host = host
        self._port = port
        self._client: chromadb.HttpClient | None = None
        self._collection: Any = None

    def connect(self) -> None:
        """Connect to ChromaDB and get or create the news collection."""
        self._client = chromadb.HttpClient(host=self._host, port=self._port)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            "chromadb_connected",
            host=self._host,
            port=self._port,
            collection=_COLLECTION_NAME,
        )

    @property
    def collection(self) -> Any:
        """Get the underlying collection, connecting if needed."""
        if self._collection is None:
            self.connect()
        return self._collection

    def add_news(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a news article to the vector store.

        Args:
            doc_id: Unique identifier for the document.
            text: The article text to embed and store.
            metadata: Optional metadata (ticker, source, date, etc.).
        """
        self.collection.add(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata or {}],
        )
        log.debug("news_added", doc_id=doc_id)

    def search_similar(self, query: str, n_results: int = 5, ticker: str | None = None) -> dict[str, Any]:
        """Search for news similar to a query string.

        Args:
            query: Text to find similar articles for.
            n_results: Maximum number of results to return.
            ticker: Optional ticker filter.

        Returns:
            ChromaDB query results dict with ids, documents, metadatas, distances.
        """
        where_filter = {"ticker": ticker} if ticker else None
        results: dict[str, Any] = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
        )
        return results

    def get_by_ticker(self, ticker: str, limit: int = 20) -> dict[str, Any]:
        """Get all stored news for a specific ticker.

        Args:
            ticker: The ticker symbol to filter by.
            limit: Maximum number of results.

        Returns:
            ChromaDB get results dict.
        """
        results: dict[str, Any] = self.collection.get(
            where={"ticker": ticker},
            limit=limit,
        )
        return results

    def count(self) -> int:
        """Get total number of documents in the collection."""
        return self.collection.count()
