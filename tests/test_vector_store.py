"""Tests for VectorStore — search_similar with days_back and cleanup_old.

Uses a mocked ChromaDB collection — no external service required.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

from src.storage.vector_store import VectorStore


def _make_store() -> tuple[VectorStore, MagicMock]:
    """Return a VectorStore with a mocked collection."""
    store = VectorStore(host="localhost", port=8000)
    mock_col = MagicMock()
    store._collection = mock_col
    return store, mock_col


# ── search_similar ────────────────────────────────────────────────────────────


class TestSearchSimilar:
    def test_no_filter(self) -> None:
        store, col = _make_store()
        col.query.return_value = {"documents": [["headline"]], "metadatas": [[{}]], "distances": [[0.1]]}

        store.search_similar("NVDA earnings")

        call_kwargs = col.query.call_args.kwargs
        assert call_kwargs["where"] is None

    def test_ticker_only_filter(self) -> None:
        store, col = _make_store()
        col.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        store.search_similar("NVDA earnings", ticker="NVDA")

        call_kwargs = col.query.call_args.kwargs
        assert call_kwargs["where"] == {"ticker": {"$eq": "NVDA"}}

    def test_days_back_only_filter(self) -> None:
        store, col = _make_store()
        col.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        with patch("src.storage.vector_store.datetime") as mock_dt:
            mock_dt.utcnow.return_value = datetime(2026, 2, 23)
            store.search_similar("macro risk", days_back=30)

        call_kwargs = col.query.call_args.kwargs
        assert call_kwargs["where"] == {"published_at": {"$gte": "2026-01-24"}}

    def test_ticker_and_days_back_uses_and_filter(self) -> None:
        store, col = _make_store()
        col.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        with patch("src.storage.vector_store.datetime") as mock_dt:
            mock_dt.utcnow.return_value = datetime(2026, 2, 23)
            store.search_similar("NVDA guidance", ticker="NVDA", days_back=30)

        call_kwargs = col.query.call_args.kwargs
        where = call_kwargs["where"]
        assert "$and" in where
        conditions = where["$and"]
        assert {"ticker": {"$eq": "NVDA"}} in conditions
        assert {"published_at": {"$gte": "2026-01-24"}} in conditions

    def test_returns_collection_results(self) -> None:
        store, col = _make_store()
        expected = {
            "documents": [["NVDA beats estimates"]],
            "metadatas": [[{"ticker": "NVDA"}]],
            "distances": [[0.12]],
        }
        col.query.return_value = expected

        result = store.search_similar("NVDA earnings")

        assert result == expected


# ── cleanup_old ───────────────────────────────────────────────────────────────


class TestCleanupOld:
    def test_deletes_old_articles(self) -> None:
        store, col = _make_store()
        col.get.return_value = {"ids": ["id1", "id2", "id3"], "documents": [], "metadatas": []}

        with patch("src.storage.vector_store.datetime") as mock_dt:
            mock_dt.utcnow.return_value = datetime(2026, 2, 23)
            deleted = store.cleanup_old(days=180)

        assert deleted == 3
        col.delete.assert_called_once_with(ids=["id1", "id2", "id3"])

    def test_returns_zero_when_nothing_to_delete(self) -> None:
        store, col = _make_store()
        col.get.return_value = {"ids": [], "documents": [], "metadatas": []}

        deleted = store.cleanup_old(days=180)

        assert deleted == 0
        col.delete.assert_not_called()

    def test_uses_correct_cutoff_date(self) -> None:
        store, col = _make_store()
        col.get.return_value = {"ids": [], "documents": [], "metadatas": []}

        with patch("src.storage.vector_store.datetime") as mock_dt:
            mock_dt.utcnow.return_value = datetime(2026, 2, 23)
            store.cleanup_old(days=180)

        call_kwargs = col.get.call_args.kwargs
        assert call_kwargs["where"] == {"published_at": {"$lt": "2025-08-27"}}

    def test_deletes_in_batches(self) -> None:
        store, col = _make_store()
        ids = [f"id{i}" for i in range(250)]
        col.get.return_value = {"ids": ids, "documents": [], "metadatas": []}

        deleted = store.cleanup_old(days=180)

        assert deleted == 250
        # Should be called 3 times: 100 + 100 + 50
        assert col.delete.call_count == 3

    def test_default_retention_180_days(self) -> None:
        store, col = _make_store()
        col.get.return_value = {"ids": [], "documents": [], "metadatas": []}

        with patch("src.storage.vector_store.datetime") as mock_dt:
            mock_dt.utcnow.return_value = datetime(2026, 2, 23)
            store.cleanup_old()  # no days param — should default to 180

        call_kwargs = col.get.call_args.kwargs
        assert call_kwargs["where"] == {"published_at": {"$lt": "2025-08-27"}}
