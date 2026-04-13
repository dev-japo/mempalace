"""Tests for SQLite backend implementation."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from mempalace.backends.sqlite_backend import SQLiteBackend, SQLiteCollection


@pytest.fixture
def temp_palace():
    """Create a temporary palace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sqlite_collection(temp_palace):
    """Create a SQLite collection for testing."""
    backend = SQLiteBackend()
    return backend.get_collection(temp_palace, "test_collection", create=True)


def test_sqlite_backend_creation(temp_palace):
    """Test SQLite backend can be created."""
    backend = SQLiteBackend()
    col = backend.get_collection(temp_palace, "test", create=True)
    assert isinstance(col, SQLiteCollection)
    assert col.count() == 0


def test_add_documents(sqlite_collection):
    """Test adding documents to SQLite collection."""
    sqlite_collection.add(
        documents=["Document 1", "Document 2", "Document 3"],
        ids=["id1", "id2", "id3"],
        metadatas=[
            {"wing": "test", "room": "room1"},
            {"wing": "test", "room": "room2"},
            {"wing": "other", "room": "room1"},
        ],
    )
    assert sqlite_collection.count() == 3


def test_upsert_documents(sqlite_collection):
    """Test upserting documents (insert or update)."""
    # Initial insert
    sqlite_collection.add(
        documents=["Original document"],
        ids=["id1"],
        metadatas=[{"wing": "test", "room": "room1"}],
    )
    assert sqlite_collection.count() == 1

    # Upsert with same ID should update
    sqlite_collection.upsert(
        documents=["Updated document"],
        ids=["id1"],
        metadatas=[{"wing": "test", "room": "room2"}],
    )
    assert sqlite_collection.count() == 1

    # Verify update
    result = sqlite_collection.get(ids=["id1"])
    assert result["documents"][0] == "Updated document"
    assert result["metadatas"][0]["room"] == "room2"


def test_query_documents(sqlite_collection):
    """Test FTS5 full-text search query."""
    sqlite_collection.add(
        documents=[
            "Python programming language",
            "JavaScript web development",
            "Python data science",
        ],
        ids=["id1", "id2", "id3"],
        metadatas=[
            {"wing": "languages", "room": "python"},
            {"wing": "languages", "room": "javascript"},
            {"wing": "languages", "room": "python"},
        ],
    )

    # Query for Python
    results = sqlite_collection.query(query_texts=["Python"], n_results=5)
    assert len(results["documents"][0]) == 2
    assert all("Python" in doc for doc in results["documents"][0])


def test_query_with_metadata_filter(sqlite_collection):
    """Test query with metadata filtering."""
    sqlite_collection.add(
        documents=["Doc 1", "Doc 2", "Doc 3"],
        ids=["id1", "id2", "id3"],
        metadatas=[
            {"wing": "test", "room": "room1"},
            {"wing": "test", "room": "room2"},
            {"wing": "other", "room": "room1"},
        ],
    )

    # Query with wing filter
    results = sqlite_collection.query(
        query_texts=["Doc"],
        n_results=5,
        where={"wing": "test"},
    )
    assert len(results["documents"][0]) == 2

    # Query with $and filter
    results = sqlite_collection.query(
        query_texts=["Doc"],
        n_results=5,
        where={"$and": [{"wing": "test"}, {"room": "room1"}]},
    )
    assert len(results["documents"][0]) == 1


def test_get_by_ids(sqlite_collection):
    """Test getting documents by IDs."""
    sqlite_collection.add(
        documents=["Doc 1", "Doc 2", "Doc 3"],
        ids=["id1", "id2", "id3"],
        metadatas=[{"wing": "test", "room": "room1"}] * 3,
    )

    result = sqlite_collection.get(ids=["id1", "id3"])
    assert len(result["ids"]) == 2
    assert "id1" in result["ids"]
    assert "id3" in result["ids"]


def test_get_with_metadata_filter(sqlite_collection):
    """Test getting documents with metadata filter."""
    sqlite_collection.add(
        documents=["Doc 1", "Doc 2", "Doc 3"],
        ids=["id1", "id2", "id3"],
        metadatas=[
            {"wing": "test", "room": "room1"},
            {"wing": "test", "room": "room2"},
            {"wing": "other", "room": "room1"},
        ],
    )

    result = sqlite_collection.get(where={"wing": "test"})
    assert len(result["ids"]) == 2


def test_delete_by_ids(sqlite_collection):
    """Test deleting documents by IDs."""
    sqlite_collection.add(
        documents=["Doc 1", "Doc 2", "Doc 3"],
        ids=["id1", "id2", "id3"],
        metadatas=[{"wing": "test", "room": "room1"}] * 3,
    )
    assert sqlite_collection.count() == 3

    sqlite_collection.delete(ids=["id1", "id3"])
    assert sqlite_collection.count() == 1

    result = sqlite_collection.get(ids=["id2"])
    assert len(result["ids"]) == 1


def test_delete_with_metadata_filter(sqlite_collection):
    """Test deleting documents with metadata filter."""
    sqlite_collection.add(
        documents=["Doc 1", "Doc 2", "Doc 3"],
        ids=["id1", "id2", "id3"],
        metadatas=[
            {"wing": "test", "room": "room1"},
            {"wing": "test", "room": "room2"},
            {"wing": "other", "room": "room1"},
        ],
    )
    assert sqlite_collection.count() == 3

    sqlite_collection.delete(where={"wing": "test"})
    assert sqlite_collection.count() == 1


def test_count(sqlite_collection):
    """Test document count."""
    assert sqlite_collection.count() == 0

    sqlite_collection.add(
        documents=["Doc 1", "Doc 2"],
        ids=["id1", "id2"],
        metadatas=[{"wing": "test", "room": "room1"}] * 2,
    )
    assert sqlite_collection.count() == 2


def test_backend_auto_selection():
    """Test automatic backend selection."""
    from mempalace.palace import _get_backend

    # Should return SQLiteBackend when chromadb not available
    backend = _get_backend()
    assert backend is not None


def test_backend_env_var_sqlite(temp_palace, monkeypatch):
    """Test MEMPALACE_BACKEND=sqlite environment variable."""
    monkeypatch.setenv("MEMPALACE_BACKEND", "sqlite")
    from mempalace.palace import _get_backend

    backend = _get_backend()
    assert isinstance(backend, SQLiteBackend)


def test_fts5_triggers(temp_palace):
    """Test that FTS5 triggers keep search index in sync."""
    backend = SQLiteBackend()
    col = backend.get_collection(temp_palace, "test", create=True)

    # Add two documents with distinct content
    col.add(
        documents=["Python programming language", "Rust systems programming"],
        ids=["id1", "id2"],
        metadatas=[{}, {}]
    )

    # Both should be findable
    results = col.query(query_texts=["Python"], n_results=2)
    assert len(results["documents"][0]) == 1
    results = col.query(query_texts=["Rust"], n_results=2)
    assert len(results["documents"][0]) == 1

    # Update first document
    col.upsert(documents=["JavaScript web development"], ids=["id1"], metadatas=[{}])

    # Should find JavaScript now
    results = col.query(query_texts=["JavaScript"], n_results=2)
    assert len(results["documents"][0]) == 1
    assert results["documents"][0][0] == "JavaScript web development"

    # Rust should still be findable
    results = col.query(query_texts=["Rust"], n_results=2)
    assert len(results["documents"][0]) == 1

    # Delete first document
    col.delete(ids=["id1"])

    # JavaScript should not be found
    results = col.query(query_texts=["JavaScript"], n_results=2)
    assert len(results["documents"][0]) == 0

    # Rust should still be there
    results = col.query(query_texts=["Rust"], n_results=2)
    assert len(results["documents"][0]) == 1
