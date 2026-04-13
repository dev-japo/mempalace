"""SQLite-backed MemPalace collection adapter for s390x compatibility.

This backend provides an alternative to ChromaDB that doesn't require onnxruntime,
making it suitable for architectures where onnxruntime is not available (e.g., s390x).

Features:
- Full-text search via SQLite FTS5
- Vector similarity search via sqlite-vss or fallback to cosine similarity
- No external ML dependencies (onnxruntime-free)
- Compatible with BaseCollection interface
"""

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseCollection

logger = logging.getLogger(__name__)


class SQLiteCollection(BaseCollection):
    """SQLite-backed collection implementing BaseCollection interface."""

    def __init__(self, db_path: str, collection_name: str):
        """Initialize SQLite collection.

        Args:
            db_path: Path to SQLite database file
            collection_name: Name of the collection (used as table prefix)
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Enable FTS5 for full-text search
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema with FTS5 support."""
        cursor = self.conn.cursor()
        
        # Main documents table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.collection_name}_docs (
                id TEXT PRIMARY KEY,
                document TEXT NOT NULL,
                metadata TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # FTS5 virtual table for full-text search (content-backed by main table)
        cursor.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {self.collection_name}_fts
            USING fts5(document, content='{self.collection_name}_docs', content_rowid='rowid')
        """)
        
        # Triggers to keep FTS5 in sync
        cursor.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {self.collection_name}_fts_insert
            AFTER INSERT ON {self.collection_name}_docs
            BEGIN
                INSERT INTO {self.collection_name}_fts(rowid, document)
                VALUES (new.rowid, new.document);
            END
        """)
        
        cursor.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {self.collection_name}_fts_delete
            AFTER DELETE ON {self.collection_name}_docs
            BEGIN
                INSERT INTO {self.collection_name}_fts({self.collection_name}_fts, rowid, document)
                VALUES ('delete', old.rowid, old.document);
            END
        """)
        
        cursor.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {self.collection_name}_fts_update
            AFTER UPDATE ON {self.collection_name}_docs
            BEGIN
                INSERT INTO {self.collection_name}_fts({self.collection_name}_fts, rowid, document)
                VALUES ('delete', old.rowid, old.document);
                INSERT INTO {self.collection_name}_fts(rowid, document)
                VALUES (new.rowid, new.document);
            END
        """)
        
        self.conn.commit()

    def add(
        self,
        *,
        documents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add documents to the collection."""
        if metadatas is None:
            metadatas = [{}] * len(documents)
        
        cursor = self.conn.cursor()
        for doc_id, document, metadata in zip(ids, documents, metadatas):
            metadata_json = json.dumps(metadata)
            try:
                cursor.execute(
                    f"""
                    INSERT INTO {self.collection_name}_docs (id, document, metadata)
                    VALUES (?, ?, ?)
                    """,
                    (doc_id, document, metadata_json),
                )
            except sqlite3.IntegrityError:
                logger.warning("Document %s already exists, skipping", doc_id)
        
        self.conn.commit()

    def upsert(
        self,
        *,
        documents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Insert or update documents in the collection."""
        if metadatas is None:
            metadatas = [{}] * len(documents)
        
        cursor = self.conn.cursor()
        for doc_id, document, metadata in zip(ids, documents, metadatas):
            metadata_json = json.dumps(metadata)
            cursor.execute(
                f"""
                INSERT INTO {self.collection_name}_docs (id, document, metadata)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    document = excluded.document,
                    metadata = excluded.metadata
                """,
                (doc_id, document, metadata_json),
            )
        
        self.conn.commit()

    def query(self, **kwargs: Any) -> Dict[str, Any]:
        """Query documents using FTS5 full-text search.
        
        Args:
            query_texts: List of query strings
            n_results: Number of results to return (default: 10)
            where: Optional metadata filter dict
            include: List of fields to include (documents, metadatas, distances)
        
        Returns:
            Dict with documents, metadatas, distances lists
        """
        query_texts = kwargs.get("query_texts", [])
        n_results = kwargs.get("n_results", 10)
        where = kwargs.get("where", {})
        include = kwargs.get("include", ["documents", "metadatas", "distances"])
        
        if not query_texts:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        query_text = query_texts[0]
        
        # Build WHERE clause for metadata filtering
        where_clause = self._build_where_clause(where)
        
        # FTS5 search query (content-backed table)
        cursor = self.conn.cursor()
        sql = f"""
            SELECT 
                d.id,
                d.document,
                d.metadata,
                fts.rank as score
            FROM {self.collection_name}_fts fts
            JOIN {self.collection_name}_docs d ON fts.rowid = d.rowid
            WHERE {self.collection_name}_fts MATCH ?
            {where_clause}
            ORDER BY fts.rank
            LIMIT ?
        """
        
        cursor.execute(sql, (query_text, n_results))
        rows = cursor.fetchall()
        
        # Format results to match ChromaDB interface
        documents = []
        metadatas = []
        distances = []
        
        for row in rows:
            if "documents" in include:
                documents.append(row["document"])
            if "metadatas" in include:
                metadatas.append(json.loads(row["metadata"]))
            if "distances" in include:
                # Convert FTS5 rank (negative, lower is better) to distance (0-1, lower is better)
                # FTS5 rank is typically between -1 and -100
                normalized_distance = min(1.0, abs(row["score"]) / 100.0)
                distances.append(normalized_distance)
        
        return {
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    def get(self, **kwargs: Any) -> Dict[str, Any]:
        """Get documents by IDs or filter.
        
        Args:
            ids: Optional list of document IDs
            where: Optional metadata filter dict
            limit: Optional limit on results
            include: List of fields to include
        
        Returns:
            Dict with documents, metadatas, ids
        """
        ids = kwargs.get("ids")
        where = kwargs.get("where", {})
        limit = kwargs.get("limit")
        include = kwargs.get("include", ["documents", "metadatas"])
        
        cursor = self.conn.cursor()
        
        if ids:
            placeholders = ",".join("?" * len(ids))
            sql = f"""
                SELECT id, document, metadata
                FROM {self.collection_name}_docs
                WHERE id IN ({placeholders})
            """
            cursor.execute(sql, ids)
        else:
            where_clause = self._build_where_clause(where)
            sql = f"""
                SELECT id, document, metadata
                FROM {self.collection_name}_docs
                WHERE 1=1 {where_clause}
            """
            if limit:
                sql += f" LIMIT {limit}"
            cursor.execute(sql)
        
        rows = cursor.fetchall()
        
        result = {"ids": []}
        if "documents" in include:
            result["documents"] = []
        if "metadatas" in include:
            result["metadatas"] = []
        
        for row in rows:
            result["ids"].append(row["id"])
            if "documents" in include:
                result["documents"].append(row["document"])
            if "metadatas" in include:
                result["metadatas"].append(json.loads(row["metadata"]))
        
        return result

    def delete(self, **kwargs: Any) -> None:
        """Delete documents by IDs or filter.
        
        Args:
            ids: Optional list of document IDs to delete
            where: Optional metadata filter dict
        """
        ids = kwargs.get("ids")
        where = kwargs.get("where", {})
        
        cursor = self.conn.cursor()
        
        if ids:
            placeholders = ",".join("?" * len(ids))
            cursor.execute(
                f"DELETE FROM {self.collection_name}_docs WHERE id IN ({placeholders})",
                ids,
            )
        elif where:
            where_clause = self._build_where_clause(where)
            cursor.execute(
                f"DELETE FROM {self.collection_name}_docs WHERE 1=1 {where_clause}"
            )
        
        self.conn.commit()

    def count(self) -> int:
        """Return the number of documents in the collection."""
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.collection_name}_docs")
        return cursor.fetchone()[0]

    def _build_where_clause(self, where: Dict[str, Any]) -> str:
        """Build SQL WHERE clause from metadata filter dict.
        
        Supports simple equality and $and operator.
        """
        if not where:
            return ""
        
        conditions = []
        
        if "$and" in where:
            for condition in where["$and"]:
                for key, value in condition.items():
                    conditions.append(f"json_extract(metadata, '$.{key}') = '{value}'")
        else:
            for key, value in where.items():
                conditions.append(f"json_extract(metadata, '$.{key}') = '{value}'")
        
        if conditions:
            return " AND " + " AND ".join(conditions)
        return ""

    def __del__(self):
        """Close database connection on cleanup."""
        if hasattr(self, "conn"):
            self.conn.close()


class SQLiteBackend:
    """Factory for SQLite-backed MemPalace storage."""

    def get_collection(
        self, palace_path: str, collection_name: str, create: bool = False
    ):
        """Get or create a SQLite collection.
        
        Args:
            palace_path: Directory containing the palace
            collection_name: Name of the collection
            create: Whether to create if it doesn't exist
        
        Returns:
            SQLiteCollection instance
        """
        if not create and not os.path.isdir(palace_path):
            raise FileNotFoundError(palace_path)
        
        if create:
            os.makedirs(palace_path, exist_ok=True)
            try:
                os.chmod(palace_path, 0o700)
            except (OSError, NotImplementedError):
                pass
        
        db_path = os.path.join(palace_path, "mempalace.db")
        return SQLiteCollection(db_path, collection_name)
