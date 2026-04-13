"""Storage backend implementations for MemPalace."""

import logging

from .base import BaseCollection
from .sqlite_backend import SQLiteBackend, SQLiteCollection

logger = logging.getLogger(__name__)

__all__ = [
    "BaseCollection",
    "SQLiteBackend",
    "SQLiteCollection",
]

# ChromaDB is optional (not available on all architectures like s390x)
try:
    from .chroma import ChromaBackend, ChromaCollection
    __all__.extend(["ChromaBackend", "ChromaCollection"])
except Exception as exc:
    logger.info("Chroma backend unavailable, falling back to SQLite: %s", exc)
    ChromaBackend = None
    ChromaCollection = None
