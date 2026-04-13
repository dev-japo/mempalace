"""Storage backend implementations for MemPalace."""

from .base import BaseCollection
from .sqlite_backend import SQLiteBackend, SQLiteCollection

__all__ = [
    "BaseCollection",
    "SQLiteBackend",
    "SQLiteCollection",
]

# ChromaDB is optional (not available on all architectures like s390x)
try:
    from .chroma import ChromaBackend, ChromaCollection
    __all__.extend(["ChromaBackend", "ChromaCollection"])
except ImportError:
    ChromaBackend = None
    ChromaCollection = None
