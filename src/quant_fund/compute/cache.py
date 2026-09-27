"""Content-hash cache for repeated model fits.

The key is a SHA-256 of the caller tag plus the raw bytes of each array.
A hit returns the stored payload; it does not refit. Payloads are whatever
the caller stored (typically a snapshot dict). The cache is process-local
and bounded.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any

import numpy as np


class FitCache:
    """LRU map from content hashes to fit snapshots."""

    def __init__(self, max_entries: int = 64) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._max = int(max_entries)
        self._lock = threading.Lock()
        self._store: OrderedDict[str, Any] = OrderedDict()

    def hash_key(self, tag: tuple[Any, ...], arrays: tuple[np.ndarray, ...]) -> str:
        digest = hashlib.sha256()
        digest.update(repr(tag).encode())
        for raw in arrays:
            arr = np.ascontiguousarray(raw)
            digest.update(repr(arr.shape).encode())
            digest.update(str(arr.dtype).encode())
            digest.update(memoryview(arr.view(np.uint8)))
        return digest.hexdigest()

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key not in self._store:
                return None
            self._store.move_to_end(key)
            return self._store[key]

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
