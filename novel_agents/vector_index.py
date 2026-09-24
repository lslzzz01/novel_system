from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class VectorIndex(Protocol):
    dimensions: int

    def upsert(self, ids: list[str], vectors: np.ndarray) -> None: ...
    def delete(self, ids: list[str]) -> None: ...
    def search(self, vector: np.ndarray, top_k: int) -> list[dict[str, Any]]: ...
    def count(self) -> int: ...


class NumpyVectorIndex:
    """Exact cosine index. A few thousand 512D vectors fit lightweight laptops."""

    _locks_guard = threading.Lock()
    _path_locks: dict[Path, threading.RLock] = {}

    def __init__(self, memory_dir: str | Path, dimensions: int) -> None:
        self.memory_dir = Path(memory_dir).resolve()
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.memory_dir / "vector_index.npz"
        self.dimensions = int(dimensions)
        with self._locks_guard:
            self._lock = self._path_locks.setdefault(self.path, threading.RLock())

    def upsert(self, ids: list[str], vectors: np.ndarray) -> None:
        if not ids:
            return
        matrix = self._normalize(np.asarray(vectors, dtype=np.float32))
        if matrix.shape != (len(ids), self.dimensions):
            raise ValueError(
                f"向量形状应为 ({len(ids)}, {self.dimensions})，实际为 {matrix.shape}"
            )
        with self._lock:
            existing_ids, existing_vectors = self._load()
            replace = set(ids)
            keep_indexes = [index for index, item in enumerate(existing_ids) if item not in replace]
            kept_ids = [existing_ids[index] for index in keep_indexes]
            kept_vectors = (
                existing_vectors[keep_indexes]
                if keep_indexes
                else np.empty((0, self.dimensions), dtype=np.float32)
            )
            all_ids = kept_ids + ids
            all_vectors = np.vstack([kept_vectors, matrix])
            self._save(all_ids, all_vectors)

    def delete(self, ids: list[str]) -> None:
        if not ids or not self.path.exists():
            return
        remove = set(ids)
        with self._lock:
            existing_ids, vectors = self._load()
            indexes = [index for index, item in enumerate(existing_ids) if item not in remove]
            self._save(
                [existing_ids[index] for index in indexes],
                vectors[indexes] if indexes else np.empty((0, self.dimensions), dtype=np.float32),
            )

    def delete_prefix(self, prefix: str) -> None:
        with self._lock:
            ids, vectors = self._load()
            indexes = [index for index, item in enumerate(ids) if not item.startswith(prefix)]
            self._save(
                [ids[index] for index in indexes],
                vectors[indexes]
                if indexes
                else np.empty((0, self.dimensions), dtype=np.float32),
            )

    def search(self, vector: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        with self._lock:
            ids, matrix = self._load()
        if not ids:
            return []
        query = self._normalize(np.asarray(vector, dtype=np.float32).reshape(1, -1))[0]
        if query.shape[0] != self.dimensions:
            raise ValueError("查询向量维度与索引不一致")
        scores = matrix @ query
        count = min(max(1, int(top_k)), len(ids))
        indexes = np.argpartition(-scores, count - 1)[:count]
        ordered = indexes[np.argsort(-scores[indexes])]
        return [{"id": ids[int(index)], "score": float(scores[int(index)])} for index in ordered]

    def count(self) -> int:
        with self._lock:
            ids, _ = self._load()
        return len(ids)

    def clear(self) -> None:
        with self._lock:
            self._save([], np.empty((0, self.dimensions), dtype=np.float32))

    def _load(self) -> tuple[list[str], np.ndarray]:
        if not self.path.exists():
            return [], np.empty((0, self.dimensions), dtype=np.float32)
        with np.load(self.path, allow_pickle=False) as data:
            ids = [str(item) for item in data["ids"].tolist()]
            vectors = np.asarray(data["vectors"], dtype=np.float32)
            stored_dimensions = int(data["dimensions"].item())
        if stored_dimensions != self.dimensions:
            raise ValueError("现有向量索引维度与当前配置不一致，需要重建索引")
        return ids, vectors

    def _save(self, ids: list[str], vectors: np.ndarray) -> None:
        fd, temp_name = tempfile.mkstemp(prefix=".vector.", suffix=".npz", dir=self.memory_dir)
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez_compressed(
                    handle,
                    ids=np.asarray(ids, dtype=np.str_),
                    vectors=np.asarray(vectors, dtype=np.float32),
                    dimensions=np.asarray(self.dimensions, dtype=np.int32),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms
