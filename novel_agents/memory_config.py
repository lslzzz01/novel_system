from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_MEMORY_SETTINGS: dict[str, Any] = {
    "enabled": True,
    "target_book_chars": {
        "minimum": 700_000,
        "maximum": 1_200_000,
    },
    "indexing": {
        "batch_size": 8,
        "max_parallel_jobs": 1,
        "pause_during_writing": True,
        "checkpoint_every_chapters": 10,
    },
    "embedding": {
        "provider": "local_onnx",
        "model": "BAAI/bge-small-zh-v1.5",
        "model_path": "",
        "device": "cpu",
        "quantization": "int8",
        "dimensions": 512,
        "local_files_only": True,
    },
    "chunking": {
        "target_chars": 400,
        "max_chars": 600,
        "overlap_chars": 80,
        "preserve_scene_boundary": True,
        "preserve_dialogue_block": True,
    },
    "retrieval": {
        "vector_top_k": 40,
        "metadata_filtered_top_k": 25,
        "rerank_top_k": 12,
        "final_context_chunks": 8,
        "minimum_similarity": 0.55,
        "recent_full_chapters": 5,
        "maximum_context_chars": 240_000,
    },
    "policy": {
        "index_source": "finalized_only",
        "strict_latest_chapter_sync": True,
        "style_learning": False,
        "style_learning_min_chapters": 5,
    },
}


class MemorySettingsStore:
    """Per-project memory settings. Absence means memory is not enabled."""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.memory_dir = self.project_root / "memory"
        self.path = self.memory_dir / "settings.json"

    def exists(self) -> bool:
        return self.path.is_file()

    def initialize(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.exists():
            return self.load()
        settings = copy.deepcopy(DEFAULT_MEMORY_SETTINGS)
        if overrides:
            _deep_merge(settings, overrides)
        self._validate(settings)
        self._write(settings)
        return settings

    def load(self) -> dict[str, Any]:
        if not self.exists():
            raise FileNotFoundError("该项目尚未启用记忆系统")
        with self.path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("memory/settings.json 必须是 JSON 对象")
        settings = copy.deepcopy(DEFAULT_MEMORY_SETTINGS)
        _deep_merge(settings, data)
        self._validate(settings)
        return settings

    def public(self) -> dict[str, Any]:
        if not self.exists():
            return {
                "enabled": False,
                "available_defaults": copy.deepcopy(DEFAULT_MEMORY_SETTINGS),
            }
        return self.load()

    def update(self, values: dict[str, Any]) -> dict[str, Any]:
        settings = self.load() if self.exists() else copy.deepcopy(DEFAULT_MEMORY_SETTINGS)
        _deep_merge(settings, values)
        self._validate(settings)
        self._write(settings)
        return settings

    @staticmethod
    def requires_rebuild(before: dict[str, Any], after: dict[str, Any]) -> bool:
        watched = (
            ("embedding", "provider"),
            ("embedding", "model"),
            ("embedding", "model_path"),
            ("embedding", "dimensions"),
            ("embedding", "quantization"),
            ("chunking", "target_chars"),
            ("chunking", "max_chars"),
            ("chunking", "overlap_chars"),
            ("policy", "style_learning"),
        )
        return any(before[group][key] != after[group][key] for group, key in watched)

    def _write(self, settings: dict[str, Any]) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=".settings.", suffix=".tmp", dir=self.memory_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(settings, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
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
    def _validate(settings: dict[str, Any]) -> None:
        target = settings["target_book_chars"]
        if not 100_000 <= int(target["minimum"]) <= int(target["maximum"]):
            raise ValueError("目标最小字数必须大于等于 100000 且不超过最大字数")
        if int(target["maximum"]) > 5_000_000:
            raise ValueError("目标最大字数不能超过 5000000")

        indexing = settings["indexing"]
        _range("batch_size", indexing["batch_size"], 1, 64)
        _range("max_parallel_jobs", indexing["max_parallel_jobs"], 1, 4)
        _range(
            "checkpoint_every_chapters",
            indexing["checkpoint_every_chapters"],
            1,
            100,
        )
        if not isinstance(indexing["pause_during_writing"], bool):
            raise ValueError("pause_during_writing 必须是布尔值")

        embedding = settings["embedding"]
        if embedding["provider"] not in {"local_onnx", "local_torch", "hash_test"}:
            raise ValueError("不支持的本地嵌入后端")
        if embedding["device"] not in {"cpu", "cuda"}:
            raise ValueError("嵌入设备只能是 cpu 或 cuda")
        if embedding["quantization"] not in {"int8", "fp16", "fp32"}:
            raise ValueError("不支持的量化方式")
        _range("dimensions", embedding["dimensions"], 64, 4096)

        chunking = settings["chunking"]
        _range("target_chars", chunking["target_chars"], 100, 4000)
        _range("max_chars", chunking["max_chars"], 100, 8000)
        _range("overlap_chars", chunking["overlap_chars"], 0, 1000)
        if int(chunking["target_chars"]) > int(chunking["max_chars"]):
            raise ValueError("目标分块长度不能超过最大分块长度")
        if int(chunking["overlap_chars"]) >= int(chunking["target_chars"]):
            raise ValueError("重叠长度必须小于目标分块长度")
        for name in ("preserve_scene_boundary", "preserve_dialogue_block"):
            if not isinstance(chunking[name], bool):
                raise ValueError(f"{name} 必须是布尔值")

        retrieval = settings["retrieval"]
        _range("vector_top_k", retrieval["vector_top_k"], 1, 500)
        _range(
            "metadata_filtered_top_k",
            retrieval["metadata_filtered_top_k"],
            1,
            500,
        )
        _range("rerank_top_k", retrieval["rerank_top_k"], 1, 100)
        _range("final_context_chunks", retrieval["final_context_chunks"], 1, 50)
        similarity = float(retrieval["minimum_similarity"])
        if not -1 <= similarity <= 1:
            raise ValueError("最低相似度必须在 -1 到 1 之间")
        _range("recent_full_chapters", retrieval["recent_full_chapters"], 0, 50)
        _range(
            "maximum_context_chars", retrieval["maximum_context_chars"], 10_000, 2_000_000
        )

        policy = settings["policy"]
        for name in ("strict_latest_chapter_sync", "style_learning"):
            if not isinstance(policy[name], bool):
                raise ValueError(f"{name} 必须是布尔值")
        _range(
            "style_learning_min_chapters",
            policy["style_learning_min_chapters"],
            1,
            100,
        )


def _range(name: str, value: Any, minimum: int, maximum: int) -> None:
    number = int(value)
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} 必须在 {minimum} 到 {maximum} 之间")


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
