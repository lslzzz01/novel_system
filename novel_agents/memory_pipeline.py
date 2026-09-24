from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from .chapter_versions import ChapterVersionStore, content_hash
from .embedding import EmbeddingProvider, create_embedder
from .llm import LanguageModel
from .memory_agents import (
    MEMORY_COMPRESSION_PROMPT,
    MEMORY_EXTRACTION_PROMPT,
    MEMORY_RERANK_PROMPT,
    MemoryCompressionAgent,
    MemoryExtractorAgent,
    MemoryRerankerAgent,
    normalize_facts,
)
from .memory_chunking import chunk_chapter
from .memory_config import MemorySettingsStore
from .memory_store import MemoryStore
from .runtime import RuntimeCoordinator
from .vector_index import NumpyVectorIndex


class MemoryPipeline:
    def __init__(
        self,
        project_root: str | Path,
        model: LanguageModel,
        model_cache_dir: str | Path,
        coordinator: RuntimeCoordinator | None = None,
        embedder: EmbeddingProvider | None = None,
        agent_models: dict[str, LanguageModel] | None = None,
        prompt_overrides: dict[str, str] | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.project_id = self.project_root.name
        self.model = model
        self.settings_store = MemorySettingsStore(self.project_root)
        self.settings = self.settings_store.load()
        self.memory_store = MemoryStore(self.project_root)
        self.versions = ChapterVersionStore(self.project_root)
        self.coordinator = coordinator or RuntimeCoordinator()
        self.embedder = embedder or create_embedder(
            self.settings["embedding"], model_cache_dir, allow_download=False
        )
        self.index = NumpyVectorIndex(
            self.project_root / "memory", int(self.settings["embedding"]["dimensions"])
        )
        models = agent_models or {}
        prompts = prompt_overrides or {}
        self.extractor = MemoryExtractorAgent(
            models.get("memory_extract", model),
            prompts.get("memory_extract", MEMORY_EXTRACTION_PROMPT),
        )
        self.compressor = MemoryCompressionAgent(
            models.get("memory_compress", model),
            prompts.get("memory_compress", MEMORY_COMPRESSION_PROMPT),
        )
        self.reranker = MemoryRerankerAgent(
            models.get("memory_rerank", model),
            prompts.get("memory_rerank", MEMORY_RERANK_PROMPT),
        )

    def index_chapter(
        self,
        chapter_number: int,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, Any]:
        final_path = self.versions.final_path(chapter_number)
        if not final_path.exists():
            raise FileNotFoundError("只有作者确认的定稿章节可以建立记忆索引")
        text = final_path.read_text(encoding="utf-8")
        source_hash = content_hash(text)
        current = self.memory_store.chapter_record(chapter_number)
        if current and current.get("status") == "synced" and current.get("source_hash") == source_hash:
            return {"chapter_number": chapter_number, "status": "already_synced"}

        self.memory_store.begin_indexing(chapter_number)
        self.versions.set_status(chapter_number, "extracting_memory")
        try:
            chunks = [
                item.to_dict()
                for item in chunk_chapter(chapter_number, text, self.settings["chunking"])
            ]
            if not chunks:
                raise ValueError("定稿正文无法切分出有效记忆块")
            if progress:
                progress("extracting_memory", 0, len(chunks))
            extraction = self.extractor.run(chapter_number, text, chunks)
            annotations = {
                str(item.get("chunk_id", "")): item
                for item in extraction.get("chunk_annotations", [])
                if isinstance(item, dict)
            }
            for chunk in chunks:
                annotation = annotations.get(chunk["id"], {})
                chunk["importance"] = _unit(annotation.get("importance", 0.5))
                chunk["characters"] = _strings(annotation.get("characters", []))
                chunk["location"] = str(annotation.get("location", ""))
                chunk["timeline"] = str(annotation.get("timeline", ""))
                chunk["keywords"] = _strings(annotation.get("keywords", []))
                chunk["metadata"] = {"source": "author_final", "source_hash": source_hash}

            self.versions.set_status(chapter_number, "embedding")
            batch_size = int(self.settings["indexing"]["batch_size"])
            vectors = []
            for start in range(0, len(chunks), batch_size):
                self.coordinator.wait_for_index_slot(
                    self.project_id,
                    bool(self.settings["indexing"]["pause_during_writing"]),
                )
                batch = chunks[start : start + batch_size]
                vectors.append(
                    self.embedder.embed_documents(
                        [item["text"] for item in batch], batch_size=batch_size
                    )
                )
                if progress:
                    progress("embedding", min(start + len(batch), len(chunks)), len(chunks))
            import numpy as np

            matrix = np.vstack(vectors)
            self.index.delete_prefix(f"CH{chapter_number:03d}-")
            self.index.upsert([item["id"] for item in chunks], matrix)
            facts = normalize_facts(
                chapter_number,
                [item for item in extraction.get("facts", []) if isinstance(item, dict)],
                {item["id"] for item in chunks},
            )
            self.memory_store.replace_chapter(
                chapter_number=chapter_number,
                source_hash=source_hash,
                summary=str(extraction["summary"]),
                word_count=len(text),
                chunks=chunks,
                facts=facts,
                style_observations=(
                    extraction.get("style_observations", [])
                    if self.settings["policy"]["style_learning"]
                    else []
                ),
            )
            self.versions.set_status(chapter_number, "synced")
            checkpoint = self._checkpoint_if_needed(chapter_number)
            return {
                "chapter_number": chapter_number,
                "status": "synced",
                "chunk_count": len(chunks),
                "fact_count": len(facts),
                "source_hash": source_hash,
                "checkpoint": checkpoint,
            }
        except Exception as exc:
            self.memory_store.fail_indexing(chapter_number, str(exc))
            self.versions.set_status(chapter_number, "failed", str(exc))
            raise

    def retrieve(
        self,
        query: str,
        chapter_number: int | None = None,
        characters: list[str] | None = None,
        use_reranker: bool = True,
        access: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        retrieval = self.settings["retrieval"]
        permissions = access or {}
        if permissions.get("enabled") is False:
            return self._trim_context(
                {
                    "query": query,
                    "retrieved_chunks": [],
                    "structured_facts": [],
                    "recent_final_chapters": [],
                    "author_style_observations": [],
                    "candidate_count": 0,
                    "filtered_count": 0,
                },
                int(retrieval["maximum_context_chars"]),
            )

        include_raw = permissions.get("read_raw_passages", True)
        filtered_characters = (
            characters if permissions.get("character_filter", True) else None
        )
        raw: list[dict[str, Any]] = []
        chunks: list[dict[str, Any]] = []
        if include_raw:
            query_vector = self.embedder.embed_query(query)
            raw = self.index.search(query_vector, int(retrieval["vector_top_k"]))
            raw = [
                item
                for item in raw
                if item["score"] >= float(retrieval["minimum_similarity"])
            ]
            chunks = self.memory_store.get_chunks([item["id"] for item in raw])
            score_by_id = {item["id"]: item["score"] for item in raw}
            for chunk in chunks:
                chunk["score"] = score_by_id.get(chunk["id"], 0.0)

        if filtered_characters:
            relevant = [
                item
                for item in chunks
                if not item.get("characters")
                or any(
                    character in item.get("characters", [])
                    for character in filtered_characters
                )
            ]
            if relevant:
                chunks = relevant
        chunks = chunks[: int(retrieval["metadata_filtered_top_k"])]
        ordered_ids = [item["id"] for item in chunks]
        if use_reranker and chunks:
            reranked = self.reranker.run(query, chunks)
            if reranked:
                ordered_ids = reranked + [item for item in ordered_ids if item not in reranked]
        ordered_ids = ordered_ids[: int(retrieval["rerank_top_k"])]
        by_id = {item["id"]: item for item in chunks}
        final_limit = int(retrieval["final_context_chunks"])
        if permissions.get("top_k") is not None:
            final_limit = min(final_limit, max(1, int(permissions["top_k"])))
        final_chunks = [
            by_id[item]
            for item in ordered_ids[:final_limit]
            if item in by_id
        ]
        facts = (
            self.memory_store.query_facts(subjects=filtered_characters, limit=80)
            if permissions.get("read_story_facts", True)
            else []
        )
        if not permissions.get("read_foreshadowing", True):
            facts = [item for item in facts if item.get("fact_type") != "foreshadowing"]
        recent = self._recent_final_chapters(chapter_number) if include_raw else []
        style_observations: list[dict[str, Any]] = []
        policy = self.settings["policy"]
        if policy["style_learning"] and permissions.get("read_author_style", False):
            if len(self.memory_store.indexed_hashes()) >= int(
                policy["style_learning_min_chapters"]
            ):
                style_observations = self.memory_store.style_observations()
        context = {
            "query": query,
            "retrieved_chunks": final_chunks,
            "structured_facts": facts,
            "recent_final_chapters": recent,
            "author_style_observations": style_observations,
            "candidate_count": len(raw),
            "filtered_count": len(chunks),
        }
        return self._trim_context(context, int(retrieval["maximum_context_chars"]))

    def assert_ready_for_chapter(self, chapter_number: int) -> None:
        if not self.settings["policy"]["strict_latest_chapter_sync"] or chapter_number <= 1:
            return
        previous = chapter_number - 1
        final_path = self.versions.final_path(previous)
        if not final_path.exists():
            return
        current_hash = content_hash(final_path.read_text(encoding="utf-8"))
        record = self.memory_store.chapter_record(previous)
        if not record or record.get("status") != "synced" or record.get("source_hash") != current_hash:
            raise RuntimeError(f"第 {previous} 章定稿尚未完成记忆同步，不能生成下一章")

    def overview(self, chapter_count: int) -> dict[str, Any]:
        indexed = self.memory_store.indexed_hashes()
        statuses = self.versions.list_statuses(chapter_count, indexed)
        base = self.memory_store.overview()
        base.update(
            {
                "enabled": True,
                "vector_count": self.index.count(),
                "vector_bytes": self.index.path.stat().st_size if self.index.path.exists() else 0,
                "waiting_index": sum(item["status"] == "waiting_index" for item in statuses),
                "stale_chapters": sum(item["status"] == "stale" for item in statuses),
                "chapter_statuses": statuses,
                "runtime": self.coordinator.status(self.project_id),
            }
        )
        return base

    def rebuild(self, chapter_numbers: list[int] | None = None) -> list[dict[str, Any]]:
        numbers = chapter_numbers or [
            int(path.stem.split("_")[0][2:])
            for path in self.versions.chapters_dir.glob("CH*_final.md")
        ]
        self.index.clear()
        self.memory_store.clear_generated_memory()
        return [self.index_chapter(number) for number in sorted(set(numbers))]

    def _checkpoint_if_needed(self, chapter_number: int) -> dict[str, Any] | None:
        interval = int(self.settings["indexing"]["checkpoint_every_chapters"])
        if chapter_number % interval != 0 or self.memory_store.has_checkpoint(chapter_number):
            return None
        start = max(1, chapter_number - interval + 1)
        summaries = [
            item
            for item in self.memory_store.chapter_summaries()
            if start <= int(item["chapter_number"]) <= chapter_number
        ]
        facts = self.memory_store.query_facts(limit=500)
        compressed = self.compressor.run(start, chapter_number, summaries, facts)
        checkpoint_dir = self.project_root / "memory" / "checkpoints" / f"CH{start:03d}-CH{chapter_number:03d}"
        summary_path = checkpoint_dir / "summary.json"
        _write_json(summary_path, compressed)
        return self.memory_store.create_checkpoint(
            start,
            chapter_number,
            checkpoint_dir,
            summary_path=str(summary_path),
            vector_path=self.index.path,
        )

    def _recent_final_chapters(self, chapter_number: int | None) -> list[dict[str, Any]]:
        limit = int(self.settings["retrieval"]["recent_full_chapters"])
        if limit <= 0:
            return []
        upper = chapter_number - 1 if chapter_number else 10**9
        paths = []
        for path in self.versions.chapters_dir.glob("CH*_final.md"):
            number = int(path.stem.split("_")[0][2:])
            if number <= upper:
                paths.append((number, path))
        return [
            {"chapter_number": number, "text": path.read_text(encoding="utf-8")}
            for number, path in sorted(paths, reverse=True)[:limit][::-1]
        ]

    @staticmethod
    def _trim_context(context: dict[str, Any], maximum_chars: int) -> dict[str, Any]:
        while len(json.dumps(context, ensure_ascii=False)) > maximum_chars:
            if context["retrieved_chunks"]:
                context["retrieved_chunks"].pop()
            elif context["recent_final_chapters"]:
                context["recent_final_chapters"].pop(0)
            elif context["structured_facts"]:
                context["structured_facts"].pop()
            elif context["author_style_observations"]:
                context["author_style_observations"].pop()
            else:
                break
        return context


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _unit(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.5
