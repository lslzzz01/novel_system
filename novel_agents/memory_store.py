from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chapter_memory (
    chapter_number INTEGER PRIMARY KEY,
    source_hash TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'waiting_index',
    summary TEXT NOT NULL DEFAULT '',
    word_count INTEGER NOT NULL DEFAULT 0,
    indexed_at TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS memory_chunks (
    id TEXT PRIMARY KEY,
    chapter_number INTEGER NOT NULL,
    scene_index INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text_hash TEXT NOT NULL,
    text TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    characters_json TEXT NOT NULL DEFAULT '[]',
    location TEXT NOT NULL DEFAULT '',
    timeline TEXT NOT NULL DEFAULT '',
    keywords_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_chunks_chapter ON memory_chunks(chapter_number);
CREATE INDEX IF NOT EXISTS idx_chunks_location ON memory_chunks(location);

CREATE TABLE IF NOT EXISTS memory_facts (
    id TEXT PRIMARY KEY,
    chapter_number INTEGER NOT NULL,
    fact_type TEXT NOT NULL,
    subject TEXT NOT NULL DEFAULT '',
    predicate TEXT NOT NULL DEFAULT '',
    object_value TEXT NOT NULL DEFAULT '',
    timeline TEXT NOT NULL DEFAULT '',
    importance REAL NOT NULL DEFAULT 0.5,
    confidence REAL NOT NULL DEFAULT 1.0,
    locked INTEGER NOT NULL DEFAULT 0,
    active INTEGER NOT NULL DEFAULT 1,
    source_chunk_id TEXT NOT NULL DEFAULT '',
    source_quote TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_facts_chapter ON memory_facts(chapter_number);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON memory_facts(subject);
CREATE INDEX IF NOT EXISTS idx_facts_type ON memory_facts(fact_type);

CREATE TABLE IF NOT EXISTS chapter_style_observations (
    chapter_number INTEGER PRIMARY KEY,
    observations_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS memory_checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_chapter INTEGER NOT NULL,
    end_chapter INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    summary_path TEXT NOT NULL DEFAULT '',
    database_path TEXT NOT NULL DEFAULT '',
    vector_path TEXT NOT NULL DEFAULT ''
);
"""


class _ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


class MemoryStore:
    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.memory_dir = self.project_root / "memory"
        self.path = self.memory_dir / "memory.db"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, factory=_ClosingConnection)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)
            connection.execute(
                "INSERT OR REPLACE INTO memory_meta(key, value) VALUES('schema_version', '1')"
            )

    def begin_indexing(self, chapter_number: int) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO chapter_memory(chapter_number, status, error)
                   VALUES(?, 'indexing', '')
                   ON CONFLICT(chapter_number) DO UPDATE SET status='indexing', error=''""",
                (chapter_number,),
            )

    def fail_indexing(self, chapter_number: int, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO chapter_memory(chapter_number, status, error)
                   VALUES(?, 'failed', ?)
                   ON CONFLICT(chapter_number) DO UPDATE SET status='failed', error=excluded.error""",
                (chapter_number, error[:2000]),
            )

    def replace_chapter(
        self,
        chapter_number: int,
        source_hash: str,
        summary: str,
        word_count: int,
        chunks: Iterable[dict[str, Any]],
        facts: Iterable[dict[str, Any]],
        style_observations: Iterable[Any] = (),
    ) -> None:
        now = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM memory_chunks WHERE chapter_number=?", (chapter_number,)
            )
            connection.execute(
                "UPDATE memory_facts SET active=0 WHERE chapter_number=?", (chapter_number,)
            )
            connection.execute(
                "DELETE FROM chapter_style_observations WHERE chapter_number=?",
                (chapter_number,),
            )
            for chunk in chunks:
                connection.execute(
                    """INSERT INTO memory_chunks(
                           id, chapter_number, scene_index, chunk_index, text_hash, text,
                           importance, characters_json, location, timeline, keywords_json,
                           metadata_json
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        chunk["id"],
                        chapter_number,
                        int(chunk.get("scene_index", 1)),
                        int(chunk.get("chunk_index", 1)),
                        chunk["text_hash"],
                        chunk["text"],
                        float(chunk.get("importance", 0.5)),
                        json.dumps(chunk.get("characters", []), ensure_ascii=False),
                        str(chunk.get("location", "")),
                        str(chunk.get("timeline", "")),
                        json.dumps(chunk.get("keywords", []), ensure_ascii=False),
                        json.dumps(chunk.get("metadata", {}), ensure_ascii=False),
                    ),
                )
            for fact in facts:
                connection.execute(
                    """INSERT OR REPLACE INTO memory_facts(
                           id, chapter_number, fact_type, subject, predicate, object_value,
                           timeline, importance, confidence, locked, active, source_chunk_id,
                           source_quote, payload_json
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)""",
                    (
                        fact["id"],
                        chapter_number,
                        str(fact.get("fact_type", "event")),
                        str(fact.get("subject", "")),
                        str(fact.get("predicate", "")),
                        str(fact.get("object", "")),
                        str(fact.get("timeline", "")),
                        float(fact.get("importance", 0.5)),
                        float(fact.get("confidence", 1.0)),
                        1 if fact.get("locked", False) else 0,
                        str(fact.get("source_chunk_id", "")),
                        str(fact.get("source_quote", "")),
                        json.dumps(fact.get("payload", {}), ensure_ascii=False),
                    ),
                )
            observations = [item for item in style_observations]
            if observations:
                connection.execute(
                    """INSERT INTO chapter_style_observations(
                           chapter_number, observations_json
                       ) VALUES (?, ?)""",
                    (chapter_number, json.dumps(observations, ensure_ascii=False)),
                )
            connection.execute(
                """INSERT INTO chapter_memory(
                       chapter_number, source_hash, status, summary, word_count, indexed_at, error
                   ) VALUES (?, ?, 'synced', ?, ?, ?, '')
                   ON CONFLICT(chapter_number) DO UPDATE SET
                       source_hash=excluded.source_hash,
                       status='synced',
                       summary=excluded.summary,
                       word_count=excluded.word_count,
                       indexed_at=excluded.indexed_at,
                       error=''""",
                (chapter_number, source_hash, summary, word_count, now),
            )
            connection.commit()

    def delete_chapter(self, chapter_number: int) -> list[str]:
        """Remove live memory derived from one deleted final chapter."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT id FROM memory_chunks WHERE chapter_number=?",
                (chapter_number,),
            ).fetchall()
            chunk_ids = [str(row["id"]) for row in rows]
            connection.execute(
                "DELETE FROM memory_chunks WHERE chapter_number=?", (chapter_number,)
            )
            connection.execute(
                "DELETE FROM memory_facts WHERE chapter_number=?", (chapter_number,)
            )
            connection.execute(
                "DELETE FROM chapter_style_observations WHERE chapter_number=?",
                (chapter_number,),
            )
            connection.execute(
                "DELETE FROM chapter_memory WHERE chapter_number=?", (chapter_number,)
            )
            connection.commit()
        return chunk_ids

    def indexed_hashes(self) -> dict[int, str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT chapter_number, source_hash FROM chapter_memory WHERE status='synced'"
            ).fetchall()
        return {int(row["chapter_number"]): str(row["source_hash"]) for row in rows}

    def chapter_record(self, chapter_number: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM chapter_memory WHERE chapter_number=?", (chapter_number,)
            ).fetchone()
        return dict(row) if row else None

    def chapter_summaries(self, limit: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT chapter_number, summary, source_hash, indexed_at FROM chapter_memory WHERE status='synced' ORDER BY chapter_number"
        parameters: tuple[Any, ...] = ()
        if limit is not None:
            query = "SELECT * FROM (" + query + " DESC LIMIT ?) ORDER BY chapter_number"
            parameters = (int(limit),)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def get_chunks(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM memory_chunks WHERE id IN ({placeholders})", ids
            ).fetchall()
        by_id = {str(row["id"]): self._decode_chunk(row) for row in rows}
        return [by_id[item] for item in ids if item in by_id]

    def query_facts(
        self,
        subjects: list[str] | None = None,
        fact_types: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses = ["active=1"]
        values: list[Any] = []
        if subjects:
            subject_clauses = []
            for subject in subjects:
                subject_clauses.append("subject LIKE ?")
                values.append(f"%{subject}%")
            clauses.append("(" + " OR ".join(subject_clauses) + ")")
        if fact_types:
            placeholders = ",".join("?" for _ in fact_types)
            clauses.append(f"fact_type IN ({placeholders})")
            values.extend(fact_types)
        values.append(int(limit))
        query = (
            "SELECT * FROM memory_facts WHERE "
            + " AND ".join(clauses)
            + " ORDER BY locked DESC, importance DESC, chapter_number DESC LIMIT ?"
        )
        with self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return [self._decode_fact(row) for row in rows]

    def style_observations(self, limit_chapters: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT chapter_number, observations_json
                   FROM chapter_style_observations
                   ORDER BY chapter_number DESC LIMIT ?""",
                (max(1, int(limit_chapters)),),
            ).fetchall()
        observations: list[dict[str, Any]] = []
        for row in reversed(rows):
            items = json.loads(str(row["observations_json"]))
            if not isinstance(items, list):
                continue
            for item in items:
                observations.append(
                    {
                        "chapter_number": int(row["chapter_number"]),
                        "observation": item,
                    }
                )
        return observations

    def overview(self) -> dict[str, Any]:
        with self._connect() as connection:
            chapters = connection.execute(
                "SELECT status, COUNT(*) AS count FROM chapter_memory GROUP BY status"
            ).fetchall()
            chunk_count = connection.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0]
            fact_count = connection.execute(
                "SELECT COUNT(*) FROM memory_facts WHERE active=1"
            ).fetchone()[0]
            latest = connection.execute(
                "SELECT MAX(indexed_at) FROM chapter_memory WHERE status='synced'"
            ).fetchone()[0]
            checkpoints = connection.execute(
                "SELECT COUNT(*) FROM memory_checkpoints"
            ).fetchone()[0]
        counts = {str(row["status"]): int(row["count"]) for row in chapters}
        return {
            "chapters": counts,
            "indexed_chapters": counts.get("synced", 0),
            "failed_chapters": counts.get("failed", 0),
            "chunk_count": int(chunk_count),
            "fact_count": int(fact_count),
            "checkpoint_count": int(checkpoints),
            "last_indexed_at": latest or "",
            "database_bytes": self.path.stat().st_size if self.path.exists() else 0,
        }

    def all_chunk_ids(self) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute("SELECT id FROM memory_chunks").fetchall()
        return [str(row["id"]) for row in rows]

    def clear_generated_memory(self) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM memory_chunks")
            connection.execute("DELETE FROM memory_facts WHERE locked=0")
            connection.execute("DELETE FROM chapter_style_observations")
            connection.execute("DELETE FROM chapter_memory")
            connection.execute("DELETE FROM memory_checkpoints")
            connection.commit()

    def has_checkpoint(self, end_chapter: int) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM memory_checkpoints WHERE end_chapter=? LIMIT 1",
                (end_chapter,),
            ).fetchone()
        return bool(row)

    def create_checkpoint(
        self,
        start_chapter: int,
        end_chapter: int,
        checkpoint_dir: Path,
        summary_path: str = "",
        vector_path: Path | None = None,
    ) -> dict[str, Any]:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        database_copy = checkpoint_dir / "memory.db"
        source = self._connect()
        destination = sqlite3.connect(database_copy)
        try:
            source.backup(destination)
        finally:
            destination.close()
            source.close()
        vector_copy = ""
        if vector_path and vector_path.exists():
            target = checkpoint_dir / vector_path.name
            shutil.copy2(vector_path, target)
            vector_copy = str(target)
        record = {
            "start_chapter": start_chapter,
            "end_chapter": end_chapter,
            "created_at": _now(),
            "summary_path": summary_path,
            "database_path": str(database_copy),
            "vector_path": vector_copy,
        }
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO memory_checkpoints(
                       start_chapter, end_chapter, created_at, summary_path,
                       database_path, vector_path
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                tuple(record.values()),
            )
        return record

    @staticmethod
    def _decode_chunk(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["characters"] = json.loads(data.pop("characters_json"))
        data["keywords"] = json.loads(data.pop("keywords_json"))
        data["metadata"] = json.loads(data.pop("metadata_json"))
        return data

    @staticmethod
    def _decode_fact(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["object"] = data.pop("object_value")
        data["locked"] = bool(data["locked"])
        data["active"] = bool(data["active"])
        data["payload"] = json.loads(data.pop("payload_json"))
        return data


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
