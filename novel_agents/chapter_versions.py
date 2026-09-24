from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ChapterVersionStore:
    _locks_guard = threading.Lock()
    _manifest_locks: dict[Path, threading.RLock] = {}

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.chapters_dir = self.project_root / "chapters"
        self.versions_dir = self.chapters_dir / "versions"
        self.manifest_path = self.chapters_dir / "manifest.json"
        with self._locks_guard:
            self._lock = self._manifest_locks.setdefault(
                self.manifest_path, threading.RLock()
            )

    def draft_path(self, chapter_number: int) -> Path:
        return self.chapters_dir / f"CH{chapter_number:03d}_draft.md"

    def author_path(self, chapter_number: int) -> Path:
        return self.chapters_dir / f"CH{chapter_number:03d}_author.md"

    def final_path(self, chapter_number: int) -> Path:
        return self.chapters_dir / f"CH{chapter_number:03d}_final.md"

    def legacy_path(self, chapter_number: int) -> Path:
        return self.chapters_dir / f"CH{chapter_number:03d}.md"

    def write_draft(self, chapter_number: int, text: str) -> Path:
        with self._lock:
            path = self.draft_path(chapter_number)
            self._write_text(path, text.rstrip() + "\n")
            manifest = self._manifest()
            entry = manifest.setdefault("chapters", {}).setdefault(str(chapter_number), {})
            entry.update({"status": "draft", "draft_hash": content_hash(text), "updated_at": _now()})
            self._write_manifest(manifest)
            return path

    def write_author(self, chapter_number: int, text: str) -> Path:
        if not text.strip():
            raise ValueError("作者润色稿不能为空")
        with self._lock:
            path = self.author_path(chapter_number)
            if path.exists():
                existing = path.read_text(encoding="utf-8")
                if content_hash(existing) != content_hash(text):
                    self.versions_dir.mkdir(parents=True, exist_ok=True)
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    archive = self.versions_dir / f"CH{chapter_number:03d}_author_{stamp}.md"
                    shutil.copy2(path, archive)
            self._write_text(path, text.rstrip() + "\n")
            manifest = self._manifest()
            entry = manifest.setdefault("chapters", {}).setdefault(str(chapter_number), {})
            entry.update({"status": "author_edit", "author_hash": content_hash(text), "updated_at": _now()})
            self._write_manifest(manifest)
            return path

    def finalize(self, chapter_number: int, text: str | None = None) -> dict[str, Any]:
        with self._lock:
            self.chapters_dir.mkdir(parents=True, exist_ok=True)
            if text is None:
                source = self.author_path(chapter_number)
                if not source.exists():
                    source = self.draft_path(chapter_number)
                if not source.exists():
                    source = self.legacy_path(chapter_number)
                if not source.exists():
                    raise FileNotFoundError("没有可定稿的章节内容")
                text = source.read_text(encoding="utf-8")
            if not text.strip():
                raise ValueError("定稿正文不能为空")

            final_path = self.final_path(chapter_number)
            if final_path.exists():
                self.versions_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                archive = self.versions_dir / f"CH{chapter_number:03d}_final_{stamp}.md"
                shutil.copy2(final_path, archive)
            self._write_text(final_path, text.rstrip() + "\n")
            digest = content_hash(text)
            manifest = self._manifest()
            entry = manifest.setdefault("chapters", {}).setdefault(str(chapter_number), {})
            entry.update(
                {
                    "status": "waiting_index",
                    "final_hash": digest,
                    "finalized_at": _now(),
                    "updated_at": _now(),
                }
            )
            self._write_manifest(manifest)
            return {"path": str(final_path), "final_hash": digest, "status": "waiting_index"}

    def set_status(self, chapter_number: int, status: str, detail: str = "") -> None:
        with self._lock:
            manifest = self._manifest()
            entry = manifest.setdefault("chapters", {}).setdefault(str(chapter_number), {})
            entry.update({"status": status, "detail": detail, "updated_at": _now()})
            self._write_manifest(manifest)

    def get_text(self, chapter_number: int, preferred: str = "best") -> str:
        candidates: list[Path]
        if preferred == "final":
            candidates = [self.final_path(chapter_number)]
        elif preferred == "author":
            candidates = [self.author_path(chapter_number)]
        elif preferred == "draft":
            candidates = [self.draft_path(chapter_number), self.legacy_path(chapter_number)]
        else:
            candidates = [
                self.final_path(chapter_number),
                self.author_path(chapter_number),
                self.draft_path(chapter_number),
                self.legacy_path(chapter_number),
            ]
        for path in candidates:
            if path.exists():
                return path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"第 {chapter_number} 章不存在")

    def status(self, chapter_number: int, indexed_hash: str = "") -> dict[str, Any]:
        with self._lock:
            manifest = self._manifest()
            entry = dict(manifest.get("chapters", {}).get(str(chapter_number), {}))
            paths = {
                "draft": self.draft_path(chapter_number),
                "author": self.author_path(chapter_number),
                "final": self.final_path(chapter_number),
                "legacy": self.legacy_path(chapter_number),
            }
            entry["chapter_number"] = chapter_number
            entry["files"] = {key: path.exists() for key, path in paths.items()}
            entry["indexed_hash"] = indexed_hash
            final_path = paths["final"]
            current_hash = content_hash(final_path.read_text(encoding="utf-8")) if final_path.exists() else ""
            entry["current_final_hash"] = current_hash
            processing = {"indexing", "extracting_memory", "embedding", "failed"}
            if final_path.exists():
                if entry.get("status") in processing:
                    pass
                elif not indexed_hash:
                    entry["status"] = "waiting_index"
                elif indexed_hash != current_hash:
                    entry["status"] = "stale"
                else:
                    entry["status"] = "synced"
            elif paths["author"].exists():
                entry["status"] = "author_edit"
            elif paths["draft"].exists() or paths["legacy"].exists():
                entry["status"] = "draft"
            else:
                entry["status"] = "empty"
            return entry

    def list_statuses(self, chapter_count: int, indexed_hashes: dict[int, str] | None = None) -> list[dict[str, Any]]:
        hashes = indexed_hashes or {}
        return [self.status(number, hashes.get(number, "")) for number in range(1, chapter_count + 1)]

    def _manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"version": 1, "chapters": {}}
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {"version": 1, "chapters": {}}

    def _write_manifest(self, data: dict[str, Any]) -> None:
        self._write_text(
            self.manifest_path,
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        )

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise


def content_hash(text: str) -> str:
    normalized = text.replace("\r\n", "\n").strip().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
