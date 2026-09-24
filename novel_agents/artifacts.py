from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chapter_versions import ChapterVersionStore
from .models import ProjectBrief


class ArtifactStore:
    """Persistent, atomic storage for all agent handoffs."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.artifacts_dir = self.root / "artifacts"
        self.chapter_plans_dir = self.artifacts_dir / "chapter_plans"
        self.relationship_beats_dir = self.artifacts_dir / "relationship_beats"
        self.chapters_dir = self.root / "chapters"
        self.chapter_versions = ChapterVersionStore(self.root)
        self._lock = threading.RLock()

    def initialize(self, brief: ProjectBrief, overwrite: bool = False) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        project_path = self.root / "project.json"
        if project_path.exists() and not overwrite:
            raise FileExistsError(f"Project already exists: {project_path}")
        if overwrite:
            for generated_dir in (self.artifacts_dir, self.chapters_dir):
                target = generated_dir.resolve()
                if target.parent != self.root:
                    raise ValueError(f"Unsafe generated directory: {target}")
                if target.exists():
                    shutil.rmtree(target)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.chapter_plans_dir.mkdir(parents=True, exist_ok=True)
        self.relationship_beats_dir.mkdir(parents=True, exist_ok=True)
        self.chapters_dir.mkdir(parents=True, exist_ok=True)
        self._write_json_path(project_path, brief.to_dict())
        self._write_json_path(
            self.root / "state.json",
            {"version": 1, "stages": {}, "updated_at": self._now()},
        )

    def load_brief(self) -> ProjectBrief:
        return ProjectBrief.from_dict(self._read_json_path(self.root / "project.json"))

    def update_brief(self, brief: ProjectBrief) -> None:
        with self._lock:
            self._write_json_path(self.root / "project.json", brief.to_dict())
            state_path = self.root / "state.json"
            state = (
                self._read_json_path(state_path)
                if state_path.exists()
                else {"version": 1, "stages": {}}
            )
            state["brief_updated_at"] = self._now()
            state["updated_at"] = self._now()
            self._write_json_path(state_path, state)

    def highest_material_chapter(self) -> int:
        highest = 0
        for folder in (self.chapter_plans_dir, self.chapters_dir):
            if not folder.exists():
                continue
            for path in folder.iterdir():
                match = re.match(r"CH(\d+)", path.name, flags=re.IGNORECASE)
                if match:
                    highest = max(highest, int(match.group(1)))
        return highest

    def artifact_path(self, name: str) -> Path:
        return self.artifacts_dir / f"{name}.json"

    def has_artifact(self, name: str) -> bool:
        return self.artifact_path(name).exists()

    def read_artifact(self, name: str) -> dict[str, Any]:
        return self._read_json_path(self.artifact_path(name))

    def write_artifact(self, name: str, data: dict[str, Any]) -> Path:
        path = self.artifact_path(name)
        self._write_json_path(path, data)
        return path

    def resolve_output_path(
        self, relative_path: str, *, must_exist: bool = True
    ) -> Path:
        normalized = str(relative_path).strip().replace("\\", "/")
        if not normalized:
            raise ValueError("产物路径不能为空")
        candidate = (self.root / normalized).resolve()
        allowed_roots = (self.artifacts_dir.resolve(), self.chapters_dir.resolve())
        if not any(candidate.is_relative_to(folder) for folder in allowed_roots):
            raise ValueError("产物只能位于 artifacts 或 chapters 目录")
        if candidate == (self.chapters_dir / "manifest.json").resolve():
            raise ValueError("章节清单是内部文件，不能作为产物修改")
        if candidate.suffix.lower() not in {".json", ".md", ".txt"}:
            raise ValueError("仅支持 JSON、Markdown 和 TXT 产物")
        if must_exist and not candidate.is_file():
            raise FileNotFoundError("产物不存在")
        return candidate

    def read_output(self, relative_path: str) -> str:
        path = self.resolve_output_path(relative_path)
        return path.read_text(encoding="utf-8")

    def write_output(self, relative_path: str, content: str) -> Path:
        path = self.resolve_output_path(relative_path)
        if path.suffix.lower() == ".json":
            data = json.loads(content)
            if not isinstance(data, dict):
                raise ValueError("JSON 产物必须是一个对象")
            self._write_json_path(path, data)
        else:
            self._write_text_path(path, content.replace("\r\n", "\n"))
        return path

    def delete_output(self, relative_path: str) -> Path:
        path = self.resolve_output_path(relative_path)
        with self._lock:
            path.unlink()
        return path

    def chapter_plan_path(self, chapter_number: int) -> Path:
        return self.chapter_plans_dir / f"CH{chapter_number:03d}.json"

    def has_chapter_plan(self, chapter_number: int) -> bool:
        return self.chapter_plan_path(chapter_number).exists()

    def read_chapter_plan(self, chapter_number: int) -> dict[str, Any]:
        return self._read_json_path(self.chapter_plan_path(chapter_number))

    def write_chapter_plan(self, chapter_number: int, data: dict[str, Any]) -> Path:
        path = self.chapter_plan_path(chapter_number)
        self._write_json_path(path, data)
        return path

    def relationship_beat_path(self, chapter_number: int) -> Path:
        return self.relationship_beats_dir / f"CH{chapter_number:03d}.json"

    def has_relationship_beat(self, chapter_number: int) -> bool:
        return self.relationship_beat_path(chapter_number).exists()

    def read_relationship_beat(self, chapter_number: int) -> dict[str, Any]:
        return self._read_json_path(self.relationship_beat_path(chapter_number))

    def write_relationship_beat(
        self, chapter_number: int, data: dict[str, Any]
    ) -> Path:
        path = self.relationship_beat_path(chapter_number)
        self._write_json_path(path, data)
        return path

    def chapter_path(self, chapter_number: int) -> Path:
        return self.chapter_versions.draft_path(chapter_number)

    def has_chapter(self, chapter_number: int) -> bool:
        return any(
            path.exists()
            for path in (
                self.chapter_versions.draft_path(chapter_number),
                self.chapter_versions.author_path(chapter_number),
                self.chapter_versions.final_path(chapter_number),
                self.chapter_versions.legacy_path(chapter_number),
            )
        )

    def read_chapter(self, chapter_number: int) -> str:
        return self.chapter_versions.get_text(chapter_number, preferred="best")

    def write_chapter(self, chapter_number: int, text: str) -> Path:
        return self.chapter_versions.write_draft(chapter_number, text)

    def update_stage(self, stage: str, status: str, detail: str = "") -> None:
        with self._lock:
            state_path = self.root / "state.json"
            state = (
                self._read_json_path(state_path)
                if state_path.exists()
                else {"version": 1, "stages": {}}
            )
            state.setdefault("stages", {})[stage] = {
                "status": status,
                "detail": detail,
                "updated_at": self._now(),
            }
            state["updated_at"] = self._now()
            self._write_json_path(state_path, state)

    def read_state(self) -> dict[str, Any]:
        path = self.root / "state.json"
        if not path.exists():
            return {"version": 1, "stages": {}}
        return self._read_json_path(path)

    def _write_json_path(self, path: Path, data: dict[str, Any]) -> None:
        text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        self._write_text_path(path, text)

    def _write_text_path(self, path: Path, text: str) -> None:
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
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

    @staticmethod
    def _read_json_path(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a JSON object in {path}")
        return data

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
