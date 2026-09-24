from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import ProjectBrief, normalize_writing_genre


SECTION_ORDER = (
    ("core_relationship", "核心关系"),
    ("protagonist", "主角"),
    ("counterpart", "对方"),
    ("opening", "开局情境"),
    ("midgame", "中段升级"),
    ("ending", "结局方向与禁忌"),
    ("tone_scale", "氛围与尺度"),
    ("narration", "叙事"),
    ("freeform", "补充"),
)

DEFAULT_IDEA: dict[str, Any] = {
    "title": "",
    "sections": {key: "" for key, _ in SECTION_ORDER},
    "premise": "",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compose_premise(sections: dict[str, Any], free_premise: str = "") -> str:
    blocks: list[str] = []
    for key, label in SECTION_ORDER:
        text = str(sections.get(key, "") or "").strip()
        if not text:
            continue
        blocks.append(f"【{label}】\n{text}")
    free = str(free_premise or "").strip()
    if free and free not in "\n\n".join(blocks):
        # Avoid duplicating if free text is already the composed premise.
        if not any(free.startswith(f"【{label}】") for _, label in SECTION_ORDER):
            blocks.append(f"【补充】\n{free}" if blocks else free)
    return "\n\n".join(blocks).strip()


def idea_premise_text(idea: dict[str, Any]) -> str:
    sections = idea.get("sections") if isinstance(idea.get("sections"), dict) else {}
    return str(idea.get("premise", "")).strip() or compose_premise(sections)


def idea_to_project_values(idea: dict[str, Any]) -> dict[str, Any]:
    """Build project payload from an idea.

    Story ideas are premise-first. Project fields may be filled later in the
    create-project dialog; only require a non-empty premise here when used as a
    helper. Full ProjectBrief validation happens at project creation time.
    """
    sections = idea.get("sections") if isinstance(idea.get("sections"), dict) else {}
    premise = idea_premise_text(idea)
    if not premise:
        raise ValueError("故事设想不能为空，请先填写至少一个分区")
    constraints = idea.get("constraints", [])
    if isinstance(constraints, str):
        constraints = [
            line.strip() for line in constraints.splitlines() if line.strip()
        ]
    elif not isinstance(constraints, list):
        constraints = []
    else:
        constraints = [str(item).strip() for item in constraints if str(item).strip()]
    title = str(idea.get("title", "")).strip()
    if not title:
        title = premise.splitlines()[0][:24] if premise else "未命名项目"
    values = {
        "title": title,
        "premise": premise,
        "genre": str(idea.get("genre", "")).strip() or "校园",
        "writing_genre": normalize_writing_genre(
            str(idea.get("writing_genre", "")),
            " ".join(
                (
                    title,
                    str(idea.get("genre", "")),
                    premise,
                )
            ),
        ),
        "target_readers": str(idea.get("target_readers", "")).strip() or "青年读者",
        "style": str(idea.get("style", "")).strip()
        or "自然、克制、注重人物互动与身体细节",
        "chapter_count": int(idea.get("chapter_count", 100) or 100),
        "chapter_word_count": int(idea.get("chapter_word_count", 4000) or 4000),
        "language": str(idea.get("language", "")).strip() or "Simplified Chinese",
        "constraints": constraints,
        "memory_enabled": bool(idea.get("memory_enabled", True)),
    }
    return values


class StoryIdeaStore:
    """File-backed story idea drafts under .novel_agents/story_ideas/."""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()
        self.root = self.workspace / ".novel_agents" / "story_ideas"
        self._lock = threading.RLock()

    def list_ideas(self) -> list[dict[str, Any]]:
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            items: list[dict[str, Any]] = []
            for path in self.root.glob("*.json"):
                try:
                    data = self._read(path)
                except (OSError, json.JSONDecodeError, ValueError):
                    continue
                items.append(self._public_summary(data))
            return sorted(
                items, key=lambda item: item.get("updated_at", ""), reverse=True
            )

    def get(self, idea_id: str) -> dict[str, Any]:
        with self._lock:
            path = self._path(idea_id)
            if not path.exists():
                raise KeyError(f"未找到故事设想：{idea_id}")
            return self._public(self._read(path))

    def create(self, values: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            idea_id = "idea-" + uuid.uuid4().hex[:10]
            now = _now()
            data = self._normalize(values or {}, idea_id=idea_id, created_at=now)
            data["created_at"] = now
            data["updated_at"] = now
            self._write(self._path(idea_id), data)
            return self._public(data)

    def update(self, idea_id: str, values: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            path = self._path(idea_id)
            if not path.exists():
                raise KeyError(f"未找到故事设想：{idea_id}")
            current = self._read(path)
            data = self._normalize(
                {**current, **values},
                idea_id=idea_id,
                created_at=str(current.get("created_at") or _now()),
            )
            data["created_at"] = current.get("created_at") or data["created_at"]
            data["updated_at"] = _now()
            if current.get("project_id"):
                data["project_id"] = current["project_id"]
            self._write(path, data)
            return self._public(data)

    def delete(self, idea_id: str) -> dict[str, Any]:
        with self._lock:
            path = self._path(idea_id)
            if not path.exists():
                raise KeyError(f"未找到故事设想：{idea_id}")
            path.unlink()
            return {"id": idea_id, "deleted": True}

    def mark_created_project(self, idea_id: str, project_id: str) -> dict[str, Any]:
        with self._lock:
            path = self._path(idea_id)
            if not path.exists():
                raise KeyError(f"未找到故事设想：{idea_id}")
            data = self._read(path)
            data["project_id"] = project_id
            data["updated_at"] = _now()
            self._write(path, data)
            return self._public(data)

    def _normalize(
        self,
        values: dict[str, Any],
        *,
        idea_id: str,
        created_at: str,
    ) -> dict[str, Any]:
        sections_in = values.get("sections") if isinstance(values.get("sections"), dict) else {}
        sections = {
            key: str(sections_in.get(key, values.get(key, "")) or "").strip()
            for key, _ in SECTION_ORDER
        }
        # Allow top-level section keys as convenience.
        for key, _ in SECTION_ORDER:
            if key in values and key != "sections":
                sections[key] = str(values.get(key) or "").strip()

        premise = str(values.get("premise", "") or "").strip()
        if not premise:
            premise = compose_premise(sections)
        title = str(values.get("title", "") or "").strip()
        if not title and premise:
            title = premise.splitlines()[0][:24]

        return {
            "id": idea_id,
            "title": title,
            "sections": sections,
            "premise": premise,
            "composed_premise": compose_premise(sections),
            "created_at": created_at,
            "updated_at": str(values.get("updated_at") or created_at),
            "project_id": str(values.get("project_id", "") or "").strip(),
        }

    def _public(self, data: dict[str, Any]) -> dict[str, Any]:
        sections = data.get("sections") if isinstance(data.get("sections"), dict) else {}
        composed = compose_premise(sections)
        premise = str(data.get("premise") or composed)
        return {
            "id": data["id"],
            "title": data.get("title", ""),
            "sections": {key: str(sections.get(key, "") or "") for key, _ in SECTION_ORDER},
            "section_labels": {key: label for key, label in SECTION_ORDER},
            "premise": premise,
            "composed_premise": composed,
            "created_at": data.get("created_at", ""),
            "updated_at": data.get("updated_at", ""),
            "project_id": data.get("project_id", ""),
        }

    def _public_summary(self, data: dict[str, Any]) -> dict[str, Any]:
        title = str(data.get("title") or "").strip() or "未命名设想"
        premise = str(data.get("premise") or data.get("composed_premise") or "")
        return {
            "id": data["id"],
            "title": title,
            "updated_at": data.get("updated_at", ""),
            "project_id": data.get("project_id", ""),
            "snippet": re.sub(r"\s+", " ", premise)[:80],
        }

    def _path(self, idea_id: str) -> Path:
        clean = str(idea_id or "").strip()
        if not re.fullmatch(r"idea-[a-z0-9]{6,16}", clean):
            raise ValueError("无效的故事设想 ID")
        return self.root / f"{clean}.json"

    def _read(self, path: Path) -> dict[str, Any]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("故事设想文件格式无效")
        return data

    def _write(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=".idea.", suffix=".tmp", dir=path.parent
        )
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
