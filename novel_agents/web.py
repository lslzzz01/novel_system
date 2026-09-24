from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import parse_qs, unquote, urlparse

from .artifacts import ArtifactStore
from .agent_profiles import AgentProfileStore
from .chapter_versions import ChapterVersionStore, content_hash
from .embedding import HashEmbeddingProvider, LocalSentenceTransformerEmbedder
from .llm import MockLanguageModel
from .memory_config import MemorySettingsStore
from .memory_pipeline import MemoryPipeline
from .memory_store import MemoryStore
from .models import ProjectBrief
from .prompt_store import PromptStore
from .relationship_agents import RelationshipAuditor, RelationshipMemoryAgent
from .relationship_store import RelationshipStateStore
from .runtime import RuntimeCoordinator
from .settings import ModelSettingsStore, load_workspace_env
from .style_agents import StyleAuditor, StyleEditor
from .style_review import StyleReviewService, StyleReviewSettingsStore
from .vector_index import NumpyVectorIndex
from .story_ideas import StoryIdeaStore, idea_to_project_values
from .workflow import NovelWorkflow


def count_generated_chars(text: str) -> int:
    """Count visible characters for generation logs (whitespace excluded)."""
    return sum(1 for character in str(text or "") if not character.isspace())


def format_char_count(count: int) -> str:
    return f"{max(0, int(count))} 字"


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(slots=True)
class JobRecord:
    id: str
    project_id: str
    action: str
    status: str
    created_at: str
    started_at: str = ""
    completed_at: str = ""
    detail: str = ""
    error: str = ""
    kind: str = "generation"
    phase: str = ""
    progress_current: int = 0
    progress_total: int = 0
    exclusive: bool = False
    resource_key: str = ""
    model_snapshot: dict[str, Any] = field(default_factory=dict)


class JobManager:
    _history_locks_guard = threading.Lock()
    _history_locks: dict[Path, threading.RLock] = {}

    def __init__(
        self,
        max_workers: int = 2,
        default_project_limit: int = 1,
        history_path: str | Path | None = None,
    ) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.default_project_limit = default_project_limit
        self.jobs: dict[str, JobRecord] = {}
        self.active_counts: dict[str, int] = {}
        self.exclusive_projects: set[str] = set()
        self.active_resources: set[str] = set()
        self.lock = threading.RLock()
        self.history_path = Path(history_path).resolve() if history_path else None
        self.history_lock: threading.RLock | None = None
        if self.history_path is not None:
            with self._history_locks_guard:
                self.history_lock = self._history_locks.setdefault(
                    self.history_path, threading.RLock()
                )

    def submit(
        self,
        project_id: str,
        action: str,
        work: Callable[[JobRecord], str],
        kind: str = "generation",
        project_limit: int | Callable[[], int] | None = None,
        exclusive: bool = False,
        resource_key: str = "",
    ) -> dict[str, Any]:
        with self.lock:
            limit = project_limit() if callable(project_limit) else project_limit
            limit = int(limit or self.default_project_limit)
            if project_id in self.exclusive_projects or (
                exclusive and self.active_counts.get(project_id, 0) > 0
            ):
                raise ApiError(HTTPStatus.CONFLICT, "该项目已有互斥任务正在运行")
            if self.active_counts.get(project_id, 0) >= limit:
                raise ApiError(HTTPStatus.CONFLICT, "该项目已有任务达到并发上限")
            if resource_key and resource_key in self.active_resources:
                raise ApiError(HTTPStatus.CONFLICT, "相同资源的任务已经在运行")
            job_id = uuid.uuid4().hex[:12]
            record = JobRecord(
                id=job_id,
                project_id=project_id,
                action=action,
                status="queued",
                created_at=_now(),
                kind=kind,
                exclusive=exclusive,
                resource_key=resource_key,
            )
            self.jobs[job_id] = record
            self.active_counts[project_id] = self.active_counts.get(project_id, 0) + 1
            if exclusive:
                self.exclusive_projects.add(project_id)
            if resource_key:
                self.active_resources.add(resource_key)
            self.executor.submit(self._run, record, work)
            return asdict(record)

    def get(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            record = self.jobs.get(job_id)
            if record:
                return asdict(record)
        for item in reversed(self._load_history()):
            if item.get("id") == job_id:
                return item
        raise ApiError(HTTPStatus.NOT_FOUND, "任务不存在")

    def recent(self, project_id: str = "", limit: int = 50) -> list[dict[str, Any]]:
        items = self._load_history()
        with self.lock:
            items.extend(asdict(record) for record in self.jobs.values())
        unique: dict[str, dict[str, Any]] = {}
        for item in items:
            if project_id and item.get("project_id") != project_id:
                continue
            unique[str(item.get("id", ""))] = item
        ordered = sorted(
            unique.values(), key=lambda item: str(item.get("created_at", "")), reverse=True
        )
        return ordered[: max(1, min(int(limit), 200))]

    def has_active(self, project_id: str) -> bool:
        with self.lock:
            return self.active_counts.get(project_id, 0) > 0

    def update_progress(self, record: JobRecord, phase: str, current: int, total: int) -> None:
        with self.lock:
            record.phase = phase
            record.progress_current = int(current)
            record.progress_total = int(total)

    def _run(self, record: JobRecord, work: Callable[[JobRecord], str]) -> None:
        with self.lock:
            record.status = "running"
            record.started_at = _now()
        status = "completed"
        detail = ""
        error = ""
        try:
            detail = work(record)
        except Exception as exc:
            status = "failed"
            error = str(exc)
        # 先释放占用，再发布终态：客户端（以及正文质检这类“轮询到完成就立刻提交
        # 下一步”的串联流程）一旦看到 completed 就可能马上提交同一项目的新任务，
        # 若此时占用尚未归还，就会撞上一个假的 409。
        self._release_slot(record)
        with self.lock:
            record.status = status
            record.detail = detail
            record.error = error
            record.completed_at = _now()
        self._persist(record)

    def _release_slot(self, record: JobRecord) -> None:
        with self.lock:
            count = self.active_counts.get(record.project_id, 0) - 1
            if count <= 0:
                self.active_counts.pop(record.project_id, None)
            else:
                self.active_counts[record.project_id] = count
            if record.exclusive:
                self.exclusive_projects.discard(record.project_id)
            if record.resource_key:
                self.active_resources.discard(record.resource_key)

    def _load_history(self) -> list[dict[str, Any]]:
        if self.history_path is None or not self.history_path.is_file():
            return []
        lock = self.history_lock or self.lock
        with lock:
            try:
                with self.history_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, ValueError, json.JSONDecodeError):
                return []
        return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []

    def _persist(self, record: JobRecord) -> None:
        if self.history_path is None:
            return
        lock = self.history_lock or self.lock
        with lock:
            items = self._load_history()
            snapshot = asdict(record)
            items = [item for item in items if item.get("id") != record.id]
            items.append(snapshot)
            items = items[-300:]
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(
                prefix=".jobs.", suffix=".tmp", dir=self.history_path.parent
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                    json.dump(items, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_name, self.history_path)
            except Exception:
                try:
                    os.unlink(temp_name)
                except FileNotFoundError:
                    pass


class AppContext:
    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()
        self.projects_root = self.workspace / "projects"
        self.projects_root.mkdir(parents=True, exist_ok=True)
        self.settings = ModelSettingsStore(self.workspace)
        self.prompts = PromptStore(self.workspace)
        self.profiles = AgentProfileStore(self.workspace)
        self.profiles.migrate_model_references(self.settings.model_name_map())
        self.runtime = RuntimeCoordinator()
        self.story_ideas = StoryIdeaStore(self.workspace)
        history_path = self.workspace / ".novel_agents" / "job_history.json"
        self.jobs = JobManager(
            max_workers=2, default_project_limit=1, history_path=history_path
        )
        self.memory_jobs = JobManager(
            max_workers=4, default_project_limit=1, history_path=history_path
        )

    def list_story_ideas(self) -> list[dict[str, Any]]:
        return self.story_ideas.list_ideas()

    def get_story_idea(self, idea_id: str) -> dict[str, Any]:
        try:
            return self.story_ideas.get(idea_id)
        except KeyError as exc:
            raise ApiError(HTTPStatus.NOT_FOUND, str(exc)) from exc
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc

    def create_story_idea(self, values: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            return self.story_ideas.create(values or {})
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc

    def update_story_idea(self, idea_id: str, values: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.story_ideas.update(idea_id, values)
        except KeyError as exc:
            raise ApiError(HTTPStatus.NOT_FOUND, str(exc)) from exc
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc

    def delete_story_idea(self, idea_id: str) -> dict[str, Any]:
        try:
            return self.story_ideas.delete(idea_id)
        except KeyError as exc:
            raise ApiError(HTTPStatus.NOT_FOUND, str(exc)) from exc
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc

    def create_project_from_story_idea(self, idea_id: str) -> dict[str, Any]:
        try:
            idea = self.story_ideas.get(idea_id)
            values = idea_to_project_values(idea)
        except KeyError as exc:
            raise ApiError(HTTPStatus.NOT_FOUND, str(exc)) from exc
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        project = self.create_project(values)
        try:
            idea = self.story_ideas.mark_created_project(idea_id, project["id"])
        except KeyError:
            idea = {**idea, "project_id": project["id"]}
        return {"project": project, "idea": idea}

    def list_projects(self) -> list[dict[str, Any]]:
        projects = []
        for path in self.projects_root.iterdir():
            if not path.is_dir() or not (path / "project.json").exists():
                continue
            try:
                store = ArtifactStore(path)
                brief = store.load_brief()
                state = store.read_state()
                projects.append(
                    {
                        "id": path.name,
                        "title": brief.title,
                        "genre": brief.genre,
                        "writing_genre": brief.writing_genre,
                        "chapter_count": brief.chapter_count,
                        "memory_enabled": MemorySettingsStore(path).exists(),
                        "updated_at": state.get("updated_at", ""),
                    }
                )
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return sorted(projects, key=lambda item: item["updated_at"], reverse=True)

    def create_project(self, values: dict[str, Any]) -> dict[str, Any]:
        brief = ProjectBrief.from_dict(values)
        slug = _slugify(brief.title)
        project_id = f"{slug}-{uuid.uuid4().hex[:6]}"
        store = ArtifactStore(self._project_path(project_id))
        store.initialize(brief)
        RelationshipStateStore(store.root).initialize()
        StyleReviewSettingsStore(store.root).initialize()
        if bool(values.get("memory_enabled", True)):
            MemorySettingsStore(store.root).initialize()
        return self.project_detail(project_id)

    def update_project(self, project_id: str, values: dict[str, Any]) -> dict[str, Any]:
        project_path = self._existing_project_path(project_id)
        if self.jobs.has_active(project_id) or self.memory_jobs.has_active(project_id):
            raise ApiError(HTTPStatus.CONFLICT, "项目有任务正在运行，暂时不能修改")
        store = ArtifactStore(project_path)
        before = store.load_brief().to_dict()
        merged = dict(before)
        for field in (
            "title",
            "premise",
            "genre",
            "writing_genre",
            "target_readers",
            "style",
            "chapter_count",
            "chapter_word_count",
            "language",
            "constraints",
        ):
            if field in values:
                merged[field] = values[field]
        brief = ProjectBrief.from_dict(merged)
        highest = store.highest_material_chapter()
        if brief.chapter_count < highest:
            raise ValueError(
                f"项目已有第 {highest} 章资料，章节数不能减少到 {brief.chapter_count}"
            )
        store.update_brief(brief)
        if bool(values.get("memory_enabled", False)):
            MemorySettingsStore(project_path).initialize()
        changed_fields = [key for key in before if before[key] != brief.to_dict()[key]]
        detail = self.project_detail(project_id)
        detail["changed_fields"] = changed_fields
        detail["planning_stale"] = bool(
            set(changed_fields)
            & {
                "premise",
                "genre",
                "writing_genre",
                "target_readers",
                "style",
                "chapter_count",
                "chapter_word_count",
                "language",
                "constraints",
            }
        )
        return detail

    def delete_project(self, project_id: str) -> dict[str, Any]:
        project_path = self._existing_project_path(project_id)
        if self.jobs.has_active(project_id) or self.memory_jobs.has_active(project_id):
            raise ApiError(HTTPStatus.CONFLICT, "项目有任务正在运行，不能删除")
        if project_path.parent != self.projects_root or not project_path.is_dir():
            raise ApiError(HTTPStatus.BAD_REQUEST, "非法项目删除路径")
        title = ArtifactStore(project_path).load_brief().title
        self.runtime.resume(project_id)
        shutil.rmtree(project_path)
        return {"deleted": True, "id": project_id, "title": title}

    def update_agent(self, agent_id: str, values: dict[str, Any]) -> dict[str, Any]:
        return self.prompts.update_custom(agent_id, values)

    def delete_agent(self, agent_id: str) -> dict[str, Any]:
        deleted = self.prompts.delete_custom(agent_id)
        self.profiles.delete(agent_id)
        return {"deleted": True, "agent": deleted}

    def agent_profile(self, agent_id: str) -> dict[str, Any]:
        agent = self.prompts.get_agent(agent_id)
        profile = self.profiles.get(agent_id)
        parent_id = str(agent.get("profile_parent", ""))
        resolved_profile_id = profile.get("model_profile_id")
        resolved_legacy_model = profile.get("model")
        if parent_id:
            parent_profile = self.profiles.get(parent_id)
            inherited_access = dict(parent_profile["memory_access"])
            inherited_access.update(self.profiles.memory_access_overrides(agent_id))
            profile["memory_access"] = inherited_access
            profile["profile_parent"] = parent_id
            if not resolved_profile_id and not resolved_legacy_model:
                resolved_profile_id = parent_profile.get("model_profile_id")
                resolved_legacy_model = parent_profile.get("model")
                profile["model_inherited_from"] = parent_id
        profile["resolved_model"] = self.settings.resolved_public(
            str(resolved_profile_id) if resolved_profile_id else None,
            str(resolved_legacy_model) if resolved_legacy_model else None,
        )
        return profile

    def update_agent_profile(
        self, agent_id: str, values: dict[str, Any]
    ) -> dict[str, Any]:
        self.prompts.get_agent(agent_id)
        if "model_profile_id" in values:
            selected_id = str(values.get("model_profile_id") or "").strip()
            if selected_id:
                available = {
                    item["id"]: item for item in self.settings.registry_public()["models"]
                }
                if selected_id not in available:
                    raise ValueError("选择的模型配置不存在")
                if not available[selected_id]["enabled"]:
                    raise ValueError("已停用的模型不能分配给智能体")
            values.setdefault("model", None)
        self.profiles.update(agent_id, values)
        return self.agent_profile(agent_id)

    def models_public(self) -> dict[str, Any]:
        registry = self.settings.registry_public()
        agents = {
            str(item["id"]): str(item.get("display_name") or item["id"])
            for item in self.prompts.list_agents()
        }
        for model in registry["models"]:
            references = self.profiles.model_profile_references(str(model["id"]))
            model["used_by"] = [agents.get(agent_id, agent_id) for agent_id in references]
            model["usage_count"] = len(references)
        return registry

    def create_model(self, values: dict[str, Any]) -> dict[str, Any]:
        model = self.settings.create_model(values)
        self.profiles.migrate_model_references(self.settings.model_name_map())
        return {"model": model, "registry": self.models_public()}

    def update_model(self, model_id: str, values: dict[str, Any]) -> dict[str, Any]:
        model = self.settings.update_model(model_id, values)
        self.profiles.migrate_model_references(self.settings.model_name_map())
        return {"model": model, "registry": self.models_public()}

    def delete_model(self, model_id: str) -> dict[str, Any]:
        references = self.profiles.model_profile_references(model_id)
        if references:
            names = {
                str(item["id"]): str(item.get("display_name") or item["id"])
                for item in self.prompts.list_agents()
            }
            labels = "、".join(names.get(agent_id, agent_id) for agent_id in references)
            raise ApiError(
                HTTPStatus.CONFLICT,
                f"该模型仍被智能体使用：{labels}。请先重新分配模型",
            )
        return self.settings.delete_model(model_id)

    def set_default_model(self, model_id: str) -> dict[str, Any]:
        model = self.settings.set_active(model_id)
        return {"model": model, "registry": self.models_public()}

    def _model_assignment_snapshot(self, mock: bool) -> dict[str, Any]:
        if mock:
            return {
                "captured_at": _now(),
                "default": {"id": "mock", "display_name": "模拟模型", "model": "mock"},
                "agents": {},
            }
        registry = self.settings.registry_public()
        default = next(
            item for item in registry["models"] if item["id"] == registry["active_model_id"]
        )
        agents: dict[str, dict[str, Any]] = {}
        for descriptor in self.prompts.list_agents():
            agent_id = str(descriptor["id"])
            resolved = self.agent_profile(agent_id)["resolved_model"]
            agents[agent_id] = {
                "id": resolved["id"],
                "display_name": resolved["display_name"],
                "model": resolved["model"],
                "fallback": bool(resolved.get("fallback", False)),
            }
        return {
            "captured_at": _now(),
            "default": {
                "id": default["id"],
                "display_name": default["display_name"],
                "model": default["model"],
            },
            "agents": agents,
        }

    def project_detail(self, project_id: str) -> dict[str, Any]:
        store = ArtifactStore(self._existing_project_path(project_id))
        memory = self.memory_public(project_id)
        return {
            "id": project_id,
            "brief": store.load_brief().to_dict(),
            "state": store.read_state(),
            "artifacts": self.list_artifacts(project_id),
            "memory": memory,
            "chapters": self.chapter_statuses(project_id),
        }

    def list_artifacts(self, project_id: str) -> list[dict[str, Any]]:
        root = self._existing_project_path(project_id)
        files: list[dict[str, Any]] = []
        for folder in (root / "artifacts", root / "chapters"):
            if not folder.exists():
                continue
            for path in folder.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in {".json", ".md", ".txt"}:
                    continue
                if path == root / "chapters" / "manifest.json":
                    continue
                stat = path.stat()
                try:
                    content = path.read_text(encoding="utf-8")
                    char_count = count_generated_chars(content)
                except OSError:
                    content = ""
                    char_count = 0
                files.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "name": path.name,
                        "type": path.suffix.lower().lstrip("."),
                        "size": stat.st_size,
                        "char_count": char_count,
                        "char_count_label": format_char_count(char_count),
                        "updated_at": datetime.fromtimestamp(
                            stat.st_mtime, timezone.utc
                        ).isoformat(),
                    }
                )
        return sorted(files, key=lambda item: item["path"])

    def read_artifact(self, project_id: str, relative_path: str) -> dict[str, Any]:
        store = ArtifactStore(self._existing_project_path(project_id))
        try:
            candidate = store.resolve_output_path(relative_path)
        except FileNotFoundError as exc:
            raise ApiError(HTTPStatus.NOT_FOUND, str(exc)) from exc
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        content = store.read_output(relative_path)
        char_count = count_generated_chars(content)
        return {
            "path": candidate.relative_to(store.root).as_posix(),
            "content": content,
            "char_count": char_count,
            "char_count_label": format_char_count(char_count),
        }

    def update_artifact(
        self, project_id: str, relative_path: str, values: dict[str, Any]
    ) -> dict[str, Any]:
        project_path = self._existing_project_path(project_id)
        self._assert_project_idle(project_id, "修改产物")
        if "content" not in values:
            raise ApiError(HTTPStatus.BAD_REQUEST, "缺少产物内容")
        store = ArtifactStore(project_path)
        previous_content = ""
        try:
            previous_content = store.read_output(relative_path)
        except (FileNotFoundError, ValueError):
            pass
        try:
            path = store.write_output(relative_path, str(values["content"]))
        except FileNotFoundError as exc:
            raise ApiError(HTTPStatus.NOT_FOUND, str(exc)) from exc
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        relative = path.relative_to(project_path).as_posix()
        planning_stale = relative.startswith("artifacts/")
        if planning_stale:
            store.update_stage("required_planning", "stale", f"产物已人工修改：{relative}")
        relationship_stale = {"affected": 0, "restored": 0}
        final_match = re.fullmatch(r"chapters/CH(\d+)_final\.md", relative, re.I)
        if (
            final_match
            and content_hash(previous_content) != content_hash(str(values["content"]))
            and RelationshipStateStore(project_path).exists()
        ):
            relationship_stale = RelationshipStateStore(
                project_path
            ).mark_chapter_stale(
                int(final_match.group(1)), "定稿正文已被人工修改"
            )
        return {
            "path": relative,
            "content": path.read_text(encoding="utf-8"),
            "planning_stale": planning_stale,
            "memory_stale": bool(re.fullmatch(r"chapters/CH\d+_final\.md", relative, re.I)),
            "relationship_stale": relationship_stale["affected"],
        }

    def delete_artifact(self, project_id: str, relative_path: str) -> dict[str, Any]:
        project_path = self._existing_project_path(project_id)
        self._assert_project_idle(project_id, "删除产物")
        store = ArtifactStore(project_path)
        try:
            path = store.resolve_output_path(relative_path)
        except FileNotFoundError as exc:
            raise ApiError(HTTPStatus.NOT_FOUND, str(exc)) from exc
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc

        relative = path.relative_to(project_path).as_posix()
        final_match = re.fullmatch(r"chapters/CH(\d+)_final\.md", relative, re.I)
        memory_chunks_removed = 0
        if final_match and MemorySettingsStore(project_path).exists():
            chapter_number = int(final_match.group(1))
            chunk_ids = MemoryStore(project_path).delete_chapter(chapter_number)
            settings = MemorySettingsStore(project_path).load()
            NumpyVectorIndex(
                project_path / "memory", int(settings["embedding"]["dimensions"])
            ).delete(chunk_ids)
            memory_chunks_removed = len(chunk_ids)
        relationship_stale = {"affected": 0, "restored": 0}
        if final_match and RelationshipStateStore(project_path).exists():
            relationship_stale = RelationshipStateStore(
                project_path
            ).mark_chapter_stale(
                int(final_match.group(1)), "定稿正文已删除"
            )

        deleted = store.delete_output(relative)
        planning_stale = relative.startswith("artifacts/")
        if planning_stale:
            store.update_stage("required_planning", "stale", f"产物已删除：{relative}")
        return {
            "deleted": True,
            "path": deleted.relative_to(project_path).as_posix(),
            "planning_stale": planning_stale,
            "memory_chunks_removed": memory_chunks_removed,
            "relationship_events_stale": relationship_stale["affected"],
        }

    def start_action(self, project_id: str, values: dict[str, Any]) -> dict[str, Any]:
        project_path = self._existing_project_path(project_id)
        action = str(values.get("action", "")).strip()
        if action not in {"plan", "chapter_plan", "write", "run"}:
            raise ApiError(HTTPStatus.BAD_REQUEST, "不支持的任务类型")
        chapter = int(values.get("chapter", 0) or 0)
        force = bool(values.get("force", False))
        mock = bool(values.get("mock", False))

        def work(record: JobRecord) -> str:
            record.model_snapshot = self._model_assignment_snapshot(mock)
            workflow, memory = self._build_workflow(project_id, project_path, mock)
            if action == "plan":
                workflow.plan(force=force)
                return "总体规划已完成"
            if action == "chapter_plan":
                path = workflow.plan_chapter(chapter, force=force)
                return self._format_generation_detail(
                    path,
                    kind="章节规划",
                    chapter_number=chapter,
                )
            if action == "write":
                with self.runtime.writing(project_id):
                    path = workflow.write_chapter(chapter, force=force)
                    style_detail = self._run_post_write_style(project_path, chapter, mock)
                return (
                    self._format_generation_detail(
                        path,
                        kind="章节正文",
                        chapter_number=chapter,
                    )
                    + style_detail
                )
            with self.runtime.writing(project_id):
                paths = workflow.run(force=force)
                style_messages = [
                    self._run_post_write_style(project_path, number, mock)
                    for number in range(1, len(paths) + 1)
                ]
            failures = sum("失败" in message for message in style_messages)
            total_chars = 0
            for path in paths:
                try:
                    total_chars += count_generated_chars(
                        Path(path).read_text(encoding="utf-8")
                    )
                except OSError:
                    continue
            suffix = f"；{failures} 章质检失败" if failures else ""
            return (
                f"已生成 {len(paths)} 个章节，合计 {format_char_count(total_chars)}"
                f"{suffix}"
            )

        return self.jobs.submit(project_id, action, work)

    def _format_generation_detail(
        self,
        path: str | Path,
        *,
        kind: str,
        chapter_number: int | None = None,
    ) -> str:
        target = Path(path)
        try:
            content = target.read_text(encoding="utf-8")
            char_count = count_generated_chars(content)
        except OSError:
            char_count = 0
        chapter_label = (
            f"第 {chapter_number} 章" if chapter_number and chapter_number > 0 else kind
        )
        return (
            f"{chapter_label}{kind}已生成，{format_char_count(char_count)}"
            f"：{target}"
        )

    def style_public(self, project_id: str, chapter_number: int) -> dict[str, Any]:
        project_path = self._existing_project_path(project_id)
        return StyleReviewService(project_path).public(chapter_number)

    def update_style_settings(
        self, project_id: str, values: dict[str, Any]
    ) -> dict[str, Any]:
        project_path = self._existing_project_path(project_id)
        self._assert_project_idle(project_id, "修改正文质检设置")
        settings = StyleReviewSettingsStore(project_path).update(values)
        return {"settings": settings}

    def start_style_action(
        self,
        project_id: str,
        chapter_number: int,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        project_path = self._existing_project_path(project_id)
        action = str(values.get("action", "")).strip()
        if action not in {"audit", "revise", "recheck"}:
            raise ApiError(HTTPStatus.BAD_REQUEST, "不支持的正文质检任务")
        mock = bool(values.get("mock", False))
        source = str(values.get("source", "draft")).strip() or "draft"
        issue_ids = values.get("issue_ids")
        if issue_ids is not None and not isinstance(issue_ids, list):
            raise ApiError(HTTPStatus.BAD_REQUEST, "issue_ids 必须是数组")
        normalized_issue_ids = (
            [str(item) for item in issue_ids] if issue_ids is not None else None
        )

        def work(record: JobRecord) -> str:
            record.model_snapshot = self._model_assignment_snapshot(mock)
            service = self._build_style_service(project_path, mock)
            store = ArtifactStore(project_path)
            stage = f"style_review_{chapter_number:03d}"
            store.update_stage(stage, "running", action)
            try:
                with self.runtime.writing(project_id):
                    if action == "audit":
                        settings = StyleReviewSettingsStore(project_path).initialize()
                        if settings["mode"] == "off":
                            raise ValueError("正文质检当前已关闭")
                        result = service.run_pipeline(chapter_number, source)
                        audit = result["review"].get("audit") or {}
                        detail = f"正文检测完成，得分 {audit.get('score', 0)}"
                    elif action == "revise":
                        result = service.revise_chapter(
                            chapter_number,
                            normalized_issue_ids,
                            int(values.get("rounds", 0) or 0) or None,
                            bool(values["second_audit"])
                            if "second_audit" in values
                            else None,
                        )
                        review = result["review"]
                        detail = f"修改稿已生成，第 {review.get('round', 1)} 轮"
                    else:
                        result = service.recheck_candidate(chapter_number)
                        audit = result["review"].get("revision_audit") or {}
                        detail = f"修改稿复检完成，得分 {audit.get('score', 0)}"
            except Exception as exc:
                store.update_stage(stage, "failed", str(exc))
                raise
            store.update_stage(stage, "completed", detail)
            return detail

        return self.jobs.submit(
            project_id,
            f"style_{action}",
            work,
            kind="style_review",
            resource_key=f"{project_id}:style:{chapter_number}",
        )

    def save_style_candidate(
        self, project_id: str, chapter_number: int, text: str
    ) -> dict[str, Any]:
        self._assert_project_idle(project_id, "保存正文修改稿")
        return StyleReviewService(
            self._existing_project_path(project_id)
        ).update_candidate(chapter_number, text)

    def accept_style_candidate(
        self,
        project_id: str,
        chapter_number: int,
        text: str | None,
    ) -> dict[str, Any]:
        self._assert_project_idle(project_id, "接受正文修改稿")
        result = StyleReviewService(
            self._existing_project_path(project_id)
        ).accept_candidate(chapter_number, text)
        result["chapter"] = self.chapter_detail(project_id, chapter_number)
        return result

    def reject_style_candidate(
        self, project_id: str, chapter_number: int
    ) -> dict[str, Any]:
        self._assert_project_idle(project_id, "拒绝正文修改稿")
        return StyleReviewService(
            self._existing_project_path(project_id)
        ).reject_candidate(chapter_number)

    def _build_style_service(
        self, project_path: Path, mock: bool
    ) -> StyleReviewService:
        if mock:
            audit_model = MockLanguageModel()
            editor_model = audit_model
            relationship_model = audit_model
        else:
            base_model = self.settings.create_client()
            try:
                audit_model = self.settings.create_client(
                    self.profiles.model_overrides("style_audit")
                )
            except ValueError:
                audit_model = base_model
            try:
                editor_model = self.settings.create_client(
                    self.profiles.model_overrides("style_editor")
                )
            except ValueError:
                editor_model = base_model
            try:
                relationship_model = self.settings.create_client(
                    self.profiles.model_overrides("relationship_audit")
                )
            except ValueError:
                relationship_model = base_model
        prompts = self.prompts.prompt_overrides()
        return StyleReviewService(
            project_path,
            StyleAuditor(audit_model, prompts.get("style_audit")),
            StyleEditor(editor_model, prompts.get("style_editor")),
            RelationshipAuditor(
                relationship_model, prompts.get("relationship_audit")
            ),
        )

    def _run_post_write_style(
        self, project_path: Path, chapter_number: int, mock: bool
    ) -> str:
        settings = StyleReviewSettingsStore(project_path).initialize()
        if settings["mode"] == "off":
            return ""
        store = ArtifactStore(project_path)
        stage = f"style_review_{chapter_number:03d}"
        store.update_stage(stage, "running", "正文生成后自动检测")
        try:
            result = self._build_style_service(project_path, mock).run_pipeline(
                chapter_number, "draft"
            )
            audit = result["review"].get("audit") or {}
            detail = f"文风检测 {audit.get('score', 0)} 分"
            store.update_stage(stage, "completed", detail)
            return f" | {detail}"
        except Exception as exc:
            store.update_stage(stage, "failed", str(exc))
            return f" | 文风检测失败：{exc}"

    def recent_jobs(self, project_id: str = "", limit: int = 50) -> list[dict[str, Any]]:
        combined = self.jobs.recent(project_id, limit) + self.memory_jobs.recent(
            project_id, limit
        )
        unique = {str(item.get("id", "")): item for item in combined}
        return sorted(
            unique.values(), key=lambda item: str(item.get("created_at", "")), reverse=True
        )[: max(1, min(int(limit), 200))]

    def _assert_project_idle(self, project_id: str, operation: str) -> None:
        if self.jobs.has_active(project_id) or self.memory_jobs.has_active(project_id):
            raise ApiError(
                HTTPStatus.CONFLICT,
                f"项目有任务正在运行，暂时不能{operation}",
            )

    def test_model(self, values: dict[str, Any] | None = None) -> dict[str, Any]:
        client = self.settings.create_client(values)
        text = client.generate_text(
            "你是连接测试助手。只能回复：连接成功",
            {"task": "connection_test"},
        )
        return {"ok": True, "response": text[:200]}

    def test_saved_model(self, model_id: str) -> dict[str, Any]:
        try:
            client = self.settings.create_client(model_profile_id=model_id)
            text = client.generate_text(
                "你是连接测试助手。只能回复：连接成功",
                {"task": "connection_test"},
            )
        except Exception as exc:
            self.settings.record_test(model_id, False, str(exc))
            raise
        model = self.settings.record_test(model_id, True, text[:200])
        return {"ok": True, "response": text[:200], "model": model}

    def _build_workflow(
        self,
        project_id: str,
        project_path: Path,
        mock: bool,
    ) -> tuple[NovelWorkflow, MemoryPipeline | None]:
        base_model = MockLanguageModel() if mock else self.settings.create_client()
        agent_models: dict[str, Any] = {}
        agent_memory_access: dict[str, dict[str, Any]] = {}
        descriptors = self.prompts.list_agents()
        for descriptor in descriptors:
            agent_id = str(descriptor["id"])
            parent_id = str(descriptor.get("profile_parent", ""))
            if parent_id:
                memory_access = dict(self.profiles.get(parent_id)["memory_access"])
                memory_access.update(self.profiles.memory_access_overrides(agent_id))
                agent_memory_access[agent_id] = memory_access
            else:
                agent_memory_access[agent_id] = self.profiles.get(agent_id)["memory_access"]
        if not mock:
            for descriptor in descriptors:
                agent_id = str(descriptor["id"])
                overrides: dict[str, Any] = {}
                parent_id = str(descriptor.get("profile_parent", ""))
                if parent_id:
                    overrides.update(self.profiles.model_overrides(parent_id))
                overrides.update(self.profiles.model_overrides(agent_id))
                try:
                    agent_models[agent_id] = self.settings.create_client(
                        overrides
                    )
                except ValueError:
                    continue
        memory = None
        memory_settings = MemorySettingsStore(project_path)
        if memory_settings.exists():
            embedder = None
            if mock:
                embedder = HashEmbeddingProvider(
                    int(memory_settings.load()["embedding"]["dimensions"])
                )
            memory = MemoryPipeline(
                project_path,
                base_model,
                self.workspace / ".novel_agents" / "models",
                coordinator=self.runtime,
                embedder=embedder,
                agent_models=agent_models,
                prompt_overrides=self.prompts.prompt_overrides(),
            )
        workflow = NovelWorkflow(
            project_path,
            base_model,
            prompt_overrides=self.prompts.prompt_overrides(),
            agent_models=agent_models,
            agent_memory_access=agent_memory_access,
            memory_pipeline=memory,
        )
        return workflow, memory

    def relationship_public(self, project_id: str) -> dict[str, Any]:
        project_path = self._existing_project_path(project_id)
        relationships = RelationshipStateStore(project_path)
        relationships.initialize()
        artifacts = ArtifactStore(project_path)
        if artifacts.has_artifact("relationship_arcs"):
            relationships.import_plan(artifacts.read_artifact("relationship_arcs"))
        result = relationships.public()
        result["runtime"] = self.runtime.status(project_id)
        return result

    def update_relationship_settings(
        self, project_id: str, values: dict[str, Any]
    ) -> dict[str, Any]:
        self._assert_project_idle(project_id, "修改人物关系设置")
        store = RelationshipStateStore(self._existing_project_path(project_id))
        return {"settings": store.update_settings(values)}

    def create_relationship_pair(
        self, project_id: str, values: dict[str, Any]
    ) -> dict[str, Any]:
        self._assert_project_idle(project_id, "新增人物关系")
        pair = RelationshipStateStore(
            self._existing_project_path(project_id)
        ).create_pair(values)
        return {"pair": pair}

    def update_relationship_pair(
        self, project_id: str, pair_id: str, values: dict[str, Any]
    ) -> dict[str, Any]:
        self._assert_project_idle(project_id, "修改人物关系")
        pair = RelationshipStateStore(
            self._existing_project_path(project_id)
        ).update_pair(pair_id, values)
        return {"pair": pair}

    def delete_relationship_pair(
        self, project_id: str, pair_id: str
    ) -> dict[str, Any]:
        self._assert_project_idle(project_id, "删除人物关系")
        pair = RelationshipStateStore(
            self._existing_project_path(project_id)
        ).delete_pair(pair_id)
        return {"deleted": True, "pair": pair}

    def relationship_event_action(
        self, project_id: str, event_id: str, values: dict[str, Any]
    ) -> dict[str, Any]:
        self._assert_project_idle(project_id, "处理人物关系记录")
        store = RelationshipStateStore(self._existing_project_path(project_id))
        action = str(values.get("action", "")).strip()
        if action in {"accept", "reject"}:
            event = store.decide_event(event_id, action)
        elif action == "rollback":
            event = store.rollback_event(event_id)
        else:
            raise ApiError(HTTPStatus.BAD_REQUEST, "不支持的关系事件操作")
        return {"event": event, "relationships": store.public()}

    def start_relationship_action(
        self, project_id: str, values: dict[str, Any]
    ) -> dict[str, Any]:
        project_path = self._existing_project_path(project_id)
        action = str(values.get("action", "")).strip()
        if action not in {"analyze_chapter", "reimport_plan"}:
            raise ApiError(HTTPStatus.BAD_REQUEST, "不支持的人物关系任务")
        chapter = int(values.get("chapter", 0) or 0)
        mock = bool(values.get("mock", False))
        if action == "analyze_chapter":
            chapter_count = ArtifactStore(project_path).load_brief().chapter_count
            if chapter < 1 or chapter > chapter_count:
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    f"章节编号必须在 1 到 {chapter_count} 之间",
                )
            try:
                ChapterVersionStore(project_path).get_text(chapter, "final")
            except FileNotFoundError as exc:
                raise ApiError(HTTPStatus.BAD_REQUEST, "该章节尚未定稿") from exc

        def work(record: JobRecord) -> str:
            record.model_snapshot = self._model_assignment_snapshot(mock)
            artifacts = ArtifactStore(project_path)
            relationships = RelationshipStateStore(project_path)
            relationships.initialize()
            if action == "reimport_plan":
                if not artifacts.has_artifact("relationship_arcs"):
                    workflow, _ = self._build_workflow(
                        project_id, project_path, mock
                    )
                    workflow.plan(force=False)
                result = relationships.import_plan(
                    artifacts.read_artifact("relationship_arcs")
                )
                return json.dumps(result, ensure_ascii=False)
            stage = f"relationship_memory_{chapter:03d}"
            artifacts.update_stage(stage, "running", "分析最终定稿中的人物关系")
            try:
                chapter_plan = (
                    artifacts.read_chapter_plan(chapter)
                    if artifacts.has_chapter_plan(chapter)
                    else {"chapter_number": chapter, "characters": []}
                )
                characters = [
                    str(item)
                    for item in chapter_plan.get("characters", [])
                    if str(item).strip()
                ]
                context = relationships.context_for_characters(characters)
                schedule = {"proposed": 0, "skipped": 0, "event_ids": []}
                result = {"applied": 0, "pending": 0, "ignored": 0}
                if context.get("pairs"):
                    final_text = ChapterVersionStore(project_path).get_text(
                        chapter, "final"
                    )
                    beat = (
                        artifacts.read_relationship_beat(chapter)
                        if artifacts.has_relationship_beat(chapter)
                        else {}
                    )
                    agent = self._build_relationship_memory_agent(mock)
                    extraction = agent.run(
                        artifacts.load_brief(),
                        chapter,
                        final_text,
                        chapter_plan,
                        context,
                        beat,
                    )
                    result = relationships.apply_extraction(
                        chapter,
                        content_hash(final_text),
                        extraction,
                    )
                schedule = relationships.apply_schedule_proposals(chapter)
                if not context.get("pairs") and schedule["proposed"] == 0:
                    detail = "本章没有可更新的人物关系组合"
                else:
                    detail = (
                        f"关系记忆已更新：{result['applied']} 条自动记录，"
                        f"{result['pending']} 条抽取阶段建议待确认；"
                        f"计划表新增 {schedule['proposed']} 条到章阶段建议"
                    )
                artifacts.update_stage(stage, "completed", detail)
                return detail
            except Exception as exc:
                artifacts.update_stage(stage, "failed", str(exc))
                raise

        return self.jobs.submit(
            project_id,
            f"relationship_{action}",
            work,
            kind="relationship",
            resource_key=(
                f"{project_id}:relationship:{chapter}"
                if action == "analyze_chapter"
                else f"{project_id}:relationship-plan"
            ),
        )

    def _build_relationship_memory_agent(
        self, mock: bool
    ) -> RelationshipMemoryAgent:
        if mock:
            model = MockLanguageModel()
        else:
            base_model = self.settings.create_client()
            try:
                model = self.settings.create_client(
                    self.profiles.model_overrides("relationship_memory")
                )
            except ValueError:
                model = base_model
        return RelationshipMemoryAgent(
            model, self.prompts.prompt_overrides().get("relationship_memory")
        )

    def memory_public(self, project_id: str) -> dict[str, Any]:
        project_path = self._existing_project_path(project_id)
        settings_store = MemorySettingsStore(project_path)
        if not settings_store.exists():
            return {"enabled": False}
        settings = settings_store.load()
        store = MemoryStore(project_path)
        versions = ChapterVersionStore(project_path)
        indexed = store.indexed_hashes()
        brief = ArtifactStore(project_path).load_brief()
        statuses = versions.list_statuses(brief.chapter_count, indexed)
        overview = store.overview()
        vector_path = project_path / "memory" / "vector_index.npz"
        overview.update(
            {
                "enabled": True,
                "settings": settings,
                "vector_count": NumpyVectorIndex(
                    project_path / "memory", int(settings["embedding"]["dimensions"])
                ).count(),
                "vector_bytes": vector_path.stat().st_size if vector_path.exists() else 0,
                "waiting_index": sum(item["status"] == "waiting_index" for item in statuses),
                "stale_chapters": sum(item["status"] == "stale" for item in statuses),
                "chapter_statuses": statuses,
                "runtime": self.runtime.status(project_id),
                "model_state": self._model_state(settings["embedding"]),
            }
        )
        return overview

    def _model_state(self, embedding: dict[str, Any]) -> dict[str, Any]:
        path = self.workspace / ".novel_agents" / "models" / "model_state.json"
        if not path.exists():
            return {"status": "not_prepared"}
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            reference = str(embedding.get("model_path", "")).strip() or str(
                embedding["model"]
            )
            compatible = (
                data.get("model_reference", data.get("model")) == reference
                and data.get("backend")
                == ("onnx" if embedding["provider"] == "local_onnx" else "torch")
                and data.get("quantization_actual") == embedding["quantization"]
                and int(data.get("dimensions", 0) or 0)
                == int(embedding["dimensions"])
            )
            if embedding["quantization"] == "int8":
                root = Path(str(data.get("quantized_model_path", "")))
                compatible = compatible and (
                    root / str(data.get("quantized_file", ""))
                ).is_file()
            data["status"] = "prepared" if compatible else "needs_prepare"
            return data
        except (OSError, ValueError, json.JSONDecodeError):
            return {"status": "invalid"}

    def memory_settings(self, project_id: str) -> dict[str, Any]:
        return MemorySettingsStore(self._existing_project_path(project_id)).public()

    def update_memory_settings(self, project_id: str, values: dict[str, Any]) -> dict[str, Any]:
        project_path = self._existing_project_path(project_id)
        store = MemorySettingsStore(project_path)
        before = store.load() if store.exists() else store.initialize()
        after = store.update(values)
        return {
            "settings": after,
            "requires_rebuild": store.requires_rebuild(before, after),
        }

    def enable_memory(self, project_id: str, values: dict[str, Any] | None = None) -> dict[str, Any]:
        store = MemorySettingsStore(self._existing_project_path(project_id))
        settings = store.initialize(values)
        return {"enabled": True, "settings": settings}

    def chapter_statuses(self, project_id: str) -> list[dict[str, Any]]:
        project_path = self._existing_project_path(project_id)
        brief = ArtifactStore(project_path).load_brief()
        indexed = MemoryStore(project_path).indexed_hashes() if MemorySettingsStore(project_path).exists() else {}
        return ChapterVersionStore(project_path).list_statuses(brief.chapter_count, indexed)

    def chapter_detail(self, project_id: str, chapter_number: int) -> dict[str, Any]:
        project_path = self._existing_project_path(project_id)
        versions = ChapterVersionStore(project_path)
        result = versions.status(
            chapter_number,
            MemoryStore(project_path).indexed_hashes().get(chapter_number, "")
            if MemorySettingsStore(project_path).exists()
            else "",
        )
        for key, preferred in (("draft", "draft"), ("author", "author"), ("final", "final")):
            try:
                text = versions.get_text(chapter_number, preferred)
            except FileNotFoundError:
                text = ""
            char_count = count_generated_chars(text)
            result[key + "_text"] = text
            result[key + "_char_count"] = char_count
            result[key + "_char_count_label"] = format_char_count(char_count)
        return result

    def save_author_chapter(self, project_id: str, chapter_number: int, text: str) -> dict[str, Any]:
        path = ChapterVersionStore(self._existing_project_path(project_id)).write_author(
            chapter_number, text
        )
        return {"path": str(path), "chapter": self.chapter_detail(project_id, chapter_number)}

    def finalize_chapter(
        self,
        project_id: str,
        chapter_number: int,
        text: str | None = None,
        auto_index: bool = True,
        mock: bool = False,
    ) -> dict[str, Any]:
        project_path = self._existing_project_path(project_id)
        result = ChapterVersionStore(project_path).finalize(chapter_number, text)
        relationships = RelationshipStateStore(project_path)
        relationships.initialize()
        artifacts = ArtifactStore(project_path)
        if artifacts.has_artifact("relationship_arcs"):
            relationships.import_plan(artifacts.read_artifact("relationship_arcs"))
        relationship_state = relationships.load()
        if (
            relationship_state["settings"]["enabled"]
            and relationship_state["settings"]["auto_update_after_finalize"]
            and relationship_state["pairs"]
        ):
            try:
                result["relationship_job"] = self.start_relationship_action(
                    project_id,
                    {
                        "action": "analyze_chapter",
                        "chapter": chapter_number,
                        "mock": mock,
                    },
                )
            except ApiError as exc:
                result["relationship_error"] = exc.message
        if auto_index and MemorySettingsStore(project_path).exists():
            try:
                result["job"] = self.start_memory_action(
                    project_id,
                    {"action": "index_chapter", "chapter": chapter_number, "mock": mock},
                )
            except ApiError as exc:
                result["indexing_error"] = exc.message
        result["chapter"] = self.chapter_detail(project_id, chapter_number)
        return result

    def start_memory_action(self, project_id: str, values: dict[str, Any]) -> dict[str, Any]:
        project_path = self._existing_project_path(project_id)
        if not MemorySettingsStore(project_path).exists():
            raise ApiError(HTTPStatus.BAD_REQUEST, "该项目尚未启用记忆系统")
        action = str(values.get("action", "")).strip()
        if action in {"pause", "resume"}:
            if action == "pause":
                self.runtime.pause(project_id)
            else:
                self.runtime.resume(project_id)
            return {"status": action + "d", "runtime": self.runtime.status(project_id)}
        if action not in {"prepare_model", "index_chapter", "rebuild"}:
            raise ApiError(HTTPStatus.BAD_REQUEST, "不支持的记忆任务")
        chapter = int(values.get("chapter", 0) or 0)
        mock = bool(values.get("mock", False))
        if action == "index_chapter":
            chapter_count = ArtifactStore(project_path).load_brief().chapter_count
            if chapter < 1 or chapter > chapter_count:
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    f"章节编号必须在 1 到 {chapter_count} 之间",
                )

        def work(record: JobRecord) -> str:
            if action != "prepare_model":
                record.model_snapshot = self._model_assignment_snapshot(mock)
            settings = MemorySettingsStore(project_path).load()
            if action == "prepare_model":
                from .embedding import LocalSentenceTransformerEmbedder

                embedder = LocalSentenceTransformerEmbedder(
                    settings["embedding"],
                    self.workspace / ".novel_agents" / "models",
                    allow_download=True,
                )
                return json.dumps(embedder.prepare(), ensure_ascii=False)
            workflow, memory = self._build_workflow(project_id, project_path, mock)
            if memory is None:
                raise RuntimeError("记忆系统未启用")
            if action == "index_chapter":
                result = memory.index_chapter(
                    chapter,
                    progress=lambda phase, current, total: self.memory_jobs.update_progress(
                        record, phase, current, total
                    ),
                )
                return json.dumps(result, ensure_ascii=False)
            results = memory.rebuild()
            return f"已重建 {len(results)} 个章节的记忆索引"

        settings = MemorySettingsStore(project_path).load()
        limit = int(settings["indexing"]["max_parallel_jobs"])
        exclusive = action in {"prepare_model", "rebuild"}
        if action == "prepare_model":
            resource_key = "global:embedding-model-prepare"
        elif action == "index_chapter":
            resource_key = f"{project_id}:chapter:{chapter}"
        else:
            resource_key = f"{project_id}:memory-rebuild"
        return self.memory_jobs.submit(
            project_id,
            action,
            work,
            kind="memory",
            project_limit=1 if exclusive else limit,
            exclusive=exclusive,
            resource_key=resource_key,
        )

    def close(self, wait: bool = False) -> None:
        self.jobs.executor.shutdown(wait=wait, cancel_futures=True)
        self.memory_jobs.executor.shutdown(wait=wait, cancel_futures=True)

    def memory_search(self, project_id: str, values: dict[str, Any]) -> dict[str, Any]:
        project_path = self._existing_project_path(project_id)
        if not MemorySettingsStore(project_path).exists():
            raise ApiError(HTTPStatus.BAD_REQUEST, "该项目尚未启用记忆系统")
        mock = bool(values.get("mock", False))
        _, memory = self._build_workflow(project_id, project_path, mock)
        if memory is None:
            raise ApiError(HTTPStatus.BAD_REQUEST, "记忆系统未启用")
        return memory.retrieve(
            str(values.get("query", "")).strip(),
            chapter_number=int(values.get("chapter", 0) or 0) or None,
            characters=[str(item) for item in values.get("characters", [])],
            use_reranker=bool(values.get("use_reranker", True)),
        )

    def _existing_project_path(self, project_id: str) -> Path:
        path = self._project_path(project_id)
        if not (path / "project.json").is_file():
            raise ApiError(HTTPStatus.NOT_FOUND, "项目不存在")
        return path

    def _project_path(self, project_id: str) -> Path:
        if not project_id or "/" in project_id or "\\" in project_id:
            raise ApiError(HTTPStatus.BAD_REQUEST, "非法项目 ID")
        path = (self.projects_root / project_id).resolve()
        if path.parent != self.projects_root:
            raise ApiError(HTTPStatus.BAD_REQUEST, "非法项目路径")
        return path


def make_handler(app: AppContext) -> type[BaseHTTPRequestHandler]:
    static_root = Path(__file__).resolve().parent / "web_assets"

    class Handler(BaseHTTPRequestHandler):
        server_version = "NovelAgents/0.1"

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

        def do_PUT(self) -> None:
            self._dispatch("PUT")

        def do_DELETE(self) -> None:
            self._dispatch("DELETE")

        def _dispatch(self, method: str) -> None:
            try:
                parsed = urlparse(self.path)
                path = parsed.path
                if path.startswith("/api/") or path == "/api":
                    self._handle_api(method, path, parse_qs(parsed.query))
                else:
                    self._serve_static(path)
            except ApiError as exc:
                self._json(exc.status, {"error": exc.message})
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

        def _handle_api(
            self, method: str, path: str, query: dict[str, list[str]]
        ) -> None:
            if method == "GET" and path == "/api/health":
                self._json(HTTPStatus.OK, {"ok": True})
                return
            if path == "/api/config":
                if method == "GET":
                    self._json(HTTPStatus.OK, app.settings.public())
                    return
                if method == "PUT":
                    self._json(HTTPStatus.OK, app.settings.update(self._body()))
                    return
            if method == "POST" and path == "/api/config/test":
                self._json(HTTPStatus.OK, app.test_model(self._body()))
                return
            if path == "/api/models":
                if method == "GET":
                    self._json(HTTPStatus.OK, app.models_public())
                    return
                if method == "POST":
                    self._json(HTTPStatus.CREATED, app.create_model(self._body()))
                    return
            if method == "PUT" and path == "/api/models/default":
                body = self._body()
                self._json(
                    HTTPStatus.OK,
                    app.set_default_model(str(body.get("model_id", "")).strip()),
                )
                return
            model_test_match = re.fullmatch(r"/api/models/([^/]+)/test", path)
            if model_test_match and method == "POST":
                model_id = unquote(model_test_match.group(1))
                self._json(HTTPStatus.OK, app.test_saved_model(model_id))
                return
            model_match = re.fullmatch(r"/api/models/([^/]+)", path)
            if model_match:
                model_id = unquote(model_match.group(1))
                if method == "PUT":
                    self._json(HTTPStatus.OK, app.update_model(model_id, self._body()))
                    return
                if method == "DELETE":
                    self._json(HTTPStatus.OK, app.delete_model(model_id))
                    return
            if path == "/api/agents":
                if method == "GET":
                    self._json(HTTPStatus.OK, {"agents": app.prompts.list_agents()})
                    return
                if method == "POST":
                    self._json(
                        HTTPStatus.CREATED,
                        {"agent": app.prompts.create_custom(self._body())},
                    )
                    return

            agent_match = re.fullmatch(r"/api/agents/([^/]+)", path)
            prompt_match = re.fullmatch(r"/api/agents/([^/]+)/prompt", path)
            if agent_match and method == "GET":
                agent_id = unquote(agent_match.group(1))
                self._json(
                    HTTPStatus.OK,
                    {
                        "agent": app.prompts.get_agent(agent_id),
                        "profile": app.agent_profile(agent_id),
                    },
                )
                return
            if agent_match and method == "PUT":
                agent_id = unquote(agent_match.group(1))
                self._json(
                    HTTPStatus.OK,
                    {"agent": app.update_agent(agent_id, self._body())},
                )
                return
            if agent_match and method == "DELETE":
                agent_id = unquote(agent_match.group(1))
                self._json(HTTPStatus.OK, app.delete_agent(agent_id))
                return
            profile_match = re.fullmatch(r"/api/agents/([^/]+)/profile", path)
            if profile_match:
                agent_id = unquote(profile_match.group(1))
                if method == "GET":
                    self._json(HTTPStatus.OK, {"profile": app.agent_profile(agent_id)})
                    return
                if method == "PUT":
                    self._json(
                        HTTPStatus.OK,
                        {"profile": app.update_agent_profile(agent_id, self._body())},
                    )
                    return
            if prompt_match:
                agent_id = unquote(prompt_match.group(1))
                if method == "PUT":
                    body = self._body()
                    self._json(
                        HTTPStatus.OK,
                        {"agent": app.prompts.save_prompt(agent_id, str(body.get("prompt", "")))},
                    )
                    return
                if method == "DELETE":
                    self._json(
                        HTTPStatus.OK,
                        {"agent": app.prompts.reset_prompt(agent_id)},
                    )
                    return

            if path == "/api/projects":
                if method == "GET":
                    self._json(HTTPStatus.OK, {"projects": app.list_projects()})
                    return
                if method == "POST":
                    self._json(HTTPStatus.CREATED, app.create_project(self._body()))
                    return

            if path == "/api/story-ideas":
                if method == "GET":
                    self._json(HTTPStatus.OK, {"ideas": app.list_story_ideas()})
                    return
                if method == "POST":
                    self._json(
                        HTTPStatus.CREATED,
                        {"idea": app.create_story_idea(self._body())},
                    )
                    return
            story_idea_match = re.fullmatch(r"/api/story-ideas/([^/]+)", path)
            story_idea_project_match = re.fullmatch(
                r"/api/story-ideas/([^/]+)/create-project", path
            )
            if story_idea_project_match and method == "POST":
                idea_id = unquote(story_idea_project_match.group(1))
                self._json(
                    HTTPStatus.CREATED,
                    app.create_project_from_story_idea(idea_id),
                )
                return
            if story_idea_match:
                idea_id = unquote(story_idea_match.group(1))
                if method == "GET":
                    self._json(HTTPStatus.OK, {"idea": app.get_story_idea(idea_id)})
                    return
                if method == "PUT":
                    self._json(
                        HTTPStatus.OK,
                        {"idea": app.update_story_idea(idea_id, self._body())},
                    )
                    return
                if method == "DELETE":
                    self._json(HTTPStatus.OK, app.delete_story_idea(idea_id))
                    return

            artifact_match = re.fullmatch(r"/api/projects/([^/]+)/artifact", path)
            action_match = re.fullmatch(r"/api/projects/([^/]+)/actions", path)
            memory_match = re.fullmatch(r"/api/projects/([^/]+)/memory", path)
            memory_action_match = re.fullmatch(r"/api/projects/([^/]+)/memory/actions", path)
            memory_search_match = re.fullmatch(r"/api/projects/([^/]+)/memory/search", path)
            memory_settings_match = re.fullmatch(r"/api/projects/([^/]+)/memory/settings", path)
            relationships_match = re.fullmatch(r"/api/projects/([^/]+)/relationships", path)
            relationship_settings_match = re.fullmatch(
                r"/api/projects/([^/]+)/relationships/settings", path
            )
            relationship_actions_match = re.fullmatch(
                r"/api/projects/([^/]+)/relationships/actions", path
            )
            relationship_pairs_match = re.fullmatch(
                r"/api/projects/([^/]+)/relationships/pairs", path
            )
            relationship_pair_match = re.fullmatch(
                r"/api/projects/([^/]+)/relationships/pairs/([^/]+)", path
            )
            relationship_event_match = re.fullmatch(
                r"/api/projects/([^/]+)/relationships/events/([^/]+)", path
            )
            chapters_match = re.fullmatch(r"/api/projects/([^/]+)/chapters", path)
            chapter_match = re.fullmatch(r"/api/projects/([^/]+)/chapters/(\d+)", path)
            chapter_author_match = re.fullmatch(r"/api/projects/([^/]+)/chapters/(\d+)/author", path)
            chapter_finalize_match = re.fullmatch(r"/api/projects/([^/]+)/chapters/(\d+)/finalize", path)
            chapter_style_match = re.fullmatch(r"/api/projects/([^/]+)/chapters/(\d+)/style", path)
            chapter_style_settings_match = re.fullmatch(r"/api/projects/([^/]+)/chapters/(\d+)/style/settings", path)
            chapter_style_actions_match = re.fullmatch(r"/api/projects/([^/]+)/chapters/(\d+)/style/actions", path)
            chapter_style_candidate_match = re.fullmatch(r"/api/projects/([^/]+)/chapters/(\d+)/style/candidate", path)
            chapter_style_accept_match = re.fullmatch(r"/api/projects/([^/]+)/chapters/(\d+)/style/accept", path)
            chapter_style_reject_match = re.fullmatch(r"/api/projects/([^/]+)/chapters/(\d+)/style/reject", path)
            project_match = re.fullmatch(r"/api/projects/([^/]+)", path)
            if artifact_match:
                project_id = unquote(artifact_match.group(1))
                relative_path = query.get("path", [""])[0]
                if method == "GET":
                    self._json(HTTPStatus.OK, app.read_artifact(project_id, relative_path))
                    return
                if method == "PUT":
                    self._json(
                        HTTPStatus.OK,
                        app.update_artifact(project_id, relative_path, self._body()),
                    )
                    return
                if method == "DELETE":
                    self._json(
                        HTTPStatus.OK,
                        app.delete_artifact(project_id, relative_path),
                    )
                    return
            if action_match and method == "POST":
                project_id = unquote(action_match.group(1))
                self._json(HTTPStatus.ACCEPTED, app.start_action(project_id, self._body()))
                return
            if memory_settings_match:
                project_id = unquote(memory_settings_match.group(1))
                if method == "GET":
                    self._json(HTTPStatus.OK, app.memory_settings(project_id))
                    return
                if method == "PUT":
                    self._json(HTTPStatus.OK, app.update_memory_settings(project_id, self._body()))
                    return
            if memory_match and method == "GET":
                project_id = unquote(memory_match.group(1))
                self._json(HTTPStatus.OK, app.memory_public(project_id))
                return
            if memory_match and method == "POST":
                project_id = unquote(memory_match.group(1))
                self._json(HTTPStatus.OK, app.enable_memory(project_id, self._body()))
                return
            if memory_action_match and method == "POST":
                project_id = unquote(memory_action_match.group(1))
                self._json(
                    HTTPStatus.ACCEPTED,
                    app.start_memory_action(project_id, self._body()),
                )
                return
            if memory_search_match and method == "POST":
                project_id = unquote(memory_search_match.group(1))
                self._json(HTTPStatus.OK, app.memory_search(project_id, self._body()))
                return
            if relationships_match and method == "GET":
                project_id = unquote(relationships_match.group(1))
                self._json(HTTPStatus.OK, app.relationship_public(project_id))
                return
            if relationship_settings_match and method == "PUT":
                project_id = unquote(relationship_settings_match.group(1))
                self._json(
                    HTTPStatus.OK,
                    app.update_relationship_settings(project_id, self._body()),
                )
                return
            if relationship_actions_match and method == "POST":
                project_id = unquote(relationship_actions_match.group(1))
                self._json(
                    HTTPStatus.ACCEPTED,
                    app.start_relationship_action(project_id, self._body()),
                )
                return
            if relationship_pairs_match and method == "POST":
                project_id = unquote(relationship_pairs_match.group(1))
                self._json(
                    HTTPStatus.CREATED,
                    app.create_relationship_pair(project_id, self._body()),
                )
                return
            if relationship_pair_match:
                project_id = unquote(relationship_pair_match.group(1))
                pair_id = unquote(relationship_pair_match.group(2))
                if method == "PUT":
                    self._json(
                        HTTPStatus.OK,
                        app.update_relationship_pair(
                            project_id, pair_id, self._body()
                        ),
                    )
                    return
                if method == "DELETE":
                    self._json(
                        HTTPStatus.OK,
                        app.delete_relationship_pair(project_id, pair_id),
                    )
                    return
            if relationship_event_match and method == "POST":
                project_id = unquote(relationship_event_match.group(1))
                event_id = unquote(relationship_event_match.group(2))
                self._json(
                    HTTPStatus.OK,
                    app.relationship_event_action(
                        project_id, event_id, self._body()
                    ),
                )
                return
            if chapters_match and method == "GET":
                project_id = unquote(chapters_match.group(1))
                self._json(HTTPStatus.OK, {"chapters": app.chapter_statuses(project_id)})
                return
            if chapter_style_settings_match and method == "PUT":
                project_id = unquote(chapter_style_settings_match.group(1))
                self._json(
                    HTTPStatus.OK,
                    app.update_style_settings(project_id, self._body()),
                )
                return
            if chapter_style_actions_match and method == "POST":
                project_id = unquote(chapter_style_actions_match.group(1))
                chapter = int(chapter_style_actions_match.group(2))
                self._json(
                    HTTPStatus.ACCEPTED,
                    app.start_style_action(project_id, chapter, self._body()),
                )
                return
            if chapter_style_candidate_match and method == "PUT":
                project_id = unquote(chapter_style_candidate_match.group(1))
                chapter = int(chapter_style_candidate_match.group(2))
                body = self._body()
                self._json(
                    HTTPStatus.OK,
                    app.save_style_candidate(project_id, chapter, str(body.get("text", ""))),
                )
                return
            if chapter_style_accept_match and method == "POST":
                project_id = unquote(chapter_style_accept_match.group(1))
                chapter = int(chapter_style_accept_match.group(2))
                body = self._body()
                self._json(
                    HTTPStatus.OK,
                    app.accept_style_candidate(
                        project_id,
                        chapter,
                        str(body["text"]) if "text" in body else None,
                    ),
                )
                return
            if chapter_style_reject_match and method == "POST":
                project_id = unquote(chapter_style_reject_match.group(1))
                chapter = int(chapter_style_reject_match.group(2))
                self._json(
                    HTTPStatus.OK,
                    app.reject_style_candidate(project_id, chapter),
                )
                return
            if chapter_style_match and method == "GET":
                project_id = unquote(chapter_style_match.group(1))
                chapter = int(chapter_style_match.group(2))
                self._json(HTTPStatus.OK, app.style_public(project_id, chapter))
                return
            if chapter_author_match and method == "PUT":
                project_id = unquote(chapter_author_match.group(1))
                chapter = int(chapter_author_match.group(2))
                body = self._body()
                self._json(
                    HTTPStatus.OK,
                    app.save_author_chapter(project_id, chapter, str(body.get("text", ""))),
                )
                return
            if chapter_finalize_match and method == "POST":
                project_id = unquote(chapter_finalize_match.group(1))
                chapter = int(chapter_finalize_match.group(2))
                body = self._body()
                self._json(
                    HTTPStatus.ACCEPTED,
                    app.finalize_chapter(
                        project_id,
                        chapter,
                        text=str(body["text"]) if "text" in body else None,
                        auto_index=bool(body.get("auto_index", True)),
                        mock=bool(body.get("mock", False)),
                    ),
                )
                return
            if chapter_match and method == "GET":
                project_id = unquote(chapter_match.group(1))
                chapter = int(chapter_match.group(2))
                self._json(HTTPStatus.OK, app.chapter_detail(project_id, chapter))
                return
            if project_match and method == "GET":
                project_id = unquote(project_match.group(1))
                self._json(HTTPStatus.OK, app.project_detail(project_id))
                return
            if project_match and method == "PUT":
                project_id = unquote(project_match.group(1))
                self._json(HTTPStatus.OK, app.update_project(project_id, self._body()))
                return
            if project_match and method == "DELETE":
                project_id = unquote(project_match.group(1))
                self._json(HTTPStatus.OK, app.delete_project(project_id))
                return

            job_match = re.fullmatch(r"/api/jobs/([a-z0-9]+)", path)
            if path == "/api/jobs" and method == "GET":
                project_id = query.get("project_id", [""])[0]
                limit = int(query.get("limit", ["50"])[0])
                self._json(
                    HTTPStatus.OK,
                    {"jobs": app.recent_jobs(project_id, limit)},
                )
                return
            if job_match and method == "GET":
                job_id = job_match.group(1)
                try:
                    job = app.jobs.get(job_id)
                except ApiError:
                    job = app.memory_jobs.get(job_id)
                self._json(HTTPStatus.OK, job)
                return
            raise ApiError(HTTPStatus.NOT_FOUND, "接口不存在")

        def _body(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ApiError(HTTPStatus.BAD_REQUEST, "无效请求长度") from exc
            if length < 1 or length > 1_000_000:
                raise ApiError(HTTPStatus.BAD_REQUEST, "请求内容为空或过大")
            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise ApiError(HTTPStatus.BAD_REQUEST, "请求内容必须是 JSON 对象")
            return data

        def _json(self, status: int, data: dict[str, Any]) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _serve_static(self, request_path: str) -> None:
            relative = request_path.lstrip("/") or "index.html"
            candidate = (static_root / relative).resolve()
            if not candidate.is_relative_to(static_root) or not candidate.is_file():
                candidate = static_root / "index.html"
            if not candidate.is_file():
                raise ApiError(HTTPStatus.NOT_FOUND, "前端资源不存在")
            body = candidate.read_bytes()
            content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Novel Agents local workspace.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # 模型密钥从环境变量读取；<workspace>/.env 只是它的本地文件形式，
    # 已存在的进程环境变量优先，不会被文件覆盖。
    load_workspace_env(args.workspace)
    app = AppContext(args.workspace)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"Novel Agents: http://{args.host}:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close(wait=False)
    return 0


def _slugify(title: str) -> str:
    slug = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", title, flags=re.UNICODE).strip("-_")
    return slug[:36] or "novel"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
