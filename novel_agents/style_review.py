from __future__ import annotations

import copy
import difflib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agents import ChapterWriter
from .artifacts import ArtifactStore
from .chapter_versions import ChapterVersionStore, content_hash
from .relationship_agents import RelationshipAuditor, merge_audit_reports
from .relationship_store import RelationshipStateStore
from .style_agents import StyleAuditor, StyleEditor


DEFAULT_STYLE_REVIEW_SETTINGS: dict[str, Any] = {
    "mode": "assisted",
    "generate_revision_after_audit": True,
    "auto_revision_rounds": 1,
    "second_audit": True,
    "protect_author_version": True,
    "minimum_severity": "medium",
}


class StyleReviewSettingsStore:
    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.root = self.project_root / "style_review"
        self.path = self.root / "settings.json"

    def initialize(self, values: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.path.is_file():
            return self.load()
        settings = copy.deepcopy(DEFAULT_STYLE_REVIEW_SETTINGS)
        if values:
            settings.update(values)
        self._validate(settings)
        _write_json(self.path, settings)
        return settings

    def load(self) -> dict[str, Any]:
        settings = copy.deepcopy(DEFAULT_STYLE_REVIEW_SETTINGS)
        if self.path.is_file():
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                raise ValueError("style_review/settings.json 必须是对象")
            settings.update(data)
        self._validate(settings)
        return settings

    def update(self, values: dict[str, Any]) -> dict[str, Any]:
        settings = self.load()
        for key in DEFAULT_STYLE_REVIEW_SETTINGS:
            if key in values:
                settings[key] = values[key]
        self._validate(settings)
        _write_json(self.path, settings)
        return settings

    @staticmethod
    def _validate(settings: dict[str, Any]) -> None:
        if settings.get("mode") not in {"off", "detect_only", "assisted"}:
            raise ValueError("mode 必须是 off、detect_only 或 assisted")
        rounds = int(settings.get("auto_revision_rounds", 1))
        if rounds < 1 or rounds > 2:
            raise ValueError("auto_revision_rounds 必须在 1 到 2 之间")
        settings["auto_revision_rounds"] = rounds
        if settings.get("minimum_severity") not in {"low", "medium", "high"}:
            raise ValueError("minimum_severity 必须是 low、medium 或 high")
        for key in ("generate_revision_after_audit", "second_audit", "protect_author_version"):
            if not isinstance(settings.get(key), bool):
                raise ValueError(f"{key} 必须是布尔值")
        if not settings["protect_author_version"]:
            raise ValueError("作者版本保护不能关闭")


class StyleReviewStore:
    FILES = ("source.md", "audit.json", "candidate.md", "revision_audit.json", "meta.json")

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.root = self.project_root / "style_review" / "chapters"

    def chapter_dir(self, chapter_number: int) -> Path:
        return self.root / f"CH{chapter_number:03d}"

    def begin(self, chapter_number: int, source_kind: str, source_text: str, audit: dict[str, Any], author_text: str | None) -> None:
        folder = self.chapter_dir(chapter_number)
        self._archive_current(folder)
        folder.mkdir(parents=True, exist_ok=True)
        _write_text(folder / "source.md", source_text.rstrip() + "\n")
        _write_json(folder / "audit.json", audit)
        _write_json(folder / "meta.json", {
            "chapter_number": chapter_number,
            "source_kind": source_kind,
            "source_hash": content_hash(source_text),
            "author_hash_at_audit": content_hash(author_text) if author_text is not None else "",
            "status": "audited",
            "round": 0,
            "created_at": _now(),
            "updated_at": _now(),
        })

    def save_candidate(self, chapter_number: int, candidate: str, issue_ids: list[str], round_number: int, revision_audit: dict[str, Any] | None = None) -> None:
        folder = self.chapter_dir(chapter_number)
        if not (folder / "meta.json").is_file():
            raise FileNotFoundError("该章节尚未生成检测报告")
        _write_text(folder / f"candidate_round_{round_number}.md", candidate.rstrip() + "\n")
        _write_text(folder / "candidate.md", candidate.rstrip() + "\n")
        if revision_audit is not None:
            _write_json(folder / f"revision_audit_round_{round_number}.json", revision_audit)
            _write_json(folder / "revision_audit.json", revision_audit)
        meta = self._meta(chapter_number)
        meta.update({"status": "candidate_ready", "round": round_number, "candidate_hash": content_hash(candidate), "selected_issue_ids": issue_ids, "updated_at": _now()})
        _write_json(folder / "meta.json", meta)

    def save_revision_audit(self, chapter_number: int, report: dict[str, Any]) -> None:
        folder = self.chapter_dir(chapter_number)
        _write_json(folder / "revision_audit.json", report)
        meta = self._meta(chapter_number)
        meta.update({"status": "candidate_reviewed", "updated_at": _now()})
        _write_json(folder / "meta.json", meta)

    def update_candidate(self, chapter_number: int, text: str) -> None:
        if not text.strip():
            raise ValueError("修改稿不能为空")
        folder = self.chapter_dir(chapter_number)
        candidate = folder / "candidate.md"
        if not candidate.is_file():
            raise FileNotFoundError("该章节尚无修改稿")
        archive = folder / "candidate_edits"
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, archive / f"candidate_{_stamp()}.md")
        _write_text(candidate, text.rstrip() + "\n")
        revision_audit = folder / "revision_audit.json"
        if revision_audit.exists():
            revision_audit.unlink()
        meta = self._meta(chapter_number)
        meta.update({"status": "candidate_edited", "candidate_hash": content_hash(text), "updated_at": _now()})
        _write_json(folder / "meta.json", meta)

    def mark_accepted(self, chapter_number: int, text: str) -> None:
        folder = self.chapter_dir(chapter_number)
        meta = self._meta(chapter_number)
        meta.update({"status": "accepted", "accepted_hash": content_hash(text), "author_hash_at_audit": content_hash(text), "accepted_at": _now(), "updated_at": _now()})
        _write_json(folder / "meta.json", meta)

    def mark_rejected(self, chapter_number: int) -> None:
        folder = self.chapter_dir(chapter_number)
        meta = self._meta(chapter_number)
        meta.update({"status": "rejected", "rejected_at": _now(), "updated_at": _now()})
        _write_json(folder / "meta.json", meta)

    def report(self, chapter_number: int, current_source: str = "", current_author: str | None = None) -> dict[str, Any]:
        folder = self.chapter_dir(chapter_number)
        if not folder.is_dir() or not (folder / "meta.json").is_file():
            return {"chapter_number": chapter_number, "status": "not_started", "source_text": "", "candidate_text": "", "audit": None, "revision_audit": None, "diff": [], "source_stale": False, "author_stale": False}
        meta = self._meta(chapter_number)
        source_text = _read_text(folder / "source.md")
        candidate_text = _read_text(folder / "candidate.md")
        result = dict(meta)
        result.update({
            "source_text": source_text,
            "candidate_text": candidate_text,
            "audit": _read_json(folder / "audit.json"),
            "revision_audit": _read_json(folder / "revision_audit.json"),
            "diff": _paragraph_diff(source_text, candidate_text),
            "source_stale": bool(current_source and content_hash(current_source) != str(meta.get("source_hash", ""))),
            "author_stale": bool("author_hash_at_audit" in meta and (content_hash(current_author) if current_author is not None else "") != str(meta.get("author_hash_at_audit", ""))),
        })
        return result

    def source_text(self, chapter_number: int) -> str:
        text = _read_text(self.chapter_dir(chapter_number) / "source.md")
        if not text:
            raise FileNotFoundError("检测原稿不存在")
        return text

    def candidate_text(self, chapter_number: int) -> str:
        text = _read_text(self.chapter_dir(chapter_number) / "candidate.md")
        if not text:
            raise FileNotFoundError("修改稿不存在")
        return text

    def audit(self, chapter_number: int) -> dict[str, Any]:
        data = _read_json(self.chapter_dir(chapter_number) / "audit.json")
        if data is None:
            raise FileNotFoundError("检测报告不存在")
        return data

    def _meta(self, chapter_number: int) -> dict[str, Any]:
        data = _read_json(self.chapter_dir(chapter_number) / "meta.json")
        if data is None:
            raise FileNotFoundError("质检元数据不存在")
        return data

    @classmethod
    def _archive_current(cls, folder: Path) -> None:
        existing = [folder / name for name in cls.FILES if (folder / name).is_file()]
        if not existing:
            return
        archive = folder / "history" / _stamp()
        archive.mkdir(parents=True, exist_ok=True)
        for path in existing:
            shutil.copy2(path, archive / path.name)


class StyleReviewService:
    def __init__(self, project_root: str | Path, auditor: StyleAuditor | None = None, editor: StyleEditor | None = None, relationship_auditor: RelationshipAuditor | None = None) -> None:
        self.project_root = Path(project_root).resolve()
        self.artifacts = ArtifactStore(self.project_root)
        self.versions = ChapterVersionStore(self.project_root)
        self.settings_store = StyleReviewSettingsStore(self.project_root)
        self.reviews = StyleReviewStore(self.project_root)
        self.relationships = RelationshipStateStore(self.project_root)
        self.auditor = auditor
        self.editor = editor
        self.relationship_auditor = relationship_auditor

    def public(self, chapter_number: int) -> dict[str, Any]:
        self._validate_chapter(chapter_number)
        settings = self.settings_store.initialize()
        available = {kind: self._has_source(chapter_number, kind) for kind in ("draft", "author", "final")}
        current_source = ""
        current_author = self._source(chapter_number, "author") if available["author"] else None
        report = self.reviews.report(chapter_number, current_author=current_author)
        source_kind = str(report.get("source_kind", ""))
        if source_kind in available and available[source_kind]:
            current_source = self._source(chapter_number, source_kind)
            report = self.reviews.report(chapter_number, current_source, current_author)
        elif source_kind:
            report["source_stale"] = True
        return {"settings": settings, "available_sources": available, "review": report}

    def audit_chapter(self, chapter_number: int, source_kind: str) -> dict[str, Any]:
        if self.auditor is None:
            raise RuntimeError("文风检测智能体未配置")
        self._validate_chapter(chapter_number)
        source_text = self._source(chapter_number, source_kind)
        chapter_plan = self._chapter_plan(chapter_number)
        findings = ChapterWriter._quality_issues(source_text, chapter_plan)
        report = self._audit_text(chapter_number, source_kind, source_text, chapter_plan, findings)
        report.update({"chapter_number": chapter_number, "source_kind": source_kind, "created_at": _now()})
        author_text = self._source(chapter_number, "author") if self._has_source(chapter_number, "author") else None
        self.reviews.begin(chapter_number, source_kind, source_text, report, author_text)
        return self.public(chapter_number)

    def revise_chapter(self, chapter_number: int, issue_ids: list[str] | None = None, rounds: int | None = None, second_audit: bool | None = None) -> dict[str, Any]:
        if self.editor is None or self.auditor is None:
            raise RuntimeError("文风检测或改稿智能体未配置")
        settings = self.settings_store.initialize()
        report = self.public(chapter_number)["review"]
        if report.get("status") == "not_started":
            raise FileNotFoundError("请先运行正文检测")
        if report.get("status") == "accepted":
            raise ValueError("本轮候选稿已经接受，请重新检测后再生成修改稿")
        if report.get("source_stale") or report.get("author_stale"):
            raise ValueError("检测后原稿或作者稿已经变化，请重新检测")
        audit = self.reviews.audit(chapter_number)
        selected = self._select_issues(audit.get("issues", []), issue_ids, settings["minimum_severity"])
        if not selected:
            raise ValueError("没有符合条件的修改问题")
        source_kind = str(report.get("source_kind", "draft"))
        source_text = self.reviews.source_text(chapter_number)
        chapter_plan = self._chapter_plan(chapter_number)
        total_rounds = max(1, min(int(rounds or settings["auto_revision_rounds"]), 2))
        should_recheck = settings["second_audit"] if second_audit is None else bool(second_audit)
        current_text = source_text
        current_issues = selected
        for round_number in range(1, total_rounds + 1):
            candidate = self.editor.run(self.artifacts.load_brief(), chapter_number, source_kind, current_text, chapter_plan, current_issues, self._relationship_context(chapter_plan), self._relationship_beat(chapter_number))
            revision_audit = None
            if should_recheck:
                findings = ChapterWriter._quality_issues(candidate, chapter_plan)
                revision_audit = self._audit_text(chapter_number, "candidate", candidate, chapter_plan, findings)
                revision_audit.update({"chapter_number": chapter_number, "source_kind": "candidate", "created_at": _now()})
            self.reviews.save_candidate(chapter_number, candidate, [str(item["id"]) for item in current_issues], round_number, revision_audit)
            current_text = candidate
            if not revision_audit:
                break
            current_issues = self._select_issues(revision_audit.get("issues", []), None, settings["minimum_severity"])
            if not current_issues:
                break
        return self.public(chapter_number)

    def recheck_candidate(self, chapter_number: int) -> dict[str, Any]:
        if self.auditor is None:
            raise RuntimeError("文风检测智能体未配置")
        self._validate_chapter(chapter_number)
        review = self.public(chapter_number)["review"]
        if review.get("status") in {"accepted", "rejected"}:
            raise ValueError("本轮候选稿已经结束，请重新检测或生成新修改稿")
        if review.get("source_stale") or review.get("author_stale"):
            raise ValueError("检测后原稿或作者稿已经变化，请重新检测")
        candidate = self.reviews.candidate_text(chapter_number)
        chapter_plan = self._chapter_plan(chapter_number)
        findings = ChapterWriter._quality_issues(candidate, chapter_plan)
        report = self._audit_text(chapter_number, "candidate", candidate, chapter_plan, findings)
        report.update({"chapter_number": chapter_number, "source_kind": "candidate", "created_at": _now()})
        self.reviews.save_revision_audit(chapter_number, report)
        return self.public(chapter_number)

    def run_pipeline(self, chapter_number: int, source_kind: str = "draft") -> dict[str, Any]:
        settings = self.settings_store.initialize()
        if settings["mode"] == "off":
            return self.public(chapter_number)
        result = self.audit_chapter(chapter_number, source_kind)
        if settings["mode"] == "assisted" and settings["generate_revision_after_audit"] and result["review"].get("audit", {}).get("issues"):
            result = self.revise_chapter(chapter_number)
        return result

    def update_candidate(self, chapter_number: int, text: str) -> dict[str, Any]:
        self._validate_chapter(chapter_number)
        report = self.public(chapter_number)["review"]
        if report.get("status") in {"accepted", "rejected"}:
            raise ValueError("本轮候选稿已经结束，不能继续编辑")
        if report.get("source_stale") or report.get("author_stale"):
            raise ValueError("检测后原稿或作者稿已经变化，请重新检测")
        self.reviews.update_candidate(chapter_number, text)
        return self.public(chapter_number)

    def accept_candidate(self, chapter_number: int, text: str | None = None) -> dict[str, Any]:
        self._validate_chapter(chapter_number)
        report = self.public(chapter_number)["review"]
        if report.get("status") == "not_started":
            raise FileNotFoundError("该章节尚无修改稿")
        if report.get("status") == "accepted":
            raise ValueError("本轮候选稿已经接受")
        if report.get("status") == "rejected":
            raise ValueError("本轮候选稿已经拒绝，请先生成新修改稿")
        if report.get("source_stale") or report.get("author_stale"):
            raise ValueError("原稿或作者稿在检测后发生变化，请重新检测后再接受")
        if text is not None:
            self.reviews.update_candidate(chapter_number, text)
        candidate = self.reviews.candidate_text(chapter_number)
        path = self.versions.write_author(chapter_number, candidate)
        self.reviews.mark_accepted(chapter_number, candidate)
        return {"path": str(path), "style": self.public(chapter_number)}

    def reject_candidate(self, chapter_number: int) -> dict[str, Any]:
        self._validate_chapter(chapter_number)
        report = self.public(chapter_number)["review"]
        if report.get("status") == "accepted":
            raise ValueError("已经接受的候选稿不能再拒绝")
        if report.get("status") == "rejected":
            raise ValueError("本轮候选稿已经拒绝")
        self.reviews.candidate_text(chapter_number)
        self.reviews.mark_rejected(chapter_number)
        return self.public(chapter_number)

    def _source(self, chapter_number: int, source_kind: str) -> str:
        if source_kind not in {"draft", "author", "final"}:
            raise ValueError("source 必须是 draft、author 或 final")
        return self.versions.get_text(chapter_number, source_kind)

    def _has_source(self, chapter_number: int, source_kind: str) -> bool:
        try:
            self._source(chapter_number, source_kind)
            return True
        except FileNotFoundError:
            return False

    def _chapter_plan(self, chapter_number: int) -> dict[str, Any]:
        if not self.artifacts.has_chapter_plan(chapter_number):
            raise FileNotFoundError(f"第 {chapter_number} 章尚无章节规划")
        return self.artifacts.read_chapter_plan(chapter_number)

    def _audit_text(self, chapter_number: int, source_kind: str, source_text: str, chapter_plan: dict[str, Any], findings: list[str]) -> dict[str, Any]:
        if self.auditor is None:
            raise RuntimeError("文风检测智能体未配置")
        project = self.artifacts.load_brief()
        style_report = self.auditor.run(project, chapter_number, source_kind, source_text, chapter_plan, findings, self._style_baseline())
        relationship_report = RelationshipAuditor.pass_report()
        relationship_state = self.relationships.load()
        if self.relationship_auditor is not None and relationship_state["settings"]["enabled"] and relationship_state["settings"]["consistency_audit_enabled"]:
            relationship_report = self.relationship_auditor.run(project, chapter_number, source_kind, source_text, chapter_plan, self._relationship_context(chapter_plan), self._relationship_beat(chapter_number))
        return merge_audit_reports(style_report, relationship_report)

    def _style_baseline(self) -> dict[str, Any]:
        path = self.project_root / "style_review" / "baseline.json"
        if not path.is_file():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _relationship_context(self, chapter_plan: dict[str, Any]) -> dict[str, Any]:
        characters = [str(item) for item in chapter_plan.get("characters", []) if str(item).strip()]
        return self.relationships.context_for_characters(characters)

    def _relationship_beat(self, chapter_number: int) -> dict[str, Any]:
        if self.artifacts.has_relationship_beat(chapter_number):
            return self.artifacts.read_relationship_beat(chapter_number)
        return {}

    def _validate_chapter(self, chapter_number: int) -> None:
        count = self.artifacts.load_brief().chapter_count
        if chapter_number < 1 or chapter_number > count:
            raise ValueError(f"章节编号必须在 1 到 {count} 之间")

    @staticmethod
    def _select_issues(issues: list[dict[str, Any]], issue_ids: list[str] | None, minimum_severity: str) -> list[dict[str, Any]]:
        explicit_selection = issue_ids is not None
        ids = {str(item) for item in issue_ids or []}
        rank = {"low": 1, "medium": 2, "high": 3}
        threshold = rank[minimum_severity]
        selected = []
        for issue in issues:
            if explicit_selection and str(issue.get("id")) not in ids:
                continue
            if rank.get(str(issue.get("severity", "medium")), 2) < threshold:
                continue
            if str(issue.get("continuity_risk", "low")) == "high" and not explicit_selection:
                continue
            selected.append(issue)
        return selected


def _paragraph_diff(source: str, candidate: str) -> list[dict[str, str]]:
    before = [part.strip() for part in source.strip().split("\n\n") if part.strip()]
    after = [part.strip() for part in candidate.strip().split("\n\n") if part.strip()]
    matcher = difflib.SequenceMatcher(a=before, b=after)
    result: list[dict[str, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            result.extend({"type": tag, "original": before[i], "candidate": after[j]} for i, j in zip(range(i1, i2), range(j1, j2)))
        else:
            result.append({"type": tag, "original": "\n\n".join(before[i1:i2]), "candidate": "\n\n".join(after[j1:j2])})
    return result


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    _write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8") if path.is_file() else ""
    except OSError:
        return ""


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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
