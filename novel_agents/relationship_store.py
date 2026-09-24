from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RELATIONSHIP_STAGES = (
    "stranger",
    "acquainted",
    "familiar",
    "ambiguous",
    "dating",
    "distant",
    "broken",
    "reconciled",
)

RELATIONSHIP_STAGE_LABELS = {
    "stranger": "陌生",
    "acquainted": "相识",
    "familiar": "熟悉",
    "ambiguous": "暧昧",
    "dating": "相恋",
    "distant": "疏远",
    "broken": "决裂",
    "reconciled": "和解",
}

DEFAULT_RELATIONSHIP_SETTINGS: dict[str, Any] = {
    "enabled": True,
    "auto_update_after_finalize": True,
    "require_stage_confirmation": True,
    "schedule_proposals_enabled": True,
    "consistency_audit_enabled": True,
    "allow_regression": True,
    "max_stage_change_per_chapter": 1,
    "default_intensity": "subtle",
}

_DIRECTION_METRICS = ("trust", "attraction", "dependence", "guard")
_LINEAR_STAGE_RANK = {
    "stranger": 0,
    "acquainted": 1,
    "familiar": 2,
    "ambiguous": 3,
    "dating": 4,
    "distant": 2,
    "broken": 0,
    "reconciled": 3,
}


class RelationshipStateStore:
    _locks_guard = threading.Lock()
    _locks: dict[Path, threading.RLock] = {}

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.root = self.project_root / "relationships"
        self.path = self.root / "state.json"
        with self._locks_guard:
            self._lock = self._locks.setdefault(self.path, threading.RLock())

    def exists(self) -> bool:
        return self.path.is_file()

    def initialize(self, settings: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            if self.path.is_file():
                return self.load()
            state = {
                "version": 1,
                "settings": copy.deepcopy(DEFAULT_RELATIONSHIP_SETTINGS),
                "plan_rules": [],
                "pairs": [],
                "timeline": [],
                "updated_at": _now(),
            }
            if settings:
                state["settings"].update(settings)
            self._validate_settings(state["settings"])
            self._write(state)
            return state

    def load(self) -> dict[str, Any]:
        with self._lock:
            if not self.path.is_file():
                return self.initialize()
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                raise ValueError("relationships/state.json 必须是对象")
            data.setdefault("version", 1)
            settings = copy.deepcopy(DEFAULT_RELATIONSHIP_SETTINGS)
            if isinstance(data.get("settings"), dict):
                settings.update(data["settings"])
            self._validate_settings(settings)
            data["settings"] = settings
            data["plan_rules"] = _string_list(data.get("plan_rules", []))
            data["pairs"] = [
                self._normalize_pair(item)
                for item in data.get("pairs", [])
                if isinstance(item, dict)
            ]
            data["timeline"] = [
                item for item in data.get("timeline", []) if isinstance(item, dict)
            ]
            return data

    def public(self) -> dict[str, Any]:
        state = self.load()
        timeline = sorted(
            state["timeline"],
            key=lambda item: str(item.get("created_at", "")),
            reverse=True,
        )
        return {
            "enabled": bool(state["settings"]["enabled"]),
            "settings": state["settings"],
            "stage_order": list(RELATIONSHIP_STAGES),
            "stage_labels": dict(RELATIONSHIP_STAGE_LABELS),
            "plan_rules": state["plan_rules"],
            "pairs": state["pairs"],
            "timeline": timeline,
            "pending_count": sum(
                item.get("status") == "pending" for item in timeline
            ),
            "updated_at": state.get("updated_at", ""),
        }

    def update_settings(self, values: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            state = self.load()
            for key in DEFAULT_RELATIONSHIP_SETTINGS:
                if key in values:
                    state["settings"][key] = values[key]
            self._validate_settings(state["settings"])
            state["updated_at"] = _now()
            self._write(state)
            return state["settings"]

    def import_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            state = self.load()
            state["plan_rules"] = _string_list(plan.get("global_rules", []))
            existing_by_key = {
                _pair_key(pair["character_a"], pair["character_b"]): pair
                for pair in state["pairs"]
            }
            added = 0
            updated = 0
            for raw in plan.get("relationships", []):
                if not isinstance(raw, dict):
                    continue
                character_a = str(raw.get("character_a", "")).strip()
                character_b = str(raw.get("character_b", "")).strip()
                if not character_a or not character_b or character_a == character_b:
                    continue
                key = _pair_key(character_a, character_b)
                current = existing_by_key.get(key)
                if current is None:
                    pair = self._normalize_pair(raw)
                    state["pairs"].append(pair)
                    existing_by_key[key] = pair
                    added += 1
                    continue
                for field in (
                    "relationship_type",
                    "target_stage",
                    "next_milestone",
                    "milestones",
                    "stage_schedule",
                    "forbidden_leaps",
                    "genre_signals",
                    "arc_summary",
                ):
                    if field in raw and raw[field] not in (None, "", []):
                        current[field] = self._normalize_pair_field(field, raw[field])
                if "stage_schedule" not in raw and "milestones" in raw:
                    current["stage_schedule"] = self._schedule_from_pair_fields(current)
                updated += 1
            state["updated_at"] = _now()
            self._write(state)
            return {"added": added, "updated": updated, "pairs": len(state["pairs"])}

    def create_pair(self, values: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            state = self.load()
            pair = self._normalize_pair(values)
            key = _pair_key(pair["character_a"], pair["character_b"])
            if any(
                _pair_key(item["character_a"], item["character_b"]) == key
                for item in state["pairs"]
            ):
                raise ValueError("这组人物关系已经存在")
            state["pairs"].append(pair)
            state["updated_at"] = _now()
            self._write(state)
            return pair

    def update_pair(self, pair_id: str, values: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            state = self.load()
            pair = self._find_pair(state, pair_id)
            before = copy.deepcopy(pair)
            for field in (
                "character_a",
                "character_b",
                "relationship_type",
                "stage",
                "target_stage",
                "public_status",
                "private_status",
                "touch_boundary",
                "next_milestone",
                "arc_summary",
                "locked",
                "milestones",
                "stage_schedule",
                "forbidden_leaps",
                "genre_signals",
                "shared_secrets",
                "shared_experiences",
                "unresolved_tensions",
                "address_terms",
                "a_to_b",
                "b_to_a",
            ):
                if field in values:
                    pair[field] = self._normalize_pair_field(field, values[field])
            if "stage_schedule" in values or "milestones" in values:
                pair["stage_schedule"] = self._schedule_from_pair_fields(pair)
            if not pair["character_a"] or not pair["character_b"]:
                raise ValueError("关系双方姓名不能为空")
            if pair["character_a"] == pair["character_b"]:
                raise ValueError("关系双方不能是同一人物")
            current_key = _pair_key(pair["character_a"], pair["character_b"])
            if any(
                item["id"] != pair_id
                and _pair_key(item["character_a"], item["character_b"]) == current_key
                for item in state["pairs"]
            ):
                raise ValueError("这组人物关系已经存在")
            pair["updated_at"] = _now()
            if before != pair:
                state["timeline"].append(
                    {
                        "id": "manual-" + uuid.uuid4().hex[:12],
                        "kind": "manual_edit",
                        "pair_id": pair_id,
                        "chapter_number": 0,
                        "summary": "人工修改人物关系状态",
                        "evidence_quotes": [],
                        "status": "applied",
                        "proposed_stage": pair["stage"],
                        "stage_applied": before.get("stage") != pair.get("stage"),
                        "before_pair": before,
                        "after_pair": copy.deepcopy(pair),
                        "created_at": _now(),
                    }
                )
            state["updated_at"] = _now()
            self._write(state)
            return pair

    def delete_pair(self, pair_id: str) -> dict[str, Any]:
        with self._lock:
            state = self.load()
            pair = self._find_pair(state, pair_id)
            state["pairs"] = [item for item in state["pairs"] if item["id"] != pair_id]
            state["timeline"] = [
                item for item in state["timeline"] if item.get("pair_id") != pair_id
            ]
            state["updated_at"] = _now()
            self._write(state)
            return pair

    def context_for_characters(
        self, characters: list[str] | tuple[str, ...] | None = None
    ) -> dict[str, Any]:
        state = self.load()
        names = {str(item).strip() for item in characters or [] if str(item).strip()}
        pairs = [
            pair
            for pair in state["pairs"]
            if not names
            or pair["character_a"] in names
            or pair["character_b"] in names
        ]
        if names and not pairs and len(state["pairs"]) <= 5:
            pairs = list(state["pairs"])
        contexts = []
        for pair in pairs[:8]:
            recent = [
                self._event_context(item)
                for item in reversed(state["timeline"])
                if item.get("pair_id") == pair["id"]
                and item.get("status") in {"applied", "pending"}
            ][:3]
            contexts.append(
                {
                    "id": pair["id"],
                    "character_a": pair["character_a"],
                    "character_b": pair["character_b"],
                    "relationship_type": pair["relationship_type"],
                    "stage": pair["stage"],
                    "stage_label": RELATIONSHIP_STAGE_LABELS[pair["stage"]],
                    "target_stage": pair["target_stage"],
                    "public_status": pair["public_status"],
                    "private_status": pair["private_status"],
                    "a_to_b": pair["a_to_b"],
                    "b_to_a": pair["b_to_a"],
                    "address_terms": pair["address_terms"],
                    "touch_boundary": pair["touch_boundary"],
                    "shared_secrets": pair["shared_secrets"][-6:],
                    "shared_experiences": pair["shared_experiences"][-6:],
                    "unresolved_tensions": pair["unresolved_tensions"][-6:],
                    "next_milestone": pair["next_milestone"],
                    "forbidden_leaps": pair["forbidden_leaps"],
                    "locked": pair["locked"],
                    "consistency_warning": pair["consistency_warning"],
                    "recent_events": recent,
                }
            )
        return {
            "settings": state["settings"],
            "plan_rules": state["plan_rules"],
            "pairs": contexts,
        }

    def apply_extraction(
        self,
        chapter_number: int,
        source_hash: str,
        extraction: dict[str, Any],
    ) -> dict[str, Any]:
        with self._lock:
            state = self.load()
            existing_keys = {
                (int(item.get("chapter_number", 0)), str(item.get("source_hash", "")), str(item.get("pair_id", "")))
                for item in state["timeline"]
                if item.get("kind") == "chapter_update"
            }
            applied = 0
            pending = 0
            ignored = 0
            processed_pairs: set[str] = set()
            for raw in extraction.get("events", []):
                if not isinstance(raw, dict):
                    ignored += 1
                    continue
                pair = self._resolve_event_pair(state, raw)
                if pair is None or pair["id"] in processed_pairs:
                    ignored += 1
                    continue
                dedupe_key = (chapter_number, source_hash, pair["id"])
                if dedupe_key in existing_keys:
                    ignored += 1
                    continue
                processed_pairs.add(pair["id"])
                before = copy.deepcopy(pair)
                deltas = self._normalize_deltas(raw.get("deltas", {}))
                self._apply_deltas(pair, deltas)
                self._apply_event_details(pair, raw)
                pair["consistency_warning"] = ""
                proposed_stage = _normalize_stage(raw.get("proposed_stage", ""), "")
                stage_pending = False
                stage_applied = False
                if proposed_stage and proposed_stage != pair["stage"]:
                    if (
                        pair["locked"]
                        or state["settings"]["require_stage_confirmation"]
                        or not self._can_auto_transition(
                            pair["stage"], proposed_stage, state["settings"]
                        )
                    ):
                        stage_pending = True
                    else:
                        pair["stage"] = proposed_stage
                        stage_applied = True
                pair["updated_chapter"] = max(
                    int(pair.get("updated_chapter", 0)), chapter_number
                )
                pair["updated_at"] = _now()
                status = "pending" if stage_pending else "applied"
                event = {
                    "id": "rel-" + uuid.uuid4().hex[:12],
                    "kind": "chapter_update",
                    "pair_id": pair["id"],
                    "chapter_number": chapter_number,
                    "source_hash": source_hash,
                    "summary": str(raw.get("summary", "")).strip()
                    or "本章出现人物关系变化",
                    "evidence_quotes": _string_list(raw.get("evidence_quotes", []))[:5],
                    "deltas": deltas,
                    "status": status,
                    "proposed_stage": proposed_stage,
                    "stage_applied": stage_applied,
                    "confidence": _clamp_float(raw.get("confidence", 0.7), 0.0, 1.0),
                    "before_pair": before,
                    "after_pair": copy.deepcopy(pair),
                    "created_at": _now(),
                }
                state["timeline"].append(event)
                if status == "pending":
                    pending += 1
                else:
                    applied += 1
            state["updated_at"] = _now()
            self._write(state)
            return {
                "chapter_number": chapter_number,
                "applied": applied,
                "pending": pending,
                "ignored": ignored,
            }

    def apply_schedule_proposals(self, chapter_number: int) -> dict[str, Any]:
        """S2: at/after a planned chapter, enqueue pending stage switches for confirmation."""
        chapter_number = int(chapter_number)
        if chapter_number < 1:
            raise ValueError("章节编号必须是正整数")
        with self._lock:
            state = self.load()
            if not state["settings"].get("enabled", True):
                return {
                    "chapter_number": chapter_number,
                    "proposed": 0,
                    "skipped": 0,
                    "event_ids": [],
                }
            if not state["settings"].get("schedule_proposals_enabled", True):
                return {
                    "chapter_number": chapter_number,
                    "proposed": 0,
                    "skipped": 0,
                    "event_ids": [],
                }

            proposed = 0
            skipped = 0
            event_ids: list[str] = []
            for pair in state["pairs"]:
                schedule = self._schedule_from_pair_fields(pair)
                pair["stage_schedule"] = schedule
                if pair.get("locked"):
                    skipped += 1
                    continue
                entry = self._next_schedule_proposal(
                    pair, chapter_number, state["settings"]
                )
                if entry is None:
                    skipped += 1
                    continue
                target_stage = str(entry["stage"])
                if self._has_open_stage_proposal(
                    state, pair["id"], target_stage, chapter_number
                ):
                    skipped += 1
                    continue

                before = copy.deepcopy(pair)
                pair["updated_chapter"] = max(
                    int(pair.get("updated_chapter", 0)), chapter_number
                )
                pair["updated_at"] = _now()
                pair["next_milestone"] = self._next_milestone_text(
                    schedule, chapter_number, pair.get("next_milestone", "")
                )
                event_id = "sched-" + uuid.uuid4().hex[:12]
                summary = str(entry.get("summary", "")).strip() or (
                    f"计划阶段：第 {entry['from_chapter']} 章起建议切换为"
                    f"{RELATIONSHIP_STAGE_LABELS.get(target_stage, target_stage)}"
                )
                event = {
                    "id": event_id,
                    "kind": "schedule_proposal",
                    "pair_id": pair["id"],
                    "chapter_number": chapter_number,
                    "source_hash": "",
                    "summary": summary,
                    "evidence_quotes": _string_list(entry.get("required_evidence", [])),
                    "deltas": self._normalize_deltas({}),
                    "status": "pending",
                    "proposed_stage": target_stage,
                    "stage_applied": False,
                    "confidence": 1.0,
                    "schedule_from_chapter": int(entry["from_chapter"]),
                    "before_pair": before,
                    "after_pair": copy.deepcopy(pair),
                    "created_at": _now(),
                }
                state["timeline"].append(event)
                event_ids.append(event_id)
                proposed += 1

            state["updated_at"] = _now()
            self._write(state)
            return {
                "chapter_number": chapter_number,
                "proposed": proposed,
                "skipped": skipped,
                "event_ids": event_ids,
            }

    def mark_chapter_stale(self, chapter_number: int, reason: str) -> dict[str, Any]:
        with self._lock:
            state = self.load()
            affected = 0
            restored = 0
            for event in state["timeline"]:
                if int(event.get("chapter_number", 0) or 0) != chapter_number:
                    continue
                if event.get("kind") != "chapter_update" or event.get("status") not in {
                    "applied",
                    "pending",
                }:
                    continue
                pair_id = str(event.get("pair_id", ""))
                later = [
                    item
                    for item in state["timeline"]
                    if item.get("pair_id") == pair_id
                    and int(item.get("chapter_number", 0) or 0) > chapter_number
                    and item.get("status") in {"applied", "pending"}
                ]
                if not later and isinstance(event.get("before_pair"), dict):
                    before = self._normalize_pair(event["before_pair"])
                    state["pairs"] = [
                        before if pair["id"] == pair_id else pair
                        for pair in state["pairs"]
                    ]
                    restored += 1
                else:
                    pair = next(
                        (item for item in state["pairs"] if item["id"] == pair_id),
                        None,
                    )
                    if pair is not None:
                        pair["consistency_warning"] = (
                            f"第 {chapter_number} 章定稿已变化，后续关系记录需要人工复核"
                        )
                event["status"] = "stale"
                event["stale_reason"] = reason
                event["stale_at"] = _now()
                affected += 1
            if affected:
                state["updated_at"] = _now()
                self._write(state)
            return {"affected": affected, "restored": restored}

    def decide_event(self, event_id: str, decision: str) -> dict[str, Any]:
        if decision not in {"accept", "reject"}:
            raise ValueError("decision 必须是 accept 或 reject")
        with self._lock:
            state = self.load()
            event = self._find_event(state, event_id)
            if event.get("status") != "pending":
                raise ValueError("该关系阶段建议已经处理")
            pair = self._find_pair(state, str(event.get("pair_id", "")))
            if decision == "accept":
                proposed = _normalize_stage(event.get("proposed_stage", ""), "")
                if not proposed:
                    raise ValueError("该事件没有可接受的阶段建议")
                pair["stage"] = proposed
                pair["updated_at"] = _now()
                event["status"] = "applied"
                event["stage_applied"] = True
                event["after_pair"] = copy.deepcopy(pair)
            else:
                event["status"] = "rejected"
                event["stage_applied"] = False
            event["decided_at"] = _now()
            state["updated_at"] = _now()
            self._write(state)
            return event

    def rollback_event(self, event_id: str) -> dict[str, Any]:
        with self._lock:
            state = self.load()
            event = self._find_event(state, event_id)
            if event.get("status") not in {"applied", "pending"}:
                raise ValueError("该事件不能回退")
            pair_id = str(event.get("pair_id", ""))
            active_for_pair = [
                item
                for item in state["timeline"]
                if item.get("pair_id") == pair_id
                and item.get("status") in {"applied", "pending"}
            ]
            if active_for_pair and active_for_pair[-1].get("id") != event_id:
                raise ValueError("只能回退该人物组合最新的一条有效关系记录")
            before = event.get("before_pair")
            if not isinstance(before, dict):
                raise ValueError("该事件缺少可回退快照")
            restored = self._normalize_pair(before)
            state["pairs"] = [
                restored if pair["id"] == pair_id else pair for pair in state["pairs"]
            ]
            event["status"] = "rolled_back"
            event["rolled_back_at"] = _now()
            state["updated_at"] = _now()
            self._write(state)
            return event

    def _normalize_pair(self, values: dict[str, Any]) -> dict[str, Any]:
        character_a = str(values.get("character_a", "")).strip()
        character_b = str(values.get("character_b", "")).strip()
        pair_id = str(values.get("id", "")).strip() or _pair_id(character_a, character_b)
        initial_stage = values.get("stage", values.get("initial_stage", "stranger"))
        pair = {
            "id": pair_id,
            "character_a": character_a,
            "character_b": character_b,
            "relationship_type": str(values.get("relationship_type", "romance")).strip()
            or "romance",
            "stage": _normalize_stage(initial_stage, "stranger"),
            "target_stage": _normalize_stage(values.get("target_stage", "dating"), "dating"),
            "public_status": str(values.get("public_status", "")).strip(),
            "private_status": str(values.get("private_status", "")).strip(),
            "a_to_b": _normalize_direction(values.get("a_to_b", {})),
            "b_to_a": _normalize_direction(values.get("b_to_a", {})),
            "address_terms": _normalize_address(values.get("address_terms", {})),
            "touch_boundary": str(values.get("touch_boundary", "")).strip(),
            "shared_secrets": _string_list(values.get("shared_secrets", [])),
            "shared_experiences": _string_list(values.get("shared_experiences", [])),
            "unresolved_tensions": _string_list(values.get("unresolved_tensions", [])),
            "milestones": _object_list(values.get("milestones", [])),
            "stage_schedule": [],
            "next_milestone": str(values.get("next_milestone", "")).strip(),
            "forbidden_leaps": _string_list(values.get("forbidden_leaps", [])),
            "genre_signals": _string_list(values.get("genre_signals", [])),
            "arc_summary": str(values.get("arc_summary", "")).strip(),
            "locked": bool(values.get("locked", False)),
            "consistency_warning": str(values.get("consistency_warning", "")).strip(),
            "updated_chapter": max(0, int(values.get("updated_chapter", 0) or 0)),
            "updated_at": str(values.get("updated_at", "")).strip() or _now(),
        }
        pair["stage_schedule"] = self._schedule_from_pair_fields(
            {
                **pair,
                "stage_schedule": values.get("stage_schedule", []),
                "milestones": pair["milestones"],
            }
        )
        return pair

    def _normalize_pair_field(self, field: str, value: Any) -> Any:
        if field in {"stage", "target_stage"}:
            return _normalize_stage(value, "stranger" if field == "stage" else "dating")
        if field in {
            "character_a",
            "character_b",
            "relationship_type",
            "public_status",
            "private_status",
            "touch_boundary",
            "next_milestone",
            "arc_summary",
        }:
            return str(value).strip()
        if field == "locked":
            return bool(value)
        if field in {
            "forbidden_leaps",
            "genre_signals",
            "shared_secrets",
            "shared_experiences",
            "unresolved_tensions",
        }:
            return _string_list(value)
        if field == "milestones":
            return _object_list(value)
        if field == "stage_schedule":
            return _normalize_stage_schedule(value)
        if field in {"a_to_b", "b_to_a"}:
            return _normalize_direction(value)
        if field == "address_terms":
            return _normalize_address(value)
        return value

    @staticmethod
    def _normalize_deltas(value: Any) -> dict[str, dict[str, int]]:
        source = value if isinstance(value, dict) else {}
        result: dict[str, dict[str, int]] = {}
        for direction in ("a_to_b", "b_to_a"):
            raw = source.get(direction, {})
            raw = raw if isinstance(raw, dict) else {}
            result[direction] = {
                metric: _clamp_int(raw.get(metric, 0), -2, 2)
                for metric in _DIRECTION_METRICS
            }
        return result

    @staticmethod
    def _apply_deltas(pair: dict[str, Any], deltas: dict[str, dict[str, int]]) -> None:
        for direction in ("a_to_b", "b_to_a"):
            for metric in _DIRECTION_METRICS:
                pair[direction][metric] = _clamp_int(
                    int(pair[direction].get(metric, 0))
                    + int(deltas[direction].get(metric, 0)),
                    0,
                    5,
                )

    @staticmethod
    def _apply_event_details(pair: dict[str, Any], raw: dict[str, Any]) -> None:
        perceptions = raw.get("perceptions", {})
        if isinstance(perceptions, dict):
            for direction in ("a_to_b", "b_to_a"):
                text = str(perceptions.get(direction, "")).strip()
                if text:
                    pair[direction]["perception"] = text
        addresses = raw.get("address_terms", {})
        if isinstance(addresses, dict):
            for direction in ("a_to_b", "b_to_a"):
                text = str(addresses.get(direction, "")).strip()
                if text:
                    pair["address_terms"][direction] = text
        for field in ("public_status", "private_status", "touch_boundary", "next_milestone"):
            text = str(raw.get(field, "")).strip()
            if text:
                pair[field] = text
        for field in ("shared_secrets", "shared_experiences", "unresolved_tensions"):
            pair[field] = _merge_unique(pair[field], _string_list(raw.get(field, [])), 30)
        resolved = set(_string_list(raw.get("resolved_tensions", [])))
        if resolved:
            pair["unresolved_tensions"] = [
                item for item in pair["unresolved_tensions"] if item not in resolved
            ]

    @staticmethod
    def _can_auto_transition(current: str, proposed: str, settings: dict[str, Any]) -> bool:
        if proposed == current:
            return True
        current_rank = _LINEAR_STAGE_RANK[current]
        proposed_rank = _LINEAR_STAGE_RANK[proposed]
        if proposed_rank < current_rank and not settings["allow_regression"]:
            return False
        return abs(proposed_rank - current_rank) <= int(
            settings["max_stage_change_per_chapter"]
        )

    @classmethod
    def _schedule_from_pair_fields(cls, pair: dict[str, Any]) -> list[dict[str, Any]]:
        explicit = _normalize_stage_schedule(pair.get("stage_schedule", []))
        if explicit:
            return explicit
        return _schedule_from_milestones(pair.get("milestones", []))

    @classmethod
    def _next_schedule_proposal(
        cls,
        pair: dict[str, Any],
        chapter_number: int,
        settings: dict[str, Any],
    ) -> dict[str, Any] | None:
        schedule = _normalize_stage_schedule(pair.get("stage_schedule", []))
        if not schedule:
            schedule = _schedule_from_milestones(pair.get("milestones", []))
        due = [item for item in schedule if int(item["from_chapter"]) <= chapter_number]
        if not due:
            return None
        current = str(pair.get("stage", "stranger"))
        desired_entry = due[-1]
        desired = str(desired_entry["stage"])
        if desired == current:
            return None
        # Prefer the schedule's current target when it is only one legal step away.
        if cls._can_auto_transition(current, desired, settings):
            return desired_entry
        current_rank = _LINEAR_STAGE_RANK[current]
        desired_rank = _LINEAR_STAGE_RANK[desired]
        # Catch up one adjacent step toward the latest due stage; never jump backward
        # to earlier schedule rows just because they are still in the due list.
        for entry in due:
            target = str(entry["stage"])
            if target == current:
                continue
            if not cls._can_auto_transition(current, target, settings):
                continue
            target_rank = _LINEAR_STAGE_RANK[target]
            moving_forward = (
                desired_rank >= current_rank and current_rank < target_rank <= desired_rank
            )
            moving_back = (
                desired_rank < current_rank and desired_rank <= target_rank < current_rank
            )
            if moving_forward or moving_back:
                return entry
        return None

    @staticmethod
    def _has_open_stage_proposal(
        state: dict[str, Any],
        pair_id: str,
        proposed_stage: str,
        chapter_number: int,
    ) -> bool:
        for event in state.get("timeline", []):
            if event.get("pair_id") != pair_id:
                continue
            if event.get("status") not in {"pending", "applied"}:
                continue
            if str(event.get("proposed_stage", "")) != proposed_stage:
                continue
            kind = str(event.get("kind", ""))
            if kind == "schedule_proposal":
                return True
            if kind == "chapter_update" and int(event.get("chapter_number", 0) or 0) == chapter_number:
                return True
            if event.get("status") == "pending":
                return True
        return False

    @staticmethod
    def _next_milestone_text(
        schedule: list[dict[str, Any]],
        chapter_number: int,
        fallback: str,
    ) -> str:
        future = [
            item
            for item in schedule
            if int(item["from_chapter"]) > chapter_number
        ]
        if not future:
            return fallback
        item = future[0]
        label = RELATIONSHIP_STAGE_LABELS.get(item["stage"], item["stage"])
        return str(item.get("summary", "")).strip() or (
            f"第 {item['from_chapter']} 章：{label}"
        )

    @staticmethod
    def _event_context(event: dict[str, Any]) -> dict[str, Any]:
        return {
            "chapter_number": int(event.get("chapter_number", 0) or 0),
            "summary": str(event.get("summary", "")),
            "evidence_quotes": _string_list(event.get("evidence_quotes", []))[:3],
            "status": str(event.get("status", "")),
            "proposed_stage": str(event.get("proposed_stage", "")),
        }

    @staticmethod
    def _resolve_event_pair(
        state: dict[str, Any], raw: dict[str, Any]
    ) -> dict[str, Any] | None:
        pair_id = str(raw.get("pair_id", "")).strip()
        if pair_id:
            return next((pair for pair in state["pairs"] if pair["id"] == pair_id), None)
        character_a = str(raw.get("character_a", "")).strip()
        character_b = str(raw.get("character_b", "")).strip()
        if not character_a or not character_b:
            return None
        key = _pair_key(character_a, character_b)
        return next(
            (
                pair
                for pair in state["pairs"]
                if _pair_key(pair["character_a"], pair["character_b"]) == key
            ),
            None,
        )

    @staticmethod
    def _find_pair(state: dict[str, Any], pair_id: str) -> dict[str, Any]:
        pair = next((item for item in state["pairs"] if item["id"] == pair_id), None)
        if pair is None:
            raise KeyError(f"未知人物关系：{pair_id}")
        return pair

    @staticmethod
    def _find_event(state: dict[str, Any], event_id: str) -> dict[str, Any]:
        event = next(
            (item for item in state["timeline"] if item.get("id") == event_id),
            None,
        )
        if event is None:
            raise KeyError(f"未知关系事件：{event_id}")
        return event

    @staticmethod
    def _validate_settings(settings: dict[str, Any]) -> None:
        for field in (
            "enabled",
            "auto_update_after_finalize",
            "require_stage_confirmation",
            "schedule_proposals_enabled",
            "consistency_audit_enabled",
            "allow_regression",
        ):
            if field not in settings and field == "schedule_proposals_enabled":
                settings[field] = True
            if not isinstance(settings.get(field), bool):
                raise ValueError(f"relationship settings.{field} 必须是布尔值")
        maximum = int(settings.get("max_stage_change_per_chapter", 1))
        if maximum < 0 or maximum > 2:
            raise ValueError("max_stage_change_per_chapter 必须在 0 到 2 之间")
        settings["max_stage_change_per_chapter"] = maximum
        if settings.get("default_intensity") not in {
            "hold",
            "subtle",
            "clear",
            "milestone",
        }:
            raise ValueError("default_intensity 无效")

    def _write(self, data: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=".relationship.", suffix=".tmp", dir=self.root
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
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


def _normalize_stage_schedule(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    entries: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, dict):
            continue
        stage = _normalize_stage(
            raw.get("stage", raw.get("to_stage", raw.get("target_stage", ""))),
            "",
        )
        if not stage:
            continue
        chapter = _extract_schedule_chapter(raw)
        if chapter is None or chapter < 1:
            continue
        mode = str(raw.get("mode", "confirm")).strip().lower() or "confirm"
        if mode not in {"confirm", "auto", "hold"}:
            mode = "confirm"
        # S2 always surfaces confirmations; auto is reserved for a later mode.
        if mode == "auto":
            mode = "confirm"
        entries.append(
            {
                "from_chapter": chapter,
                "stage": stage,
                "mode": mode,
                "summary": str(
                    raw.get("summary", raw.get("trigger", raw.get("label", "")))
                ).strip(),
                "required_evidence": _string_list(
                    raw.get("required_evidence", raw.get("required_evidences", []))
                ),
            }
        )
    entries.sort(key=lambda item: (int(item["from_chapter"]), item["stage"]))
    # Keep one entry per chapter (last write wins after sort stability).
    deduped: dict[int, dict[str, Any]] = {}
    for item in entries:
        deduped[int(item["from_chapter"])] = item
    return [deduped[key] for key in sorted(deduped)]


def _schedule_from_milestones(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    converted: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, dict):
            continue
        stage = _normalize_stage(
            raw.get("stage", raw.get("to_stage", raw.get("target_stage", ""))),
            "",
        )
        if not stage:
            # Allow milestones that only describe events without stage labels.
            continue
        chapter = _extract_schedule_chapter(raw)
        if chapter is None:
            continue
        converted.append(
            {
                "from_chapter": chapter,
                "stage": stage,
                "mode": "confirm",
                "summary": str(
                    raw.get(
                        "summary",
                        raw.get("trigger", raw.get("observable_change", "")),
                    )
                ).strip(),
                "required_evidence": _string_list(
                    raw.get("required_evidence", raw.get("required_evidences", []))
                ),
            }
        )
    return _normalize_stage_schedule(converted)


def _extract_schedule_chapter(raw: dict[str, Any]) -> int | None:
    for key in ("from_chapter", "chapter", "chapter_number", "at_chapter"):
        if key in raw and raw[key] not in (None, ""):
            try:
                return max(1, int(raw[key]))
            except (TypeError, ValueError):
                return None
    chapter_range = raw.get("chapter_range")
    if isinstance(chapter_range, (list, tuple)) and chapter_range:
        try:
            return max(1, int(chapter_range[0]))
        except (TypeError, ValueError):
            return None
    if isinstance(chapter_range, str) and chapter_range.strip():
        text = chapter_range.strip().replace("～", "-").replace("~", "-")
        for sep in ("-", "—", "至", ","):
            if sep in text:
                head = text.split(sep, 1)[0]
                digits = "".join(ch for ch in head if ch.isdigit())
                if digits:
                    return max(1, int(digits))
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits:
            return max(1, int(digits))
    return None


def _normalize_stage(value: Any, default: str) -> str:
    normalized = str(value or "").strip().lower()
    aliases = {
        "陌生": "stranger",
        "陌生人": "stranger",
        "相识": "acquainted",
        "认识": "acquainted",
        "熟悉": "familiar",
        "朋友": "familiar",
        "暧昧": "ambiguous",
        "相恋": "dating",
        "恋人": "dating",
        "交往": "dating",
        "疏远": "distant",
        "决裂": "broken",
        "和解": "reconciled",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in RELATIONSHIP_STAGES else default


def _normalize_direction(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    return {
        "trust": _clamp_int(source.get("trust", 0), 0, 5),
        "attraction": _clamp_int(source.get("attraction", 0), 0, 5),
        "dependence": _clamp_int(source.get("dependence", 0), 0, 5),
        "guard": _clamp_int(source.get("guard", 3), 0, 5),
        "perception": str(source.get("perception", "")).strip(),
    }


def _normalize_address(value: Any) -> dict[str, str]:
    source = value if isinstance(value, dict) else {}
    return {
        "a_to_b": str(source.get("a_to_b", "")).strip(),
        "b_to_a": str(source.get("b_to_a", "")).strip(),
    }


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        raw = value.replace("\r", "\n").replace("；", "\n").split("\n")
    elif isinstance(value, (list, tuple, set)):
        raw = list(value)
    else:
        raw = []
    result: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def _object_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _merge_unique(existing: list[str], added: list[str], limit: int) -> list[str]:
    merged = list(existing)
    for item in added:
        if item not in merged:
            merged.append(item)
    return merged[-limit:]


def _pair_key(character_a: str, character_b: str) -> tuple[str, str]:
    return tuple(sorted((character_a.strip().casefold(), character_b.strip().casefold())))


def _pair_id(character_a: str, character_b: str) -> str:
    digest = hashlib.sha1(
        (character_a.strip() + "\0" + character_b.strip()).encode("utf-8")
    ).hexdigest()[:12]
    return "pair-" + digest


def _clamp_int(value: Any, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = minimum
    return max(minimum, min(maximum, number))


def _clamp_float(value: Any, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = minimum
    return max(minimum, min(maximum, number))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
