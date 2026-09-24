from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


PROFILE_FIELDS = {
    "model_profile_id",
    "model",
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "max_tokens",
    "seed",
    "reasoning_effort",
    "use_json_response_format",
}


DEFAULT_MEMORY_ACCESS = {
    "enabled": True,
    "read_story_facts": True,
    "read_author_style": False,
    "read_raw_passages": True,
    "read_foreshadowing": True,
    "top_k": None,
    "character_filter": True,
    "timeline_filter": True,
}


class AgentProfileStore:
    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()
        self.local_dir = self.workspace / ".novel_agents"
        self.path = self.local_dir / "agent_profiles.json"

    def get(self, agent_id: str) -> dict[str, Any]:
        stored = self._load().get(agent_id, {})
        profile = {field: stored.get(field) for field in PROFILE_FIELDS}
        profile["memory_access"] = dict(DEFAULT_MEMORY_ACCESS)
        if isinstance(stored.get("memory_access"), dict):
            profile["memory_access"].update(stored["memory_access"])
        profile["agent_id"] = agent_id
        return profile

    def update(self, agent_id: str, values: dict[str, Any]) -> dict[str, Any]:
        data = self._load()
        profile = data.setdefault(agent_id, {})
        for field in PROFILE_FIELDS:
            if field in values:
                profile[field] = _nullable(values[field])
        if "memory_access" in values:
            if not isinstance(values["memory_access"], dict):
                raise ValueError("memory_access 必须是对象")
            memory_access = profile.setdefault("memory_access", {})
            memory_access.update(
                {
                    key: value
                    for key, value in values["memory_access"].items()
                    if key in DEFAULT_MEMORY_ACCESS
                }
            )
        self._validate(profile)
        self._write(data)
        return self.get(agent_id)

    def model_overrides(self, agent_id: str) -> dict[str, Any]:
        profile = self.get(agent_id)
        return {field: profile[field] for field in PROFILE_FIELDS if profile.get(field) is not None}

    def memory_access_overrides(self, agent_id: str) -> dict[str, Any]:
        stored = self._load().get(agent_id, {})
        memory_access = stored.get("memory_access", {})
        if not isinstance(memory_access, dict):
            return {}
        return {
            key: value
            for key, value in memory_access.items()
            if key in DEFAULT_MEMORY_ACCESS
        }

    def all_model_overrides(self, agent_ids: list[str]) -> dict[str, dict[str, Any]]:
        return {agent_id: self.model_overrides(agent_id) for agent_id in agent_ids}

    def model_profile_references(self, model_profile_id: str) -> list[str]:
        return sorted(
            agent_id
            for agent_id, profile in self._load().items()
            if isinstance(profile, dict)
            and profile.get("model_profile_id") == model_profile_id
        )

    def migrate_model_references(
        self, model_ids_by_name: dict[str, list[str]]
    ) -> int:
        data = self._load()
        changed = 0
        for profile in data.values():
            if not isinstance(profile, dict) or profile.get("model_profile_id"):
                continue
            legacy_model = str(profile.get("model") or "").strip()
            matches = model_ids_by_name.get(legacy_model, [])
            if legacy_model and len(matches) == 1:
                profile["model_profile_id"] = matches[0]
                profile["model"] = None
                changed += 1
        if changed:
            self._write(data)
        return changed

    def delete(self, agent_id: str) -> None:
        data = self._load()
        if agent_id not in data:
            return
        data.pop(agent_id, None)
        self._write(data)

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}

    def _write(self, data: dict[str, Any]) -> None:
        self.local_dir.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=".profiles.", suffix=".tmp", dir=self.local_dir)
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

    @staticmethod
    def _validate(profile: dict[str, Any]) -> None:
        model_profile_id = profile.get("model_profile_id")
        if model_profile_id is not None and not str(model_profile_id).strip():
            raise ValueError("model_profile_id 不能为空字符串")
        temperature = profile.get("temperature")
        if temperature is not None and not 0 <= float(temperature) <= 2:
            raise ValueError("temperature 必须在 0 到 2 之间")
        top_p = profile.get("top_p")
        if top_p is not None and not 0 <= float(top_p) <= 1:
            raise ValueError("top_p 必须在 0 到 1 之间")
        for name in ("frequency_penalty", "presence_penalty"):
            value = profile.get(name)
            if value is not None and not -2 <= float(value) <= 2:
                raise ValueError(f"{name} 必须在 -2 到 2 之间")
        max_tokens = profile.get("max_tokens")
        if max_tokens is not None and int(max_tokens) < 1:
            raise ValueError("max_tokens 必须大于 0")
        memory_access = profile.get("memory_access", {})
        if not isinstance(memory_access, dict):
            raise ValueError("memory_access 必须是对象")
        for field in (
            "enabled",
            "read_story_facts",
            "read_author_style",
            "read_raw_passages",
            "read_foreshadowing",
            "character_filter",
            "timeline_filter",
        ):
            if field in memory_access and not isinstance(memory_access[field], bool):
                raise ValueError(f"memory_access.{field} 必须是布尔值")
        top_k = memory_access.get("top_k")
        if top_k is not None and not 1 <= int(top_k) <= 50:
            raise ValueError("memory_access.top_k 必须在 1 到 50 之间")


def _nullable(value: Any) -> Any:
    return None if value == "" else value
