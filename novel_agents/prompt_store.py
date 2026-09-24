from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .agents import (
    CAMPUS_WRITER_PROMPT,
    ChapterPlanner,
    ChapterWriter,
    CharacterDesigner,
    GenreAnalyst,
    OutlineArchitect,
    SubplotDesigner,
    URBAN_WRITER_PROMPT,
    WorldBuilder,
    XUANHUAN_WRITER_PROMPT,
    compose_chapter_writer_prompt,
)
from .memory_agents import (
    MEMORY_COMPRESSION_PROMPT,
    MEMORY_EXTRACTION_PROMPT,
    MEMORY_RERANK_PROMPT,
)
from .relationship_agents import (
    RELATIONSHIP_ARCHITECT_PROMPT,
    RELATIONSHIP_AUDIT_PROMPT,
    RELATIONSHIP_BEAT_PROMPT,
    RELATIONSHIP_MEMORY_PROMPT,
)
from .style_agents import STYLE_AUDIT_PROMPT, STYLE_EDITOR_PROMPT


@dataclass(frozen=True, slots=True)
class AgentDescriptor:
    id: str
    display_name: str
    stage: str
    output_mode: str
    required: bool
    default_prompt: str
    hook: str = "required"
    kind: str = "builtin"
    activation: str = "always"
    genre_key: str = ""
    genre_label: str = ""
    profile_parent: str = ""


BUILTIN_AGENTS: tuple[AgentDescriptor, ...] = (
    AgentDescriptor(
        "genre_analysis",
        "题材与受众分析师",
        "规划",
        "json",
        True,
        GenreAnalyst.definition.prompt,
    ),
    AgentDescriptor(
        "world_bible",
        "世界观设计师",
        "规划",
        "json",
        True,
        WorldBuilder.definition.prompt,
    ),
    AgentDescriptor(
        "character_bible",
        "人物设计师",
        "规划",
        "json",
        True,
        CharacterDesigner.definition.prompt,
    ),
    AgentDescriptor(
        "story_outline",
        "大纲策划师",
        "规划",
        "json",
        True,
        OutlineArchitect.definition.prompt,
    ),
    AgentDescriptor(
        "relationship_arcs",
        "人物关系弧线规划智能体",
        "规划",
        "json",
        True,
        RELATIONSHIP_ARCHITECT_PROMPT,
    ),
    AgentDescriptor(
        "subplot_register",
        "支线与伏笔设计师",
        "规划",
        "json",
        True,
        SubplotDesigner.definition.prompt,
    ),
    AgentDescriptor(
        "chapter_plan",
        "章节规划师",
        "章节",
        "json",
        True,
        ChapterPlanner.prompt,
    ),
    AgentDescriptor(
        "relationship_beat",
        "关系场景导演智能体",
        "章节",
        "json",
        True,
        RELATIONSHIP_BEAT_PROMPT,
        hook="after_chapter_plan",
    ),
    AgentDescriptor(
        "chapter_draft",
        "正文创作调度与共享规则",
        "正文",
        "text",
        True,
        ChapterWriter.prompt,
    ),
    AgentDescriptor(
        "chapter_writer_xuanhuan",
        "玄幻正文智能体",
        "正文",
        "text",
        True,
        XUANHUAN_WRITER_PROMPT,
        activation="genre",
        genre_key="xuanhuan",
        genre_label="玄幻",
        profile_parent="chapter_draft",
    ),
    AgentDescriptor(
        "chapter_writer_campus",
        "校园正文智能体",
        "正文",
        "text",
        True,
        CAMPUS_WRITER_PROMPT,
        activation="genre",
        genre_key="campus",
        genre_label="校园",
        profile_parent="chapter_draft",
    ),
    AgentDescriptor(
        "chapter_writer_urban",
        "都市正文智能体",
        "正文",
        "text",
        True,
        URBAN_WRITER_PROMPT,
        activation="genre",
        genre_key="urban",
        genre_label="都市",
        profile_parent="chapter_draft",
    ),
    AgentDescriptor(
        "style_audit",
        "正文文风检测智能体",
        "质检",
        "json",
        False,
        STYLE_AUDIT_PROMPT,
        hook="after_chapter_draft",
    ),
    AgentDescriptor(
        "relationship_audit",
        "人物关系一致性审校智能体",
        "质检",
        "json",
        False,
        RELATIONSHIP_AUDIT_PROMPT,
        hook="after_chapter_draft",
    ),
    AgentDescriptor(
        "style_editor",
        "正文定点改稿智能体",
        "质检",
        "text",
        False,
        STYLE_EDITOR_PROMPT,
        hook="after_chapter_draft",
    ),
    AgentDescriptor(
        "memory_extract",
        "定稿记忆提取智能体",
        "记忆",
        "json",
        True,
        MEMORY_EXTRACTION_PROMPT,
        hook="after_chapter_draft",
    ),
    AgentDescriptor(
        "relationship_memory",
        "定稿关系记忆整理智能体",
        "记忆",
        "json",
        True,
        RELATIONSHIP_MEMORY_PROMPT,
        hook="after_chapter_finalized",
    ),
    AgentDescriptor(
        "memory_compress",
        "阶段记忆压缩智能体",
        "记忆",
        "json",
        True,
        MEMORY_COMPRESSION_PROMPT,
        hook="after_volume",
    ),
    AgentDescriptor(
        "memory_rerank",
        "记忆检索重排智能体",
        "记忆",
        "json",
        True,
        MEMORY_RERANK_PROMPT,
        hook="before_chapter_plan",
    ),
)


class PromptStore:
    HOOKS = {
        "after_core_planning",
        "before_chapter_plan",
        "after_chapter_plan",
        "after_chapter_draft",
        "after_chapter_finalized",
        "after_volume",
        "after_book",
    }

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()
        self.local_dir = self.workspace / ".novel_agents"
        self.prompts_dir = self.local_dir / "prompts"
        self.custom_path = self.local_dir / "custom_agents.json"

    def list_agents(self) -> list[dict[str, Any]]:
        agents = [self._builtin_public(agent) for agent in BUILTIN_AGENTS]
        agents.extend(self._load_custom())
        return agents

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        builtin = self._find_builtin(agent_id)
        if builtin:
            return self._builtin_public(builtin, include_prompt=True)
        for agent in self._load_custom():
            if agent["id"] == agent_id:
                return agent
        raise KeyError(f"未知智能体：{agent_id}")

    def save_prompt(self, agent_id: str, prompt: str) -> dict[str, Any]:
        prompt = prompt.strip()
        if len(prompt) < 20:
            raise ValueError("提示词至少需要 20 个字符")
        builtin = self._find_builtin(agent_id)
        if builtin:
            self.prompts_dir.mkdir(parents=True, exist_ok=True)
            self._write_text(self.prompts_dir / f"{agent_id}.txt", prompt + "\n")
            return self.get_agent(agent_id)

        custom = self._load_custom()
        for item in custom:
            if item["id"] == agent_id:
                item["prompt"] = prompt
                self._write_json(self.custom_path, custom)
                return item
        raise KeyError(f"未知智能体：{agent_id}")

    def reset_prompt(self, agent_id: str) -> dict[str, Any]:
        builtin = self._find_builtin(agent_id)
        if not builtin:
            raise ValueError("自定义智能体没有内置提示词可恢复")
        path = self.prompts_dir / f"{agent_id}.txt"
        if path.exists():
            path.unlink()
        return self.get_agent(agent_id)

    def prompt_overrides(self) -> dict[str, str]:
        overrides: dict[str, str] = {}
        for descriptor in BUILTIN_AGENTS:
            path = self.prompts_dir / f"{descriptor.id}.txt"
            if path.exists():
                overrides[descriptor.id] = path.read_text(encoding="utf-8").strip()
        return overrides

    def create_custom(self, values: dict[str, Any]) -> dict[str, Any]:
        agent_id = str(values.get("id", "")).strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,39}", agent_id):
            raise ValueError("智能体 ID 必须由小写字母、数字和下划线组成")
        if self._find_builtin(agent_id):
            raise ValueError("该 ID 已被内置智能体使用")
        display_name = str(values.get("display_name", "")).strip()
        prompt = str(values.get("prompt", "")).strip()
        hook = str(values.get("hook", "after_chapter_draft"))
        output_mode = str(values.get("output_mode", "json"))
        if not display_name:
            raise ValueError("智能体名称不能为空")
        if len(prompt) < 20:
            raise ValueError("提示词至少需要 20 个字符")
        if hook not in self.HOOKS:
            raise ValueError("不支持的工作流挂载点")
        if output_mode not in {"json", "text"}:
            raise ValueError("输出模式只能是 json 或 text")

        custom = self._load_custom()
        if any(item["id"] == agent_id for item in custom):
            raise ValueError("智能体 ID 已存在")
        item = {
            "id": agent_id,
            "display_name": display_name,
            "stage": "扩展",
            "output_mode": output_mode,
            "required": False,
            "prompt": prompt,
            "hook": hook,
            "kind": "custom",
            "enabled": bool(values.get("enabled", False)),
            "runtime_status": "reserved",
        }
        custom.append(item)
        self._write_json(self.custom_path, custom)
        return item

    def update_custom(self, agent_id: str, values: dict[str, Any]) -> dict[str, Any]:
        if self._find_builtin(agent_id):
            raise ValueError("内置智能体只能修改提示词和模型参数")
        custom = self._load_custom()
        item = next((entry for entry in custom if entry.get("id") == agent_id), None)
        if item is None:
            raise KeyError(f"未知智能体：{agent_id}")

        display_name = str(values.get("display_name", item.get("display_name", ""))).strip()
        prompt = str(values.get("prompt", item.get("prompt", ""))).strip()
        hook = str(values.get("hook", item.get("hook", "after_chapter_draft")))
        output_mode = str(values.get("output_mode", item.get("output_mode", "json")))
        if not display_name:
            raise ValueError("智能体名称不能为空")
        if len(prompt) < 20:
            raise ValueError("提示词至少需要 20 个字符")
        if hook not in self.HOOKS:
            raise ValueError("不支持的工作流挂载点")
        if output_mode not in {"json", "text"}:
            raise ValueError("输出模式只能是 json 或 text")

        item.update(
            {
                "display_name": display_name,
                "hook": hook,
                "output_mode": output_mode,
                "prompt": prompt,
                "enabled": bool(values.get("enabled", item.get("enabled", False))),
            }
        )
        self._write_json(self.custom_path, custom)
        return item

    def delete_custom(self, agent_id: str) -> dict[str, Any]:
        if self._find_builtin(agent_id):
            raise ValueError("内置智能体不能删除")
        custom = self._load_custom()
        item = next((entry for entry in custom if entry.get("id") == agent_id), None)
        if item is None:
            raise KeyError(f"未知智能体：{agent_id}")
        self._write_json(
            self.custom_path,
            [entry for entry in custom if entry.get("id") != agent_id],
        )
        return item

    def _builtin_public(
        self, descriptor: AgentDescriptor, include_prompt: bool = False
    ) -> dict[str, Any]:
        path = self.prompts_dir / f"{descriptor.id}.txt"
        overridden = path.exists()
        prompt = self._builtin_prompt(descriptor)
        data = asdict(descriptor)
        data.pop("default_prompt")
        data["overridden"] = overridden
        data["runtime_status"] = "active"
        if include_prompt:
            data["prompt"] = prompt
            data["default_prompt"] = descriptor.default_prompt
            if descriptor.profile_parent == "chapter_draft":
                shared = self._builtin_prompt(
                    self._find_builtin("chapter_draft")
                    or next(item for item in BUILTIN_AGENTS if item.id == "chapter_draft")
                )
                data["combined_prompt"] = compose_chapter_writer_prompt(
                    shared,
                    prompt,
                    descriptor.display_name,
                )
            else:
                data["combined_prompt"] = prompt
        return data

    def _builtin_prompt(self, descriptor: AgentDescriptor) -> str:
        path = self.prompts_dir / f"{descriptor.id}.txt"
        return (
            path.read_text(encoding="utf-8").strip()
            if path.exists()
            else descriptor.default_prompt
        )

    @staticmethod
    def _find_builtin(agent_id: str) -> AgentDescriptor | None:
        return next((item for item in BUILTIN_AGENTS if item.id == agent_id), None)

    def _load_custom(self) -> list[dict[str, Any]]:
        if not self.custom_path.exists():
            return []
        with self.custom_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("custom_agents.json 必须是数组")
        return [item for item in data if isinstance(item, dict)]

    def _write_json(self, path: Path, data: list[dict[str, Any]]) -> None:
        self._write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")

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
