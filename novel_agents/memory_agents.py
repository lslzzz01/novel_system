from __future__ import annotations

import hashlib
from typing import Any

from .llm import LanguageModel


MEMORY_EXTRACTION_PROMPT = """你是长篇小说系统中的定稿记忆提取智能体。

输入只包含作者已经确认的定稿正文及其文本分块。你的任务不是评价或改写正文，而是提取后续创作必须
记住的事实。必须区分客观发生的事实、人物推测、谎言和未确认信息。所有记忆必须能够追溯到章节和
原文片段，不得将常识或推断写成确定事实。

请返回一个 JSON 对象，包含：
- summary：本章因果摘要，必须写明开端状态、关键行动、结果和结尾状态。
- chunk_annotations：数组，每项包含 chunk_id、importance、characters、location、timeline、keywords。
- facts：数组，每项包含 fact_type、subject、predicate、object、timeline、importance、confidence、
  locked、source_chunk_id、source_quote、payload。
- style_observations：数组，只记录可观察的文本特征，不对作者作价值判断。

fact_type 可使用 event、character_state、relationship、world_rule、location_state、item_state、
foreshadowing、knowledge、promise、secret。importance 和 confidence 范围为 0 到 1。只能返回 JSON，
不得使用 Markdown 代码块或附加说明。
"""


MEMORY_COMPRESSION_PROMPT = """你是长篇小说系统中的阶段记忆压缩智能体。

请将输入的连续章节摘要和结构化事实压缩为可供后续创作使用的阶段记忆。不得删除仍未解决的冲突、
伏笔、承诺、秘密、人物目标和世界规则。已经结束且无后续影响的事件可以降低篇幅，但必须保留因果
结果。不得改变事实，不得自行填补缺失剧情。

返回一个 JSON 对象，包含：
- stage_summary：阶段因果摘要。
- character_states：主要人物在阶段结束时的状态数组。
- unresolved_threads：未解决冲突、伏笔、承诺和秘密数组。
- world_state_changes：世界、组织、地点和物品状态变化数组。
- timeline_digest：按发生顺序整理的事件数组。
只能返回 JSON，不得附加说明。
"""


MEMORY_RERANK_PROMPT = """你是小说记忆检索重排智能体。

根据当前写作问题，对候选记忆片段按剧情相关性排序。优先保留直接涉及当前人物、地点、事件因果、
未回收伏笔和时间线约束的片段；不要因为文风相似就提高无关片段排名。

返回一个 JSON 对象：ordered_ids 为按相关性从高到低排列的候选 ID 数组，reason 为一句简短说明。
只能使用输入中存在的 ID，不得生成新 ID。
"""


class MemoryExtractorAgent:
    def __init__(self, model: LanguageModel, prompt: str = MEMORY_EXTRACTION_PROMPT) -> None:
        self.model = model
        self.prompt = prompt

    def run(
        self,
        chapter_number: int,
        final_text: str,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        result = self.model.generate_json(
            self.prompt,
            {
                "task": "memory_extract",
                "chapter_number": chapter_number,
                "final_text": final_text,
                "chunks": [
                    {"id": item["id"], "scene_index": item["scene_index"], "text": item["text"]}
                    for item in chunks
                ],
            },
        )
        for key in ("summary", "chunk_annotations", "facts", "style_observations"):
            if key not in result:
                raise ValueError(f"记忆提取结果缺少字段：{key}")
        return result


class MemoryCompressionAgent:
    def __init__(self, model: LanguageModel, prompt: str = MEMORY_COMPRESSION_PROMPT) -> None:
        self.model = model
        self.prompt = prompt

    def run(
        self,
        start_chapter: int,
        end_chapter: int,
        summaries: list[dict[str, Any]],
        facts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        result = self.model.generate_json(
            self.prompt,
            {
                "task": "memory_compress",
                "start_chapter": start_chapter,
                "end_chapter": end_chapter,
                "chapter_summaries": summaries,
                "facts": facts,
            },
        )
        for key in (
            "stage_summary",
            "character_states",
            "unresolved_threads",
            "world_state_changes",
            "timeline_digest",
        ):
            if key not in result:
                raise ValueError(f"阶段压缩结果缺少字段：{key}")
        return result


class MemoryRerankerAgent:
    def __init__(self, model: LanguageModel, prompt: str = MEMORY_RERANK_PROMPT) -> None:
        self.model = model
        self.prompt = prompt

    def run(
        self,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> list[str]:
        result = self.model.generate_json(
            self.prompt,
            {
                "task": "memory_rerank",
                "query": query,
                "candidates": [
                    {
                        "id": item["id"],
                        "chapter_number": item["chapter_number"],
                        "characters": item.get("characters", []),
                        "location": item.get("location", ""),
                        "timeline": item.get("timeline", ""),
                        "text": item["text"],
                        "score": item.get("score", 0),
                    }
                    for item in candidates
                ],
            },
        )
        ordered = result.get("ordered_ids", [])
        allowed = {item["id"] for item in candidates}
        return [str(item) for item in ordered if str(item) in allowed]


def normalize_facts(
    chapter_number: int,
    facts: list[dict[str, Any]],
    valid_chunk_ids: set[str],
) -> list[dict[str, Any]]:
    normalized = []
    for index, fact in enumerate(facts, start=1):
        source_chunk_id = str(fact.get("source_chunk_id", ""))
        if source_chunk_id not in valid_chunk_ids:
            source_chunk_id = ""
        identity = "|".join(
            [
                str(chapter_number),
                str(fact.get("fact_type", "event")),
                str(fact.get("subject", "")),
                str(fact.get("predicate", "")),
                str(fact.get("object", "")),
                str(index),
            ]
        )
        fact_id = "F-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        normalized.append(
            {
                "id": fact_id,
                "fact_type": str(fact.get("fact_type", "event")),
                "subject": str(fact.get("subject", "")),
                "predicate": str(fact.get("predicate", "")),
                "object": str(fact.get("object", "")),
                "timeline": str(fact.get("timeline", "")),
                "importance": _unit(fact.get("importance", 0.5)),
                "confidence": _unit(fact.get("confidence", 1.0)),
                "locked": bool(fact.get("locked", False)),
                "source_chunk_id": source_chunk_id,
                "source_quote": str(fact.get("source_quote", ""))[:500],
                "payload": fact.get("payload", {}) if isinstance(fact.get("payload", {}), dict) else {},
            }
        )
    return normalized


def _unit(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.5

