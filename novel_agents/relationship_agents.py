from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .artifacts import ArtifactStore
from .llm import LanguageModel
from .models import ProjectBrief
from .relationship_store import RELATIONSHIP_STAGES, RelationshipStateStore


RELATIONSHIP_ARCHITECT_PROMPT = """你是多智能体小说系统中的人物关系弧线规划智能体。

你的任务是根据人物设定和故事总纲，为重要人物组合建立可持续数十万字的关系弧线。关系不能只是“陌生、暧昧、相恋”标签，
必须同时规划双向认知差异、称呼、信任、吸引、依赖、戒备、公开关系、私下关系、边界、共同经历和未解决矛盾。

工作规则：
- A 对 B 与 B 对 A 必须分开设计，禁止默认双方同步动心、同步信任或同步理解当前关系。
- 允许长时间停留、倒退、误判、疏远、决裂与和解；不能要求每一章都推进关系。
- 每次阶段变化必须依赖可观察事件与对方回应，不能因为一次救命、一次脸红或一次吃醋直接进入相恋。
- 单女主、无后宫、年龄和行为边界等项目约束是硬规则，不得增加计划外暧昧对象。
- 玄幻关系通过交付后背、共享功法或弱点、称呼和阵营代价体现；校园关系通过座位、消息、借还物品、放学路线和同伴目光体现；
  都市关系通过时间安排、工作边界、承诺兑现、生活圈和家庭责任体现。
- 关系信号必须是具体行为，不要把“关系升温、产生好感、气氛暧昧”当作可执行指令。
- 只返回一个 JSON 对象，不得使用 Markdown 或附加说明。

输出结构：
global_rules：字符串数组，全书关系线共同规则。
relationships：对象数组，每项必须包含：
id：稳定英文 ID，可留空由系统生成；character_a、character_b：人物姓名；relationship_type：关系类型；
initial_stage、target_stage：只能是 stranger、acquainted、familiar、ambiguous、dating、distant、broken、reconciled；
public_status、private_status：开篇公开与私下状态；
a_to_b、b_to_a：对象，包含 trust、attraction、dependence、guard（0 到 5）和 perception；
address_terms：对象，包含 a_to_b、b_to_a；touch_boundary：开篇肢体边界；
arc_summary：关系弧线概述；milestones：对象数组，每项包含 chapter_range、trigger、observable_change、required_evidence；
next_milestone：最近的下一节点；forbidden_leaps：禁止跃迁；genre_signals：本题材可使用的具体关系信号。
"""


RELATIONSHIP_BEAT_PROMPT = """你是多智能体小说系统中的关系场景导演智能体。你的输出会在章节规划之后交给正文智能体。

你必须根据章节计划、关系总弧线和当前关系状态，为本章安排克制、可见、不过界的关系节拍。

规则：
- 大多数章节只允许 hold 或 subtle；只有总纲与关系里程碑共同支持时才能使用 clear 或 milestone。
- 每组人物最多选择 1 到 3 个可观察信号，正文不需要把所有关系维度都展示一遍。
- 分开考虑 A 对 B 与 B 对 A。可以一方靠近、另一方回避，也可以双方对同一动作作出不同理解。
- 优先使用称呼、距离、顺手动作、信息分享、等待、承诺兑现、旧事回扣和边界试探；禁止直接要求正文写“关系更近了”或情绪标签。
- 明确列出本章不能出现的越界行为，例如突然昵称、无积累的占有欲、计划外肢体接触、突然交底或直接告白。
- 关系可以不变或倒退，但不能为了制造戏剧冲突让人物无理由失忆、翻脸或误会。
- 只返回一个 JSON 对象，不得使用 Markdown 或附加说明。

输出结构：
chapter_number：整数；intensity：hold、subtle、clear 或 milestone；
focus_pairs：对象数组，每项包含 pair_id、direction、stage_before、stage_after_allowed、scene_signals、callback、boundary、reason；
behavior_signals、callbacks、forbidden_leaps：字符串数组；progression_summary：字符串；
stage_change_allowed：布尔值。没有适用关系时 focus_pairs 和三个数组可为空，intensity 必须为 hold。
"""


RELATIONSHIP_AUDIT_PROMPT = """你是多智能体小说系统中的人物关系一致性审校智能体。你只检测并提交修改建议，不得改写正文。

输入包含当前人物关系状态、本章关系节拍卡、章节计划和带编号的正文段落。检查：
- 是否出现超出当前阶段的昵称、交底、依赖、占有欲、吃醋、肢体接触或恋人式默契。
- 是否把 A 对 B 与 B 对 A 写成同步感受，抹掉已有认知差异。
- 是否突然退回陌生或突然翻脸，却没有章节计划支持的事件和具体反应。
- 是否直接解释“关系升温、气氛暧昧、他意识到自己动心”，而不是用具体行为呈现。
- 是否遗漏节拍卡要求的关键关系动作，或一次塞入太多关系信号，显得机械打卡。
- 称呼、公开关系、私下关系、秘密共享、触碰边界和未解决矛盾是否与已有状态冲突。

判断边界：
- 普通礼貌、同伴合作和剧情所需救援不自动等于暧昧。
- 每个问题必须引用正文中真实存在的连续原文；没有可定位证据就不要报问题。
- 修改建议只说明删减、收回到哪种边界或补哪类具体反应，不要重写整段。
- 可能改变剧情事实的问题将 continuity_risk 标为 high，交给人工决定。
- paragraph_id 必须使用输入提供的段落编号。

只能返回一个 JSON 对象，结构为：status、score、summary、strengths、rewrite_recommended、issues。
每个 issue 必须包含 id、category、severity、paragraph_id、quote、reason、suggestion、continuity_risk。
category 只能是 relationship_stage、relationship_asymmetry、relationship_boundary、relationship_continuity、relationship_explanation。
severity 和 continuity_risk 只能是 low、medium、high。没有问题时 issues 必须为空数组。
"""


RELATIONSHIP_MEMORY_PROMPT = """你是多智能体小说系统中的定稿关系记忆整理智能体。

你只能分析输入的最终定稿正文，并把正文中已经实际发生的人物关系变化转成结构化事件。草稿计划、作者意图和关系节拍卡只能帮助理解，
不能替代定稿证据。没有写进定稿的变化不得记录。

规则：
- 每组人物在一章中最多输出一个汇总事件；没有变化就不要输出。
- A 对 B 与 B 对 A 分开记录。trust、attraction、dependence、guard 的变化范围只能是 -2 到 2。
- 脸红、对视、一次救援或普通关心通常只支持细微数值变化，不能单独支持阶段跃迁。
- proposed_stage 只在定稿包含明确、可引用的里程碑证据时填写，否则留空。系统会把阶段变化交给人工确认。
- evidence_quotes 必须逐字来自定稿正文，使用 1 到 3 条短引用。
- 称呼、公开关系、私下关系、共享秘密、共同经历、触碰边界和矛盾变化只记录本章实际呈现的内容。
- 不评价文风，不改写正文，不预测下一章。
- 只返回一个 JSON 对象，不得使用 Markdown 或附加说明。

输出结构：events：对象数组。每项包含 pair_id、character_a、character_b、summary、evidence_quotes、confidence、proposed_stage、
deltas（含 a_to_b 与 b_to_a，各含 trust、attraction、dependence、guard）、perceptions（a_to_b、b_to_a）、
address_terms（a_to_b、b_to_a）、public_status、private_status、touch_boundary、next_milestone、
shared_secrets、shared_experiences、unresolved_tensions、resolved_tensions。
"""


@dataclass(frozen=True, slots=True)
class RelationshipAgentDefinition:
    task: str
    display_name: str
    output_name: str
    prompt: str


class RelationshipArchitect:
    definition = RelationshipAgentDefinition(
        "relationship_arcs",
        "人物关系弧线规划智能体",
        "relationship_arcs",
        RELATIONSHIP_ARCHITECT_PROMPT,
    )

    def __init__(
        self,
        model: LanguageModel,
        artifacts: ArtifactStore,
        relationships: RelationshipStateStore,
        prompt: str | None = None,
    ) -> None:
        self.model = model
        self.artifacts = artifacts
        self.relationships = relationships
        self.prompt = prompt or self.definition.prompt

    def run(
        self,
        project: ProjectBrief,
        inputs: dict[str, Any],
        force: bool = False,
    ) -> dict[str, Any]:
        if self.artifacts.has_artifact(self.definition.output_name) and not force:
            result = self.artifacts.read_artifact(self.definition.output_name)
            self.relationships.import_plan(result)
            return result
        self.artifacts.update_stage(
            self.definition.task, "running", self.definition.display_name
        )
        payload = {
            "task": self.definition.task,
            "project": project.to_dict(),
            "inputs": inputs,
            "stage_values": list(RELATIONSHIP_STAGES),
        }
        try:
            result = _generate_json(self.model, self.prompt, payload, self._validate)
            path = self.artifacts.write_artifact(self.definition.output_name, result)
            imported = self.relationships.import_plan(result)
            self.artifacts.update_stage(
                self.definition.task,
                "completed",
                f"{path} | {imported['pairs']} 组关系",
            )
            return result
        except Exception as exc:
            self.artifacts.update_stage(self.definition.task, "failed", str(exc))
            raise

    @staticmethod
    def _validate(result: dict[str, Any]) -> None:
        if not isinstance(result.get("global_rules"), list):
            raise ValueError("关系弧线缺少 global_rules 数组")
        relationships = result.get("relationships")
        if not isinstance(relationships, list):
            raise ValueError("关系弧线缺少 relationships 数组")
        for index, item in enumerate(relationships, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"第 {index} 组关系必须是对象")
            if not str(item.get("character_a", "")).strip() or not str(
                item.get("character_b", "")
            ).strip():
                raise ValueError(f"第 {index} 组关系缺少人物姓名")
            for field in ("initial_stage", "target_stage"):
                if str(item.get(field, "")).strip() not in RELATIONSHIP_STAGES:
                    raise ValueError(f"第 {index} 组关系的 {field} 无效")


class RelationshipBeatPlanner:
    task = "relationship_beat"
    prompt = RELATIONSHIP_BEAT_PROMPT

    def __init__(
        self,
        model: LanguageModel,
        artifacts: ArtifactStore,
        prompt: str | None = None,
    ) -> None:
        self.model = model
        self.artifacts = artifacts
        self.runtime_prompt = prompt or self.prompt

    def run(
        self,
        project: ProjectBrief,
        chapter_number: int,
        inputs: dict[str, Any],
        force: bool = False,
    ) -> dict[str, Any]:
        if self.artifacts.has_relationship_beat(chapter_number) and not force:
            return self.artifacts.read_relationship_beat(chapter_number)
        stage = f"relationship_beat_{chapter_number:03d}"
        self.artifacts.update_stage(stage, "running", "关系场景导演")
        relationship_context = inputs.get("relationship_context", {})
        pairs = relationship_context.get("pairs", []) if isinstance(relationship_context, dict) else []
        if not pairs or relationship_context.get("settings", {}).get("enabled") is False:
            result = self.noop(chapter_number)
            path = self.artifacts.write_relationship_beat(chapter_number, result)
            self.artifacts.update_stage(stage, "completed", f"{path} | 本章无关系节拍")
            return result
        payload = {
            "task": self.task,
            "chapter_number": chapter_number,
            "project": project.to_dict(),
            "inputs": inputs,
        }
        try:
            result = _generate_json(
                self.model,
                self.runtime_prompt,
                payload,
                lambda value: self._validate(value, chapter_number),
            )
            path = self.artifacts.write_relationship_beat(chapter_number, result)
            self.artifacts.update_stage(
                stage,
                "completed",
                f"{path} | {result.get('intensity', 'hold')}",
            )
            return result
        except Exception as exc:
            self.artifacts.update_stage(stage, "failed", str(exc))
            raise

    @staticmethod
    def noop(chapter_number: int) -> dict[str, Any]:
        return {
            "chapter_number": chapter_number,
            "intensity": "hold",
            "focus_pairs": [],
            "behavior_signals": [],
            "callbacks": [],
            "forbidden_leaps": [],
            "progression_summary": "本章没有需要单独安排的人物关系变化。",
            "stage_change_allowed": False,
        }

    @staticmethod
    def _validate(result: dict[str, Any], chapter_number: int) -> None:
        if int(result.get("chapter_number", 0)) != chapter_number:
            raise ValueError("关系节拍的章节编号不一致")
        if result.get("intensity") not in {"hold", "subtle", "clear", "milestone"}:
            raise ValueError("关系节拍 intensity 无效")
        for field in ("focus_pairs", "behavior_signals", "callbacks", "forbidden_leaps"):
            if not isinstance(result.get(field), list):
                raise ValueError(f"关系节拍 {field} 必须是数组")
        if len(result["focus_pairs"]) > 4:
            raise ValueError("单章关系重点不能超过 4 组")
        if not isinstance(result.get("stage_change_allowed"), bool):
            raise ValueError("stage_change_allowed 必须是布尔值")


class RelationshipAuditor:
    def __init__(self, model: LanguageModel, prompt: str | None = None) -> None:
        self.model = model
        self.prompt = prompt or RELATIONSHIP_AUDIT_PROMPT

    def run(
        self,
        project: ProjectBrief,
        chapter_number: int,
        source_kind: str,
        source_text: str,
        chapter_plan: dict[str, Any],
        relationship_context: dict[str, Any],
        relationship_beat: dict[str, Any],
    ) -> dict[str, Any]:
        if not relationship_context.get("pairs"):
            return self.pass_report()
        paragraphs = _paragraphs(source_text)
        payload = {
            "task": "relationship_audit",
            "project": project.to_dict(),
            "chapter_number": chapter_number,
            "source_kind": source_kind,
            "chapter_plan": chapter_plan,
            "relationship_context": relationship_context,
            "relationship_beat": relationship_beat,
            "paragraphs": paragraphs,
        }
        last_error: ValueError | None = None
        for _ in range(2):
            request = dict(payload)
            if last_error is not None:
                request["audit_repair"] = "上一次关系审校报告无效，请完整重做。错误：" + str(last_error)
            raw = self.model.generate_json(self.prompt, request)
            try:
                return self._normalize(raw, source_text, paragraphs)
            except ValueError as exc:
                last_error = exc
        raise RuntimeError(f"人物关系审校连续两次无效：{last_error}")

    @staticmethod
    def pass_report() -> dict[str, Any]:
        return {
            "status": "pass",
            "score": 100,
            "summary": "本章没有需要审校的人物关系组合。",
            "strengths": [],
            "rewrite_recommended": False,
            "issues": [],
        }

    @staticmethod
    def _normalize(
        raw: dict[str, Any], source_text: str, paragraphs: list[dict[str, str]]
    ) -> dict[str, Any]:
        if not isinstance(raw, dict) or not isinstance(raw.get("issues", []), list):
            raise ValueError("关系审校必须返回包含 issues 的对象")
        paragraph_ids = {item["id"] for item in paragraphs}
        categories = {
            "relationship_stage",
            "relationship_asymmetry",
            "relationship_boundary",
            "relationship_continuity",
            "relationship_explanation",
        }
        issues: list[dict[str, Any]] = []
        for index, item in enumerate(raw.get("issues", [])[:30], start=1):
            if not isinstance(item, dict):
                raise ValueError("关系审校 issue 必须是对象")
            quote = str(item.get("quote", "")).strip()
            if not quote or quote not in source_text:
                raise ValueError(f"关系问题 {index} 缺少可定位原文")
            paragraph_id = next(
                (part["id"] for part in paragraphs if quote in part["text"]),
                str(item.get("paragraph_id", "")).strip(),
            )
            if paragraph_id not in paragraph_ids:
                raise ValueError(f"关系问题 {index} 的 paragraph_id 无效")
            severity = str(item.get("severity", "medium")).strip().lower()
            risk = str(item.get("continuity_risk", "low")).strip().lower()
            if severity not in {"low", "medium", "high"} or risk not in {
                "low",
                "medium",
                "high",
            }:
                raise ValueError(f"关系问题 {index} 的级别无效")
            category = str(item.get("category", "relationship_continuity")).strip()
            if category not in categories:
                category = "relationship_continuity"
            reason = str(item.get("reason", "")).strip()
            suggestion = str(item.get("suggestion", "")).strip()
            if not reason or not suggestion:
                raise ValueError(f"关系问题 {index} 缺少原因或建议")
            issues.append(
                {
                    "id": "relationship-" + (str(item.get("id", "")).strip() or f"{index:03d}"),
                    "category": category,
                    "severity": severity,
                    "paragraph_id": paragraph_id,
                    "quote": quote,
                    "reason": reason,
                    "suggestion": suggestion,
                    "continuity_risk": risk,
                    "origin": "relationship",
                }
            )
        weights = {"low": 4, "medium": 9, "high": 18}
        score = max(0, 100 - sum(weights[item["severity"]] for item in issues))
        status = "fail" if any(item["severity"] == "high" for item in issues) else "review" if issues else "pass"
        strengths = raw.get("strengths", [])
        return {
            "status": status,
            "score": score,
            "summary": str(raw.get("summary", "")).strip()
            or ("人物关系表现与当前阶段一致。" if not issues else "发现人物关系越界或连续性问题。"),
            "strengths": [str(item).strip() for item in strengths if str(item).strip()][:8]
            if isinstance(strengths, list)
            else [],
            "rewrite_recommended": bool(issues),
            "issues": issues,
        }


class RelationshipMemoryAgent:
    def __init__(self, model: LanguageModel, prompt: str | None = None) -> None:
        self.model = model
        self.prompt = prompt or RELATIONSHIP_MEMORY_PROMPT

    def run(
        self,
        project: ProjectBrief,
        chapter_number: int,
        final_text: str,
        chapter_plan: dict[str, Any],
        relationship_context: dict[str, Any],
        relationship_beat: dict[str, Any],
    ) -> dict[str, Any]:
        if not relationship_context.get("pairs"):
            return {"events": []}
        payload = {
            "task": "relationship_memory",
            "project": project.to_dict(),
            "chapter_number": chapter_number,
            "chapter_plan": chapter_plan,
            "relationship_context": relationship_context,
            "relationship_beat": relationship_beat,
            "final_text": final_text,
        }
        return _generate_json(
            self.model,
            self.prompt,
            payload,
            lambda value: self._validate(value, final_text, relationship_context),
        )

    @staticmethod
    def _validate(
        result: dict[str, Any], final_text: str, relationship_context: dict[str, Any]
    ) -> None:
        events = result.get("events")
        if not isinstance(events, list):
            raise ValueError("关系记忆结果缺少 events 数组")
        valid_ids = {str(item.get("id", "")) for item in relationship_context.get("pairs", [])}
        seen: set[str] = set()
        for index, item in enumerate(events, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"关系事件 {index} 必须是对象")
            pair_id = str(item.get("pair_id", "")).strip()
            if pair_id not in valid_ids:
                raise ValueError(f"关系事件 {index} 使用了未知 pair_id")
            if pair_id in seen:
                raise ValueError("同一人物组合一章只能输出一个关系事件")
            seen.add(pair_id)
            quotes = item.get("evidence_quotes", [])
            if not isinstance(quotes, list) or not quotes:
                raise ValueError(f"关系事件 {index} 缺少定稿证据")
            for quote in quotes:
                text = str(quote).strip()
                if not text or text not in final_text:
                    raise ValueError(f"关系事件 {index} 的证据无法在定稿中定位")
            proposed = str(item.get("proposed_stage", "")).strip()
            if proposed and proposed not in RELATIONSHIP_STAGES:
                raise ValueError(f"关系事件 {index} 的 proposed_stage 无效")


def merge_audit_reports(
    style_report: dict[str, Any], relationship_report: dict[str, Any]
) -> dict[str, Any]:
    issues = list(style_report.get("issues", [])) + list(
        relationship_report.get("issues", [])
    )
    score = min(
        int(style_report.get("score", 100)),
        int(relationship_report.get("score", 100)),
    )
    status = "fail" if any(item.get("severity") == "high" for item in issues) else "review" if issues else "pass"
    strengths = list(style_report.get("strengths", []))
    for item in relationship_report.get("strengths", []):
        if item not in strengths:
            strengths.append(item)
    summaries = [
        str(style_report.get("summary", "")).strip(),
        str(relationship_report.get("summary", "")).strip(),
    ]
    return {
        "status": status,
        "score": score,
        "summary": "；".join(item for item in summaries if item),
        "strengths": strengths[:10],
        "rewrite_recommended": bool(issues),
        "issues": issues[:60],
        "components": {
            "style": style_report,
            "relationship": relationship_report,
        },
    }


def _generate_json(
    model: LanguageModel,
    prompt: str,
    payload: dict[str, Any],
    validate: Any,
) -> dict[str, Any]:
    last_error: ValueError | None = None
    for _ in range(2):
        request = dict(payload)
        if last_error is not None:
            request["schema_repair"] = "上一次 JSON 未通过校验，请完整重做。错误：" + str(last_error)
        result = model.generate_json(prompt, request)
        try:
            validate(result)
            return result
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"模型连续两次没有返回有效关系数据：{last_error}")


def _paragraphs(text: str) -> list[dict[str, str]]:
    parts = [part.strip() for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]
    return [
        {"id": f"p{index:03d}", "text": part}
        for index, part in enumerate(parts, start=1)
    ]
