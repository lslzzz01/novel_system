from __future__ import annotations

import re
from typing import Any

from .llm import LanguageModel
from .models import ProjectBrief


STYLE_AUDIT_PROMPT = """你是多智能体小说系统中的正文文风检测智能体。你只负责审计，不得改写正文。

审计目标：以项目指定的基准章节为文风参照，判断新正文是否保持相同的叙事距离、克制程度、动作化心理、对话密度、生活细节和章末收束方式；同时定位可验证、可修改的 AI 味、连续性和关系边界问题。基准只用于风格与叙事尺度，不得复制基准剧情、人物、物件或具体句子。

基准使用规则：
- style_baseline.source_chapters 中的章节是风格标尺，不是剧情事实来源。
- 优先比较：第三人称有限视角是否稳定；心理是否通过动作、物件和选择呈现；对话是否短促、回避且有潜台词；环境是否与人物行动结合；关系信号是否少量、具体、不过界；结尾是否停在具体动作、物件、问题或未完成反应。
- 不要求新章节复刻基准的场景、节奏或句式。题材、章节功能和 POV 允许变化，但叙事控制力和克制度不能无依据漂移。
- 只有出现可定位的明显偏差才报告问题，不能因为新章节与基准内容不同就判错。

逐项检查：
- 角色是否借对话向读者科普设定，是否连续解释多个术语或因果。
- 是否直接命名愤怒、委屈、疲惫、复杂等情绪，或使用神色凝重、眼神变了、声音里带着等作者判词。
- 是否频繁使用顿了顿、沉默几秒、皱眉、犹豫一下等通用反应。
- 是否在具体细节后追加这说明、这意味着、显然、看来、不是A而是B等解释。
- 对话是否均匀问答、句句推进剧情，缺少打断、短答、回避和人物专属细节。
- 环境是否只用于烘托情绪，或反复使用火光、滴水、黑暗等电影化空镜头。
- 是否存在像是、仿佛、某种、命运齿轮、真正故事才开始等模糊套话。
- 章末是否使用预告片旁白或镜头拉远，而非具体问题、动作、物件、称呼或警告。
- 是否遗漏 chapter_plan.required_reveals、泄露 withhold，或新增计划外命名角色、关键证物和剧情节点。
- 是否偏离基准章节体现的克制尺度，出现无积累的关系跃迁、机械重复同一关系动作、过度露骨或将控制/监视/胁迫浪漫化。

判断边界：
- 不能因为句子朴素、短促、重复口语、安静段落、少量比喻或不够华丽就判定为问题。
- 每项问题必须引用正文中真实存在的连续原文，并使用输入给出的有效 paragraph_id；无法定位的模型判断必须丢弃，不得重试成另一条臆测问题。
- 同一根因只保留一项最有代表性的证据，不把一个句子拆成多个重复问题。
- 修改建议应说明要删什么、保留什么、改成哪类表达，不要代写整段正文。
- continuity_risk 为 high 的问题只提交人工判断，不建议自动改剧情。

只能返回一个 JSON 对象，不得使用 Markdown 代码块或附加说明。结构必须为：status、score、summary、strengths、rewrite_recommended、issues。每个 issue 必须包含 id、category、severity、paragraph_id、quote、reason、suggestion、continuity_risk。severity 和 continuity_risk 只能是 low、medium、high。没有问题时 issues 必须为空数组。"""


STYLE_EDITOR_PROMPT = """你是多智能体小说系统中的正文定点改稿智能体。

你的输入包含原始正文、章节计划和已经人工可见的检测问题。你只能处理 selected_issues 中列出的问题，
不得借机重写整章、统一所有人物口吻或追求更华丽的文字。

硬性边界：
- 保留章节标题、POV、场景顺序、时间线、人物知识边界、required_reveals、withhold 和 ending_hook。
- 不得新增命名角色、地点、法器、证物、通信记录、人物关系、能力、秘密或剧情节点。
- continuity_risk 为 high 的问题只做最小文字整理，不改变事实和因果。
- 未被问题引用的段落原则上原样保留；连接处最多做必要的代词、标点和节奏调整。
- relationship_context 和 relationship_beat 是人物关系硬边界。处理关系问题时只能把表达收回当前阶段或补足已有反应，
  不得新增昵称、告白、吃醋、肢体接触、共享秘密或新的关系事件。
- 删除解释性尾巴时让前面的具体细节独立成立，不要换一个同义解释。
- 百科台词改成短答、追问、回避、个人经历或具体证据，允许少给信息。
- 情绪标签改成当前人物特有的动作；已经有动作时优先直接删除标签。
- 改稿后全文长度保持在原文的 85% 至 110%，除非检测报告明确要求压缩大段重复说明。

只能返回改写后的完整 Markdown 正文。第一行必须保留一级标题，不得输出修改说明、差异、问题编号或总结。
"""


class StyleAuditor:
    def __init__(self, model: LanguageModel, prompt: str | None = None) -> None:
        self.model = model
        self.prompt = prompt or STYLE_AUDIT_PROMPT

    def run(
        self,
        project: ProjectBrief,
        chapter_number: int,
        source_kind: str,
        source_text: str,
        chapter_plan: dict[str, Any],
        rule_findings: list[str],
        style_baseline: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        paragraphs = _paragraphs(source_text)
        payload: dict[str, Any] = {
            "task": "style_audit",
            "project": project.to_dict(),
            "chapter_number": chapter_number,
            "source_kind": source_kind,
            "chapter_plan": chapter_plan,
            "paragraphs": paragraphs,
            "rule_findings": rule_findings,
            "style_baseline": style_baseline or {},
        }
        last_error: ValueError | None = None
        for _ in range(2):
            request = dict(payload)
            if last_error is not None:
                request["audit_repair"] = (
                    "上一次报告无法验证，请重新返回完整 JSON。错误：" + str(last_error)
                )
            raw = self.model.generate_json(self.prompt, request)
            try:
                return self._normalize(raw, source_text, paragraphs, rule_findings)
            except ValueError as exc:
                last_error = exc
        raise RuntimeError(f"文风检测报告连续两次无效：{last_error}")

    @staticmethod
    def _normalize(
        raw: dict[str, Any],
        source_text: str,
        paragraphs: list[dict[str, str]],
        rule_findings: list[str],
    ) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValueError("检测报告必须是对象")
        raw_issues = raw.get("issues", [])
        if not isinstance(raw_issues, list):
            raise ValueError("issues 必须是数组")
        paragraph_ids = {item["id"] for item in paragraphs}
        severity_map = {"高": "high", "中": "medium", "低": "low"}
        issues: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for index, item in enumerate(raw_issues[:30], start=1):
            if not isinstance(item, dict):
                raise ValueError("每个 issue 必须是对象")
            issue_id = str(item.get("id", "")).strip() or f"issue-{index:03d}"
            if issue_id in seen_ids:
                issue_id = f"{issue_id}-{index}"
            seen_ids.add(issue_id)
            quote = str(item.get("quote", "")).strip()
            if quote and quote not in source_text:
                continue
            paragraph_id = str(item.get("paragraph_id", "")).strip()
            if quote:
                located = next(
                    (part["id"] for part in paragraphs if quote in part["text"]), ""
                )
                paragraph_id = located or paragraph_id
            if paragraph_id and paragraph_id not in paragraph_ids:
                continue
            severity = severity_map.get(
                str(item.get("severity", "medium")).strip().lower(),
                str(item.get("severity", "medium")).strip().lower(),
            )
            risk = severity_map.get(
                str(item.get("continuity_risk", "low")).strip().lower(),
                str(item.get("continuity_risk", "low")).strip().lower(),
            )
            if severity not in {"low", "medium", "high"}:
                raise ValueError(f"问题 {issue_id} 的 severity 无效")
            if risk not in {"low", "medium", "high"}:
                raise ValueError(f"问题 {issue_id} 的 continuity_risk 无效")
            reason = str(item.get("reason", "")).strip()
            suggestion = str(item.get("suggestion", "")).strip()
            if not reason or not suggestion:
                raise ValueError(f"问题 {issue_id} 缺少原因或修改建议")
            issues.append(
                {
                    "id": issue_id,
                    "category": str(item.get("category", "other")).strip() or "other",
                    "severity": severity,
                    "paragraph_id": paragraph_id,
                    "quote": quote,
                    "reason": reason,
                    "suggestion": suggestion,
                    "continuity_risk": risk,
                    "origin": "model",
                }
            )

        combined = " ".join(
            issue["reason"] + " " + issue["quote"] for issue in issues
        )
        for index, finding in enumerate(rule_findings, start=1):
            if finding in combined:
                continue
            quote_match = re.search(r"“([^”]+)”", finding)
            quote = quote_match.group(1) if quote_match else ""
            if quote and ("..." in quote or quote not in source_text):
                quote = ""
            paragraph_id = next(
                (part["id"] for part in paragraphs if quote and quote in part["text"]),
                "",
            )
            issues.append(
                {
                    "id": f"rule-{index:03d}",
                    "category": "rule_check",
                    "severity": "medium",
                    "paragraph_id": paragraph_id,
                    "quote": quote,
                    "reason": finding,
                    "suggestion": "按项目的去 AI 味规则定点修改，并保留原有剧情事实。",
                    "continuity_risk": "low",
                    "origin": "rule",
                }
            )

        weights = {"low": 4, "medium": 9, "high": 18}
        calculated_score = max(
            0, 100 - sum(weights[issue["severity"]] for issue in issues)
        )
        try:
            score = max(0, min(100, int(raw.get("score", calculated_score))))
        except (TypeError, ValueError):
            score = calculated_score
        if issues:
            score = min(score, calculated_score)
        status = "pass"
        if any(issue["severity"] == "high" for issue in issues):
            status = "fail"
        elif issues:
            status = "review"
        strengths = raw.get("strengths", [])
        if not isinstance(strengths, list):
            strengths = []
        return {
            "status": status,
            "score": score,
            "summary": str(raw.get("summary", "")).strip()
            or ("未发现需要修改的 AI 味问题。" if not issues else "发现可定位的文风问题。"),
            "strengths": [str(item).strip() for item in strengths if str(item).strip()][:8],
            "rewrite_recommended": bool(issues),
            "issues": issues[:30],
        }


class StyleEditor:
    def __init__(self, model: LanguageModel, prompt: str | None = None) -> None:
        self.model = model
        self.prompt = prompt or STYLE_EDITOR_PROMPT

    def run(
        self,
        project: ProjectBrief,
        chapter_number: int,
        source_kind: str,
        source_text: str,
        chapter_plan: dict[str, Any],
        selected_issues: list[dict[str, Any]],
        relationship_context: dict[str, Any] | None = None,
        relationship_beat: dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "task": "style_edit",
            "project": project.to_dict(),
            "chapter_number": chapter_number,
            "source_kind": source_kind,
            "chapter_plan": chapter_plan,
            "protected_contract": {
                "characters": chapter_plan.get("characters", []),
                "required_reveals": chapter_plan.get("required_reveals", []),
                "withhold": chapter_plan.get("withhold", []),
                "ending_hook": chapter_plan.get("ending_hook", ""),
                "forbidden_major_additions": chapter_plan.get(
                    "forbidden_major_additions", []
                ),
            },
            "selected_issues": selected_issues,
            "relationship_context": relationship_context or {"pairs": []},
            "relationship_beat": relationship_beat or {},
            "source_text": source_text,
        }
        for _ in range(2):
            revised = self.model.generate_text(self.prompt, payload).strip()
            if revised:
                if not revised.startswith("# "):
                    title = str(chapter_plan.get("title", f"第{chapter_number}章"))
                    revised = f"# {title}\n\n{revised}"
                return revised
            payload["edit_repair"] = "上一次返回为空，请输出完整 Markdown 正文。"
        raise RuntimeError("正文定点改稿连续两次返回空内容")


def _paragraphs(text: str) -> list[dict[str, str]]:
    parts = [part.strip() for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]
    return [
        {"id": f"p{index:03d}", "text": part}
        for index, part in enumerate(parts, start=1)
    ]
