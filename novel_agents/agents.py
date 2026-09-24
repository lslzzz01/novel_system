from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any

from .artifacts import ArtifactStore
from .llm import LanguageModel, OpenAICompatibleClient
from .models import ProjectBrief


def _generate_validated_json(
    model: LanguageModel,
    prompt: str,
    payload: dict[str, Any],
    validate: Any,
) -> dict[str, Any]:
    last_error: ValueError | None = None
    for _ in range(2):
        request_payload = dict(payload)
        if last_error is not None:
            request_payload["schema_repair"] = (
                "上一次 JSON 未通过字段校验，请完整重做。错误：" + str(last_error)
            )
        result = model.generate_json(prompt, request_payload)
        try:
            validate(result)
            return result
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"模型连续两次没有满足输出协议：{last_error}")


def _json_prompt(role: str, responsibilities: str, contract: str) -> str:
    return f"""你是多智能体小说生产系统中的{role}。

你的职责：
{responsibilities}

工作规则：
- 将项目任务书和上游智能体产物视为不可随意改变的权威约束。
- 不得暗中修改上游智能体已经确立的事实、设定和人物关系。
- 遇到不确定信息时，采用对既有设定影响最小的保守解释。
- 输出必须具体、可执行，能够直接交给下游智能体使用，不要只给笼统建议。
- 所有创作性内容必须使用项目指定的语言。
- 只能返回一个 JSON 对象，不得使用 Markdown 代码块，不得附加解释或前后缀。
- 必须完整保留下方规定的英文字段名，字段内容可以使用项目指定语言。

必须遵守的输出协议：
{contract}
"""


@dataclass(frozen=True, slots=True)
class AgentDefinition:
    task: str
    display_name: str
    output_name: str
    required_keys: tuple[str, ...]
    prompt: str


class JsonAgent:
    definition: AgentDefinition

    def __init__(
        self,
        model: LanguageModel,
        store: ArtifactStore,
        prompt: str | None = None,
    ) -> None:
        self.model = model
        self.store = store
        self.prompt = prompt or self.definition.prompt

    def run(
        self,
        project: ProjectBrief,
        inputs: dict[str, Any],
        force: bool = False,
    ) -> dict[str, Any]:
        definition = self.definition
        if self.store.has_artifact(definition.output_name) and not force:
            return self.store.read_artifact(definition.output_name)

        self.store.update_stage(definition.task, "running", definition.display_name)
        payload = {
            "task": definition.task,
            "project": project.to_dict(),
            "inputs": inputs,
            "required_keys": list(definition.required_keys),
        }
        try:
            result = _generate_validated_json(
                self.model, self.prompt, payload, self._validate
            )
            path = self.store.write_artifact(definition.output_name, result)
            self.store.update_stage(definition.task, "completed", str(path))
            return result
        except Exception as exc:
            self.store.update_stage(definition.task, "failed", str(exc))
            raise

    def _validate(self, result: dict[str, Any]) -> None:
        missing = [key for key in self.definition.required_keys if key not in result]
        if missing:
            raise ValueError(
                f"{self.definition.display_name} omitted required keys: "
                + ", ".join(missing)
            )


class GenreAnalyst(JsonAgent):
    definition = AgentDefinition(
        task="genre_analysis",
        display_name="题材与受众分析师",
        output_name="genre_analysis",
        required_keys=(
            "positioning",
            "target_readers",
            "core_promises",
            "tone",
            "taboos",
        ),
        prompt=_json_prompt(
            "题材与受众分析师",
            "分析题材惯例、目标读者期待、作品承诺、整体基调、差异化卖点和潜在风险。"
            "此阶段不得编造具体剧情事件，也不得创建有姓名的角色。",
            "positioning：字符串，作品定位；target_readers：字符串，目标读者；"
            "core_promises：字符串数组，作品必须兑现的核心体验；tone：字符串，整体基调；"
            "taboos：字符串数组，创作禁区和应避免的内容。",
        ),
    )


class WorldBuilder(JsonAgent):
    definition = AgentDefinition(
        task="world_bible",
        display_name="世界观设计师",
        output_name="world_bible",
        required_keys=("premise", "rules", "factions", "locations", "history", "glossary"),
        prompt=_json_prompt(
            "世界观设计师",
            "围绕故事前提构建自洽且能推动剧情的世界。规则必须可以验证，力量和选择必须有明确代价，"
            "组织与制度应当能够持续产生冲突。避免创建与剧情、人物和主题无关的装饰性设定。",
            "premise：字符串，世界与故事前提；rules：字符串数组，核心规则；"
            "factions：对象数组，主要势力；locations：对象数组，关键地点；"
            "history：字符串数组，影响当前故事的历史事件；glossary：对象，专有名词表。",
        ),
    )


class CharacterDesigner(JsonAgent):
    definition = AgentDefinition(
        task="character_bible",
        display_name="人物设计师",
        output_name="character_bible",
        required_keys=("protagonists", "supporting_cast", "relationships", "arcs"),
        prompt=_json_prompt(
            "人物设计师",
            "设计能够通过目标、缺陷、关系、秘密和成长弧推动主线冲突的人物。每个主要人物都必须"
            "拥有主动选择能力，并承担明确的剧情功能；不得只作为工具人或设定说明器存在。",
            "protagonists：对象数组，主角资料；supporting_cast：对象数组，配角资料；"
            "relationships：对象数组，人物关系；arcs：字符串数组，主要人物成长弧。",
        ),
    )


class OutlineArchitect(JsonAgent):
    # Keep each chapter-map response comfortably below a provider's configured
    # output ceiling. This is intentionally independent of model max_tokens.
    CHAPTER_MAP_BATCH_SIZE = 20

    definition = AgentDefinition(
        task="story_outline",
        display_name="大纲策划师",
        output_name="story_outline",
        required_keys=("logline", "theme", "ending", "acts", "volumes", "chapter_map"),
        prompt=_json_prompt(
            "大纲策划师",
            "综合项目要求、题材定位、世界观和人物资料，构建具有清晰因果关系的完整故事。"
            "必须锁定结局、重大转折、各卷目标，并为项目要求的每一章指定唯一且明确的剧情作用。",
            "logline：字符串，一句话故事；theme：字符串，核心主题；ending：字符串，锁定结局；"
            "acts：对象数组，阶段结构；volumes：对象数组，分卷结构；chapter_map：对象数组，"
            "必须覆盖每一章，每项至少包含 chapter_number 和 purpose。",
        ),
    )

    def run(
        self,
        project: ProjectBrief,
        inputs: dict[str, Any],
        force: bool = False,
    ) -> dict[str, Any]:
        """Create a complete outline without requiring one oversized response.

        A long project can have hundreds of chapter-map entries. Treating that
        map as one JSON response lets a model hit its output ceiling while the
        old key-only validation still marks planning complete. Generate the
        structural outline first, then request the map in validated batches.
        """
        if self.store.has_artifact(self.definition.output_name) and not force:
            cached = self.store.read_artifact(self.definition.output_name)
            try:
                normalized_map, changed = self._normalize_chapter_map(
                    cached.get("chapter_map"), project
                )
                cached["chapter_map"] = normalized_map
                self._validate_complete(cached, project)
                if changed:
                    self.store.write_artifact(self.definition.output_name, cached)
                return cached
            except ValueError:
                # A legacy or interrupted map must be regenerated instead of
                # being silently reused as a completed overall plan.
                pass

        self.store.update_stage(
            self.definition.task, "running", self.definition.display_name
        )
        try:
            structure_payload = {
                "task": self.definition.task,
                "outline_mode": "structure_only",
                "project": project.to_dict(),
                "inputs": inputs,
                "required_keys": list(self.definition.required_keys),
                "chapter_map_instruction": (
                    "本次只生成总纲结构。chapter_map 必须返回空数组；"
                    "章节映射会由后续分批任务生成。"
                ),
            }
            result = _generate_validated_json(
                self.model,
                self._structure_prompt(),
                structure_payload,
                self._validate_structure,
            )

            # Some models will still return chapter_map despite the structure
            # instruction. Preserve valid entries and only request the gaps.
            try:
                chapter_map, _ = self._normalize_chapter_map(
                    result.get("chapter_map"), project
                )
            except ValueError:
                chapter_map = []
            entries_by_chapter = {
                int(entry["chapter_number"]): entry for entry in chapter_map
            }
            missing_numbers = [
                number
                for number in range(1, project.chapter_count + 1)
                if number not in entries_by_chapter
            ]
            batches = list(self._chapter_batches(missing_numbers))
            for index, batch_numbers in enumerate(batches, start=1):
                self.store.update_stage(
                    self.definition.task,
                    "running",
                    f"正在补齐章节映射（{index}/{len(batches)}）",
                )
                batch_entries = self._generate_chapter_map_batch(
                    project,
                    inputs,
                    result,
                    batch_numbers,
                )
                entries_by_chapter.update(
                    {int(entry["chapter_number"]): entry for entry in batch_entries}
                )

            result["chapter_map"] = [
                entries_by_chapter[number]
                for number in sorted(entries_by_chapter)
            ]
            self._validate_complete(result, project)
            path = self.store.write_artifact(self.definition.output_name, result)
            self.store.update_stage(self.definition.task, "completed", str(path))
            return result
        except Exception as exc:
            self.store.update_stage(self.definition.task, "failed", str(exc))
            raise

    def _structure_prompt(self) -> str:
        return self.prompt + """

本次处于总体规划的结构阶段。只输出 logline、theme、ending、acts、volumes
和 chapter_map；chapter_map 必须是空数组。不要提前列章节，也不要为了填充
chapter_map 缩减其他结构字段。系统会在保留本次结构的前提下，另行分批生成
完整章节映射。
"""

    def _chapter_map_prompt(self) -> str:
        return self.prompt + """

本次只补齐总体规划中的一批 chapter_map，不重写总纲结构。只返回一个 JSON
对象，且仅含 chapter_map 数组。chapter_map 中每项必须含 chapter_number（整数）
和 purpose（非空字符串）；保留需要的 volume、relationship_shift、
lived_detail、choice_and_cost、intensity、ending_mode、intimacy_level 等细节。
只能输出请求的章节编号，每个编号恰好一项，不能漏章、重复、越界，也不要输出
logline、theme、ending、acts、volumes 或任何解释。
"""

    def _generate_chapter_map_batch(
        self,
        project: ProjectBrief,
        inputs: dict[str, Any],
        structure: dict[str, Any],
        requested_numbers: list[int],
    ) -> list[dict[str, Any]]:
        requested = set(requested_numbers)
        payload = {
            "task": "story_outline_chapter_map",
            "outline_mode": "chapter_map_batch",
            "project": project.to_dict(),
            "inputs": inputs,
            "story_outline_structure": {
                key: structure.get(key)
                for key in ("logline", "theme", "ending", "acts", "volumes")
            },
            "requested_chapter_numbers": requested_numbers,
            "required_keys": ["chapter_map"],
        }

        def validate(batch: dict[str, Any]) -> None:
            normalized, _ = self._normalize_chapter_map(
                batch.get("chapter_map"), project
            )
            selected = [
                entry
                for entry in normalized
                if int(entry["chapter_number"]) in requested
            ]
            returned = {int(entry["chapter_number"]) for entry in selected}
            if returned != requested:
                missing = sorted(requested - returned)
                unexpected = sorted(returned - requested)
                detail = []
                if missing:
                    detail.append("缺少 " + ", ".join(map(str, missing)))
                if unexpected:
                    detail.append("多出 " + ", ".join(map(str, unexpected)))
                raise ValueError("章节映射批次不完整：" + "；".join(detail))
            batch["chapter_map"] = selected

        batch = _generate_validated_json(
            self.model,
            self._chapter_map_prompt(),
            payload,
            validate,
        )
        return batch["chapter_map"]

    @classmethod
    def _chapter_batches(cls, chapter_numbers: list[int]) -> list[list[int]]:
        return [
            chapter_numbers[index : index + cls.CHAPTER_MAP_BATCH_SIZE]
            for index in range(0, len(chapter_numbers), cls.CHAPTER_MAP_BATCH_SIZE)
        ]

    def _validate_structure(self, result: dict[str, Any]) -> None:
        super()._validate(result)
        for field in ("logline", "theme", "ending"):
            if not isinstance(result.get(field), str) or not result[field].strip():
                raise ValueError(f"大纲字段 {field} 必须是非空字符串")
        for field in ("acts", "volumes", "chapter_map"):
            if not isinstance(result.get(field), list):
                raise ValueError(f"大纲字段 {field} 必须是数组")

    def _validate_complete(
        self, result: dict[str, Any], project: ProjectBrief
    ) -> None:
        self._validate_structure(result)
        normalized, _ = self._normalize_chapter_map(result.get("chapter_map"), project)
        result["chapter_map"] = normalized
        expected = set(range(1, project.chapter_count + 1))
        actual = {int(entry["chapter_number"]) for entry in normalized}
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            detail = []
            if missing:
                detail.append("缺少 " + ", ".join(map(str, missing)))
            if unexpected:
                detail.append("越界 " + ", ".join(map(str, unexpected)))
            raise ValueError("章节映射未覆盖全部章节：" + "；".join(detail))

    @staticmethod
    def _normalize_chapter_map(
        chapter_map: Any, project: ProjectBrief
    ) -> tuple[list[dict[str, Any]], bool]:
        if not isinstance(chapter_map, list):
            raise ValueError("大纲字段 chapter_map 必须是数组")

        normalized: list[dict[str, Any]] = []
        seen: set[int] = set()
        changed = False
        for index, item in enumerate(chapter_map, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"章节映射第 {index} 项必须是对象")
            raw_number = item.get("chapter_number", item.get("chapter"))
            if isinstance(raw_number, bool):
                raise ValueError(f"章节映射第 {index} 项章节编号无效")
            try:
                chapter_number = int(raw_number)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"章节映射第 {index} 项缺少章节编号") from exc
            if isinstance(raw_number, float) and not raw_number.is_integer():
                raise ValueError(f"章节映射第 {index} 项章节编号无效")
            if not 1 <= chapter_number <= project.chapter_count:
                raise ValueError(
                    f"章节映射第 {index} 项章节编号 {chapter_number} 超出范围"
                )
            if chapter_number in seen:
                raise ValueError(f"章节映射章节 {chapter_number} 重复")
            seen.add(chapter_number)

            entry = dict(item)
            purpose = str(entry.get("purpose", "")).strip()
            if not purpose:
                purpose = next(
                    (
                        str(entry.get(field, "")).strip()
                        for field in (
                            "relationship_shift",
                            "summary",
                            "choice_and_cost",
                            "lived_detail",
                        )
                        if str(entry.get(field, "")).strip()
                    ),
                    "",
                )
            if not purpose:
                raise ValueError(f"章节映射第 {index} 项缺少 purpose")
            if entry.get("chapter_number") != chapter_number or entry.get("purpose") != purpose:
                changed = True
            entry["chapter_number"] = chapter_number
            entry["purpose"] = purpose
            normalized.append(entry)

        normalized.sort(key=lambda entry: int(entry["chapter_number"]))
        return normalized, changed


class SubplotDesigner(JsonAgent):
    definition = AgentDefinition(
        task="subplot_register",
        display_name="支线与伏笔设计师",
        output_name="subplot_register",
        required_keys=("subplots", "foreshadowing", "payoff_matrix"),
        prompt=_json_prompt(
            "支线与伏笔设计师",
            "只创建能够推动人物成长或对主线冲突施加压力的支线。每个伏笔都必须指定埋设章节和"
            "计划回收章节，每条支线都必须有退出条件，不能无限悬置或挤占主线。",
            "subplots：对象数组，支线及其开始、发展和结束条件；foreshadowing：对象数组，"
            "伏笔及其埋设和回收位置；payoff_matrix：对象数组，伏笔与回收事件的对应关系。",
        ),
    )


class ChapterPlanner:
    task = "chapter_plan"
    prompt = _json_prompt(
        "章节规划师",
        "将已经锁定的总纲转化为一份可直接执行的单章任务书。保持前后连续，明确本章目标和核心冲突，"
        "列出必须揭示与必须隐藏的信息，并以能够推动下一章的状态变化或悬念结束。"
        "情绪状态只供下游理解，不得把愤怒、委屈、疲惫、复杂神色等标签写成可直接照抄的正文指令；"
        "scene_cards、dialogue_intents 和 emotion_actions 必须使用可见动作、具体旧事、称呼变化或未说完的话。"
        "输入中的 relationship_context 是当前权威关系状态，relationship_arcs 是全书关系方向。章节规划可以安排关系停留、"
        "细微变化或倒退，但不得默认双方同步动心，也不得用一次救援、脸红或吃醋直接跨越关系阶段。"
        "需要推进关系时，只安排可观察的称呼、距离、信息分享、承诺兑现、边界试探和对方回应，不写‘关系升温’之类抽象指令。"
        "不得在场景卡中使用‘这说明、这意味着、显然、不是A而是B’替正文解释。"
        "不得新增总纲、人物资料和前后章节没有出现的命名角色、隐藏密室、关键证物或决定性记录；"
        "需要临时人物和物件时保持无名、普通且不改变后续剧情。"
        "主线必须按触发、行动、转折、结果形成完整因果链；支线必须按触发、行动、代价、回收形成闭环，"
        "并且只能给主线施压、增加人物代价或承接既有伏笔，不能另起一个计划外的大事件。"
        "length_budget 必须以项目 chapter_word_count 为最低有效字数，给初稿预留百分之五至百分之十五的余量。"
        "项目目标达到四千字时固定规划五张可执行场景卡；较短章节规划二至五张。每张场景卡都要分配 target_chars，"
        "并写清触发、行动、阻碍、转折和结果，不能只用一句概述同时代替多个环节。"
        "校园题材不等于全部情节只能发生在校内。结合总纲和最近章节地点，在上游已安排通勤、家庭、医院、商场、"
        "校外比赛或其他校外事务时自然落实；但不得仅为更换地点或补字数新增计划外事件。"
        "章末必须停在 POV 人物能够感知的具体问题、动作、物件或警告上，不得安排镜头拉远、"
        "‘两人消失在黑暗中’或‘只留下某种声音’式空镜头收束。",
        "chapter_number：整数，章节编号；title：字符串，章节标题；objective：字符串，本章目标；"
        "conflict：字符串，核心冲突；characters：字符串数组，出场人物；required_reveals：字符串数组，"
        "必须揭示的信息；withhold：字符串数组，本章必须隐藏的信息；continuity_inputs：字符串数组，"
        "需要承接的连续性信息；ending_hook：字符串，结尾钩子；target_word_count：整数，目标字数；"
        "mainline：对象，包含非空 trigger、action、turn、result；subplot：对象，包含非空 trigger、action、cost、payoff；"
        "length_budget：对象，包含 target_chars、minimum_chars、scene_count；"
        "emotional_state_before 和 emotional_state_after：对象；scene_cards：对象数组；"
        "subtext、lived_in_details、anti_repetition、forbidden_major_additions、"
        "dialogue_information_limits、rhythm_variation、incidental_details：非空字符串数组；"
        "dialogue_intents 和 emotion_actions：对象；ending_aftertaste 和 ending_concrete_residue：字符串。"
        "每张 scene_card 必须包含 scene_number、time、location、pov、surface_goal、trigger、action、obstacle、turn、result、target_chars。"
        "场景卡只能覆盖本章目标，角色不得向知情者科普设定，章末必须落在具体动作、物件或未答问题上。"
        "ending_hook 必须与最后场景结果一致，禁止新增未在总纲中安排的重大事件。",
    )
    required_keys = (
        "chapter_number",
        "title",
        "objective",
        "conflict",
        "characters",
        "required_reveals",
        "withhold",
        "continuity_inputs",
        "ending_hook",
        "target_word_count",
        "mainline",
        "subplot",
        "length_budget",
        "emotional_state_before",
        "emotional_state_after",
        "scene_cards",
        "subtext",
        "lived_in_details",
        "dialogue_intents",
        "anti_repetition",
        "ending_aftertaste",
        "forbidden_major_additions",
        "dialogue_information_limits",
        "emotion_actions",
        "rhythm_variation",
        "incidental_details",
        "ending_concrete_residue",
    )

    def __init__(
        self,
        model: LanguageModel,
        store: ArtifactStore,
        prompt: str | None = None,
    ) -> None:
        self.model = model
        self.store = store
        self.runtime_prompt = prompt or self.prompt

    def run(
        self,
        project: ProjectBrief,
        chapter_number: int,
        inputs: dict[str, Any],
        force: bool = False,
    ) -> dict[str, Any]:
        if self.store.has_chapter_plan(chapter_number) and not force:
            return self.store.read_chapter_plan(chapter_number)

        stage = f"chapter_plan_{chapter_number:03d}"
        self.store.update_stage(stage, "running", "Chapter planner")
        payload = {
            "task": self.task,
            "chapter_number": chapter_number,
            "project": project.to_dict(),
            "inputs": inputs,
            "required_keys": list(self.required_keys),
            "non_empty_array_fields": [
                "scene_cards",
                "subtext",
                "lived_in_details",
                "anti_repetition",
                "forbidden_major_additions",
                "dialogue_information_limits",
                "rhythm_variation",
                "incidental_details",
            ],
        }
        try:
            def validate(result: dict[str, Any]) -> None:
                self._normalize_optional_arrays(result)
                missing = [key for key in self.required_keys if key not in result]
                if missing:
                    raise ValueError("章节规划缺少字段：" + ", ".join(missing))
                if int(result["chapter_number"]) != chapter_number:
                    raise ValueError(
                        f"模型返回第 {result['chapter_number']} 章，实际请求第 {chapter_number} 章"
                    )
                for field in (
                    "scene_cards",
                    "subtext",
                    "lived_in_details",
                    "anti_repetition",
                    "forbidden_major_additions",
                    "dialogue_information_limits",
                    "rhythm_variation",
                    "incidental_details",
                ):
                    if not isinstance(result[field], list) or not result[field]:
                        raise ValueError(f"章节规划字段 {field} 必须是非空数组")
                if not isinstance(result["dialogue_intents"], dict):
                    raise ValueError("章节规划字段 dialogue_intents 必须是对象")
                if not isinstance(result["emotion_actions"], dict):
                    raise ValueError("章节规划字段 emotion_actions 必须是对象")

                for section, fields in (
                    ("mainline", ("trigger", "action", "turn", "result")),
                    ("subplot", ("trigger", "action", "cost", "payoff")),
                ):
                    value = result[section]
                    if not isinstance(value, dict):
                        raise ValueError(f"章节规划字段 {section} 必须是对象")
                    empty = [
                        field
                        for field in fields
                        if not str(value.get(field, "")).strip()
                    ]
                    if empty:
                        raise ValueError(
                            f"章节规划字段 {section} 缺少完整因果环节："
                            + ", ".join(empty)
                        )

                try:
                    target_word_count = int(result["target_word_count"])
                except (TypeError, ValueError) as exc:
                    raise ValueError("章节规划字段 target_word_count 必须是整数") from exc
                if target_word_count < project.chapter_word_count:
                    raise ValueError(
                        "章节目标字数不能低于项目要求 "
                        f"{project.chapter_word_count}"
                    )

                budget = result["length_budget"]
                if not isinstance(budget, dict):
                    raise ValueError("章节规划字段 length_budget 必须是对象")
                try:
                    target_chars = int(budget["target_chars"])
                    minimum_chars = int(budget["minimum_chars"])
                    scene_count = int(budget["scene_count"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "length_budget 必须包含整数 target_chars、minimum_chars、scene_count"
                    ) from exc
                if minimum_chars < project.chapter_word_count:
                    raise ValueError(
                        "length_budget.minimum_chars 不能低于项目要求 "
                        f"{project.chapter_word_count}"
                    )
                buffered_minimum = (minimum_chars * 105 + 99) // 100
                if target_chars < buffered_minimum:
                    raise ValueError(
                        "length_budget.target_chars 至少应为最低字数的 105%，"
                        f"当前要求不低于 {buffered_minimum}"
                    )

                cards = result["scene_cards"]
                if len(cards) > 5:
                    raise ValueError("单章场景卡不能超过 5 个，避免事件密度失控")
                required_scene_count = 5 if project.chapter_word_count >= 4000 else 2
                if len(cards) < required_scene_count:
                    if required_scene_count == 5:
                        raise ValueError("四千字及以上章节必须规划 5 张场景卡")
                    raise ValueError("单章至少需要 2 张可执行场景卡")
                if scene_count != len(cards):
                    raise ValueError(
                        "length_budget.scene_count 必须与 scene_cards 数量一致"
                    )

                scene_budget_total = 0
                required_scene_fields = (
                    "scene_number",
                    "time",
                    "location",
                    "pov",
                    "surface_goal",
                    "trigger",
                    "action",
                    "obstacle",
                    "turn",
                    "result",
                    "target_chars",
                )
                for index, card in enumerate(cards, start=1):
                    if not isinstance(card, dict):
                        raise ValueError(f"第 {index} 张场景卡必须是对象")
                    empty = [
                        field
                        for field in required_scene_fields
                        if field not in card
                        or card[field] is None
                        or (isinstance(card[field], str) and not card[field].strip())
                    ]
                    if empty:
                        raise ValueError(
                            f"第 {index} 张场景卡缺少可执行字段："
                            + ", ".join(empty)
                        )
                    try:
                        scene_target = int(card["target_chars"])
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"第 {index} 张场景卡 target_chars 必须是正整数"
                        ) from exc
                    if scene_target <= 0:
                        raise ValueError(
                            f"第 {index} 张场景卡 target_chars 必须是正整数"
                        )
                    scene_budget_total += scene_target
                if scene_budget_total < target_chars:
                    raise ValueError(
                        "各场景 target_chars 总和不能低于 length_budget.target_chars"
                    )

            result = _generate_validated_json(
                self.model, self.runtime_prompt, payload, validate
            )
            path = self.store.write_chapter_plan(chapter_number, result)
            self.store.update_stage(stage, "completed", str(path))
            return result
        except Exception as exc:
            self.store.update_stage(stage, "failed", str(exc))
            raise

    @staticmethod
    def _normalize_optional_arrays(result: dict[str, Any]) -> None:
        """Fill commonly omitted non-empty array fields with safe defaults.

        Models often return complete scene cards but drop secondary arrays such as
        rhythm_variation. Prefer a conservative default over failing the whole plan.
        """
        defaults: dict[str, list[str]] = {
            "subtext": ["关键对话保留未说出口的试探或回避"],
            "lived_in_details": ["一个与人物习惯相关的具体生活动作"],
            "anti_repetition": ["避免与近几章相同的冲突结构、结束方式与亲密套路"],
            "forbidden_major_additions": [
                "不新增计划外命名角色、重大证物、能力跃迁或世界真相"
            ],
            "dialogue_information_limits": [
                "角色不向知情者科普双方已知事实，不连续讲解多个设定术语"
            ],
            "rhythm_variation": [
                "至少一处短句加速",
                "至少一处打断或答非所问",
                "至少一处停在具体动作上的放慢",
            ],
            "incidental_details": ["一处简短物理不便或路人向琐事，全章不超过两处"],
        }
        for field, default_value in defaults.items():
            value = result.get(field, None)
            if value is None:
                result[field] = list(default_value)
                continue
            if isinstance(value, str):
                text = value.strip()
                result[field] = [text] if text else list(default_value)
                continue
            if not isinstance(value, list):
                result[field] = list(default_value)
                continue
            cleaned = [item for item in value if str(item).strip()]
            result[field] = cleaned if cleaned else list(default_value)
        if "dialogue_intents" not in result or not isinstance(
            result.get("dialogue_intents"), dict
        ):
            result["dialogue_intents"] = {
                "default": "围绕眼前具体事务接话，保留潜台词，不跳到关系判决句"
            }
        if "emotion_actions" not in result or not isinstance(
            result.get("emotion_actions"), dict
        ):
            result["emotion_actions"] = {
                "pressure": "用角色特有的具体小动作表现压力，不用神色标签"
            }
        if not str(result.get("ending_concrete_residue", "")).strip():
            ending_hook = str(result.get("ending_hook", "")).strip()
            result["ending_concrete_residue"] = ending_hook or (
                "章末留下一个具体未完成动作、半句话或物件状态"
            )
        if not str(result.get("ending_aftertaste", "")).strip():
            result["ending_aftertaste"] = str(
                result.get("ending_concrete_residue", "")
            ).strip()


GENRE_WRITER_AGENT_IDS = {
    "xuanhuan": "chapter_writer_xuanhuan",
    "campus": "chapter_writer_campus",
    "urban": "chapter_writer_urban",
}

GENRE_WRITER_LABELS = {
    "xuanhuan": "玄幻正文智能体",
    "campus": "校园正文智能体",
    "urban": "都市正文智能体",
}

XUANHUAN_WRITER_PROMPT = """你负责玄幻、仙侠、修炼与高武方向的章节正文。

题材专用规则：
- 力量体系必须通过行动、代价、失败和旁人的反应体现，禁止让角色完整背诵境界、功法或法器定义。
- 战斗必须写清人物位置、距离、先后手、招式作用和受伤结果。华丽名称不能代替动作因果。
- 突破、秘宝、血脉和机缘必须服从章节计划与世界规则，不得为了制造爽点临时增加能力或关键道具。
- 宏大词汇只用于真正改变局势的节点。普通交手不要反复使用天地变色、大道轰鸣、万物寂静等套话。
- 修炼资源、伤势、宗门身份和强弱差距要产生现实限制，角色不能在下一段无代价恢复。
- 设定信息优先写成个人经验、吃过的亏、具体规矩、称呼变化或不愿回答的问题，并允许读者暂时不知道全貌。
- 在洞府、宗门、坊市、荒野等场景中加入能够触碰和使用的生活细节，让人物像在真实空间里行动。
- 人物关系、感情线和主角边界严格服从项目约束，不因玄幻题材自动增加后宫、暧昧对象或宿命关系。
- 关系阶段通过交付后背、是否共享功法与弱点、宗门立场代价、称呼和疗伤边界逐步变化；一次并肩作战不自动等于动心。
"""

CAMPUS_WRITER_PROMPT = """你负责中学、大学及校园成长方向的章节正文。

题材专用规则：
- 场景必须符合课程、作息、考试、社团、宿舍、食堂、操场和校规等现实条件，时间与移动距离要可信。
- 校园题材不等于所有场景都在校内。scene_cards 已安排家庭、通勤、医院、商场、校外比赛、兼职或社区事务时，
  完整写出这些校外场景及其现实限制；不得擅自把它们改回教室，也不得为了补字数新增计划外地点。
- 对话符合人物年龄、家庭环境和关系远近。允许口误、冷场、玩笑落空、临时改口和不愿承认的小心思。
- 不让学生使用成熟职场汇报腔、心理咨询术语或替作者总结青春意义，老师也不能只承担宣布剧情的功能。
- 感情推进依靠称呼、座位、借还物品、消息回复、共同任务和对同一件小事的不同反应，不用偶像剧式宣言代替积累。
- 校园矛盾要有具体来源，例如名额、成绩、误会、家庭压力、集体规则或同伴关系，禁止全员无理由恶意。
- 使用可辨认的日常细节，但不要机械罗列品牌、课表和校园设施；细节必须与人物当下动作发生关系。
- 保留青春人物判断不完整、嘴硬、冲动和事后补救的空间，不要让每个人都及时说出最正确的话。
- 涉及未成年人时保持关系与行为边界，所有感情线和单女主等要求严格服从项目约束。
- 同一件小事在不同阶段要有不同处理：陌生时礼貌归还，相识后顺手代拿，暧昧时记住偏好又找理由掩饰，相恋后形成自然习惯。
"""

URBAN_WRITER_PROMPT = """你负责现代都市、职场、现实生活及城市情感方向的章节正文。

题材专用规则：
- 职业行为必须符合基本流程、权限和时间成本。角色不能只靠一句命令解决需要多人协作或正式手续的问题。
- 收入、住房、通勤、消费、家庭责任和人情往来应形成实际约束，但不要把正文写成行业报告或价格清单。
- 城市通过具体地点、交通、天气、营业时间和公共空间被感知，禁止只用霓虹、高楼、车流概括都市氛围。
- 对话体现职位差异、熟悉程度和利益关系；真正重要的话可以绕开、晚说或只说一半。
- 避免总裁、豪车、顶级公寓、天才履历等身份标签堆叠。人物价值由选择、能力和代价体现。
- 商战、职场冲突和社会事件必须有证据链与后果，不使用偶遇、偷拍视频或万能联系人连续推动剧情。
- 感情线应嵌入工作与生活安排，通过时间分配、承诺兑现和具体照顾累积，不用强势占有或误会拖延代替关系发展。
- 保留普通生活中的尴尬、疲惫、算计和体面，人物不需要每次都做出最聪明、最完整的回应。
- 关系变化通过是否为对方腾出时间、进入私人生活圈、承担现实成本和兑现具体承诺体现；忙碌中的实际选择比宣言更重要。
"""

GENRE_WRITER_PROMPTS = {
    "xuanhuan": XUANHUAN_WRITER_PROMPT,
    "campus": CAMPUS_WRITER_PROMPT,
    "urban": URBAN_WRITER_PROMPT,
}


def compose_chapter_writer_prompt(
    shared_prompt: str,
    genre_prompt: str,
    genre_label: str,
) -> str:
    return (
        shared_prompt.strip()
        + "\n\n## 当前题材专用规则\n\n"
        + f"本章由{genre_label}执行。以下规则在共享正文规则基础上追加：\n\n"
        + genre_prompt.strip()
    )


class ChapterWriter:
    task = "chapter_draft"
    extension_suffix = """

当前任务不是重写章节。现有正文已经通过计划和连续性边界，只因长度不足而需要续写。
只输出应当紧接在现有正文末尾的补写段落：不要标题，不要复述现有段落，不要说明补写意图。
补写只能展开 chapter_plan 中已经存在的场景、动作、对话、阻碍、选择与即时余波；不得新增地点、人物、证据、设定或下一章事件。
先补足指定缺口并留出少量余量，再检查每个细节都能被 POV 当下感知。
"""
    revision_suffix = """

当前任务是对已经生成的正文做逐段去 AI 味改稿，不是继续剧情，也不是点评。
必须完整保留章节标题、POV、章节计划明确列出的场景顺序、事实、人物知识边界和章末事件。
原稿自行增加、但 chapter_plan.scene_cards、required_reveals 和 ending_hook 没有列出的新证据、新物件、
新人物关系、新交易记录、新能力和新剧情节点不属于需要保留的事实，必须删除或改回计划已有内容。
relationship_context 和 relationship_beat 也是硬边界：只能保留本章获准的关系动作，突然昵称、交底、占有欲、肢体接触和阶段跃迁必须收回。
不要维护原句面子，可以删除解释文字，再用更具体的行为和对话重新连接。改稿后的有效正文不得低于
chapter_plan.length_budget.minimum_chars；原稿已经达标时优先保持原稿的 90% 至 105%，原稿不足时必须在既有场景内补足到最低字数以上。
补写只能展开既有场景的触发、行动、阻碍、转折、结果，以及选择后的即时余波；不得新增地点、事件、证据、人物或下一章内容凑字数。

先把 chapter_plan 视为后台工作单：objective、emotional_state、subtext 和 ending_aftertaste 只供理解，
绝不能把其中的情绪名称、解释句和总结语直接搬进正文。对每句话做两次检查：
- 摄像机检查：镜头能否准确拍到？“神色凝重、眼神变了、说得平淡、显然感知到”拍不到，必须改成可见行为或删掉。
- 解释检查：这句话是在呈现事实，还是在告诉读者该怎样理解上一句？属于后者就删除，不要换一个同义解释。

逐段执行：
1. 把向读者解释设定的台词改成角色自己的经历、抱怨、证据、说漏嘴或不完整回答；允许少给信息。
2. 删除愤怒、委屈、疲惫、复杂神色等情绪标签，用当前人物特有的具体动作替代。
3. 打破均匀问答，加入恰当的短答、打断、跑题、等待落空或具体旧记忆，但不要机械地每段插入废话。
4. 删除“这说明、这意味着、显然、看来、不是A而是B”等解释性尾巴。
5. 删除“她说得很平淡、目光里带着、神色凝重、脸色变了、眼神变了、像是怕他不相信”等作者判词。
   如果前面已有手指、食物、衣角、呼吸或说话节奏，就让那个细节独立成立。
6. 用一个精准动作替代多个概括动作，减少皱眉、攥拳、顿了顿、呼吸一滞和沉默几秒。
7. “苦笑了一下、没有露出意外、声音低了些、眼神里有惊讶/警戒/犹豫”都属于情绪标签，删除或改成具体动作。
8. 全文检查“像是、仿佛、某种”。除确实具体且不可替代的一处外全部删除，尤其禁止“像是有什么东西在看他”
   “像是在判断他的意图”“规律得像呼吸”等氛围套话和作者猜测。
9. 超过约 40 字、同时解释多个术语或因果的台词视为百科台词。拆成追问、短答、拒答和一件具体旧事，
   不得只在长台词中加入动作后原样保留科普。
10. 删除“他在脑海里梳理信息”后接名词清单的摘要式推理。让人物只抓住当前最要命的一件事并作选择。
11. 让环境保持物理存在：人物会挪开滴水、拍掉碎屑、嫌弃食物或被工具硌到，不解释其象征意义。
12. 把诗意预告片式结尾改成具体未答问题、物件、称呼、半句话或系统警告，同时保留章节计划指定的章末事实。
    到达这个终点后立即停笔，禁止再写“两人消失在黑暗里”“只留下空房间和某种声音”的镜头拉远。
13. 检查人物关系阶段。只使用 relationship_beat 选定的少量行为信号，保留 A 对 B 与 B 对 A 的认知差异；
    不得把普通合作改成暧昧，也不得为了修文风新增拥抱、吃醋、告白、昵称或共享秘密。

改写范式：
- “某术是一种封印术，需要某境界才能施放”改成“锁住了。六天，我试了十一种办法，一个没开”。
- “她很愤怒又委屈”改成她反复掰碎手里的食物、擦掉已经不存在的污迹或把一个结系了又拆。
- “这说明他早有准备”直接删除，让前面的具体证据独立存在。
- “她说得很快，像是怕他不相信”改成对方等了一会儿，她又补了一条具体证据；不要写作者判断。
- “他在脑海里梳理：圣女、封印、魔道、灵脉”改成“他把手从剑柄上拿开，又放了回去”，随后直接作决定。
- “有什么在深处觉醒”改成角色回头什么也没看见，随后留下一个具体未答问题或重复的滴水声。

只能返回改写后的完整 Markdown 正文。不得输出修改说明、对照、点评、场景小标题或检查清单。
"""
    prompt = """你是多智能体小说生产系统中的正文写作智能体。

你的任务是写出指定章节的完整正文，不是大纲、创作说明或点评。必须遵守章节任务书、权威世界规则、
人物动机、必须隐藏的信息、项目指定语言和目标文风。不得改变已经锁定的结局，不得擅自添加输入资料中
不存在且会影响后续剧情的重大设定。正文应直接从具体场景开始，保持事件之间的因果推进，并以章节
任务书指定的钩子或具有同等推动力的状态变化结束。只能返回 Markdown 正文，第一行必须使用一级标题
书写章节名称，不得在正文前后附加解释、分析、免责声明或写作总结。

正文硬规则：
- 严格执行 length_budget 和每张 scene_card.target_chars。初稿以 length_budget.target_chars 为目标，最低不得低于
  length_budget.minimum_chars；没有 length_budget 的旧章纲以 target_word_count 为最低字数。
- 约四千字章节按五张场景卡完整展开。每场都写清触发、行动、阻碍、转折和结果，不能把整场压缩成一句转述。
- 全章篇幅原则上由主线承担约 60%、支线承担约 25%、生活细节与场景过渡承担约 15%。支线只能给主线施压、
  增加代价或承接既有伏笔，不能另起计划外事件。
- 篇幅不足时补全既有场景中的动作过程、对话攻防、错误判断、选择代价和事后余波；禁止重复解释、同义改写、
  堆砌环境描写或增加计划外事件来凑字数。
- 角色不能替作者向读者定义设定。需要交代信息时，使用个人经历、具体证据、追问、回避和不完整回答。
- 不直接命名情绪，也不用“神色凝重、眼神变了、目光里带着、说得很平淡、像是怕他不相信”等作者判词。
- 一个具体动作写完就停，不追加“这说明、这意味着、显然、看来、不是A而是B”等解释性尾巴。
- 对话允许短答、打断、跑题和等待落空，不能每句话都完整推进剧情或连续讲解多个术语。
- “像是、仿佛、某种”合计最多保留一处，且不得用来猜测人物心理或制造“有什么在注视”的模糊氛围。
- 环境细节必须能与人物身体发生关系，不承担抽象象征；同一个滴水、火光等细节不能反复充当情绪配乐。
- 章末停在具体问题、动作、物件、称呼或警告上。到达章节计划终点后立即结束，不写镜头拉远式总结。
- 只写章节计划明确安排的剧情节点。未列出的关键证据、法器、记录、秘密关系和新能力不得自行补充。
- relationship_context 是当前权威关系状态，relationship_beat 是本章允许发生的关系变化。只选择其中 1 到 3 个具体信号自然写入场景，
  不得直接解释“关系更近、气氛暧昧、意识到动心”，不得让双方同步产生相同感受，也不得越过称呼、秘密和肢体边界。
- 允许本章关系不变、单向靠近、误判或倒退。阶段标签只供后台理解，正文必须用称呼、距离、等待、承诺、顺手动作和具体回应呈现。
"""

    def __init__(
        self,
        model: LanguageModel,
        store: ArtifactStore,
        prompt: str | None = None,
        genre_prompt: str | None = None,
        writer_agent_id: str = "chapter_draft",
        genre_label: str = "通用正文智能体",
    ) -> None:
        self.model = model
        self.store = store
        self.writer_agent_id = writer_agent_id
        self.genre_label = genre_label
        shared_prompt = prompt or self.prompt
        self.runtime_prompt = (
            compose_chapter_writer_prompt(
                shared_prompt,
                genre_prompt,
                genre_label,
            )
            if genre_prompt
            else shared_prompt
        )

    def run(
        self,
        project: ProjectBrief,
        chapter_number: int,
        inputs: dict[str, Any],
        force: bool = False,
    ) -> str:
        if self.store.has_chapter(chapter_number) and not force:
            return self.store.read_chapter(chapter_number)

        stage = f"chapter_draft_{chapter_number:03d}"
        self.store.update_stage(stage, "running", "Chapter writer")
        payload = {
            "task": self.task,
            "chapter_number": chapter_number,
            "project": project.to_dict(),
            "inputs": inputs,
            "writer_agent_id": self.writer_agent_id,
            "writer_genre": project.writing_genre,
        }
        try:
            issues: list[str] = []
            draft = ""
            chapter_plan = inputs["chapter_plan"]
            minimum_chars = self._minimum_content_chars(chapter_plan)
            for _ in range(2):
                request_payload = dict(payload)
                if issues:
                    request_payload["style_repair"] = (
                        "上一版正文未通过质量检查（包括最低字数），请完整重写，不要解释。问题："
                        + "；".join(issues)
                    )
                draft = self.model.generate_text(
                    self.runtime_prompt, request_payload
                ).strip()
                if not draft:
                    issues = ["正文为空"]
                    continue
                if not draft.startswith("# "):
                    title = chapter_plan["title"]
                    draft = f"# {title}\n\n{draft}"
                issues = self._quality_issues(draft, chapter_plan)
                if not issues:
                    break
            if not draft:
                raise ValueError("正文连续两次生成为空")

            revision_issues: list[str] = list(issues)
            revised = ""
            best_revised = ""
            best_revision_issues: list[str] | None = None
            best_revision_rank: tuple[int, int, int] | None = None
            for revision_attempt in range(3):
                revision_payload = {
                    "task": "chapter_draft_humanize",
                    "chapter_number": chapter_number,
                    "project": project.to_dict(),
                    "chapter_plan": chapter_plan,
                    "draft": draft,
                }
                if revision_issues:
                    revision_payload["style_repair"] = (
                        "上一版仍未通过检查。必须逐条处理下列问题，列出的原句不得原样保留，"
                        "也不能只换成同义的神态或解释句。请重新输出完整正文。问题："
                        + "；".join(revision_issues)
                    )
                revision_model: LanguageModel = self.model
                if isinstance(self.model, OpenAICompatibleClient):
                    revision_model = replace(
                        self.model,
                        temperature=0.35 if revision_attempt == 0 else 0.2,
                    )
                revised = revision_model.generate_text(
                    self.runtime_prompt + self.revision_suffix,
                    revision_payload,
                ).strip()
                if not revised:
                    revision_issues = ["改稿结果为空"]
                    continue
                if not revised.startswith("# "):
                    revised = f"# {chapter_plan['title']}\n\n{revised}"
                revision_issues = self._quality_issues(revised, chapter_plan)
                revised_chars = self._effective_content_chars(revised)
                shortfall = max(0, minimum_chars - revised_chars)
                revision_rank = (
                    1 if shortfall else 0,
                    len(revision_issues),
                    shortfall,
                )
                if best_revision_rank is None or revision_rank < best_revision_rank:
                    best_revised = revised
                    best_revision_issues = list(revision_issues)
                    best_revision_rank = revision_rank
                if not revision_issues:
                    break
            if best_revised:
                draft = best_revised
                revision_issues = best_revision_issues or []
            # A short otherwise-valid draft benefits from a bounded continuation
            # of its existing scenes, rather than another full-chapter rewrite.
            for _ in range(2):
                current_chars = self._effective_content_chars(draft)
                shortfall = max(0, minimum_chars - current_chars)
                if not shortfall:
                    break
                requested_chars = shortfall + max(180, shortfall // 4)
                extension_payload = {
                    "task": "chapter_draft_extend",
                    "chapter_number": chapter_number,
                    "project": project.to_dict(),
                    "chapter_plan": chapter_plan,
                    "draft": draft,
                    "minimum_content_chars": minimum_chars,
                    "current_content_chars": current_chars,
                    "shortfall_chars": shortfall,
                    "required_additional_chars": requested_chars,
                    "style_repair": (
                        f"当前正文有效字数为 {current_chars}，距离最低要求还差 {shortfall}。"
                        f"只补写既有场景，追加正文至少 {requested_chars} 个有效字；"
                        "不要标题、复述、说明或新增计划外事件。"
                    ),
                }
                extension = self.model.generate_text(
                    self.runtime_prompt + self.extension_suffix,
                    extension_payload,
                ).strip()
                extension = re.sub(
                    r"\A# [^\n]*(?:\n|\Z)", "", extension, count=1
                ).strip()
                if not extension:
                    continue
                candidate = f"{draft.rstrip()}\n\n{extension}"
                if self._effective_content_chars(candidate) > current_chars:
                    draft = candidate
            draft = re.sub(
                r"(?m)^##+\s*(?:场景|Scene)[^\n]*\n?",
                "",
                draft,
                flags=re.I,
            )
            final_chars = self._effective_content_chars(draft)
            if minimum_chars and final_chars < minimum_chars:
                raise ValueError(
                    "正文有效字数不足："
                    f"当前 {final_chars}，最低 {minimum_chars}。"
                    "连续重写后仍未达标，章节文件未写入"
                )
            revision_issues = self._quality_issues(draft, chapter_plan)
            path = self.store.write_chapter(chapter_number, draft)
            detail = str(path)
            if revision_issues:
                detail += " | 文风提醒：" + "；".join(revision_issues)
            detail += f" | 执行智能体：{self.genre_label}"
            self.store.update_stage(stage, "completed", detail)
            return draft
        except Exception as exc:
            self.store.update_stage(stage, "failed", str(exc))
            raise

    @staticmethod
    def _effective_content_chars(draft: str) -> int:
        body = re.sub(r"\A# [^\n]*(?:\n|\Z)", "", draft, count=1)
        return sum(1 for character in body if not character.isspace())

    @staticmethod
    def _minimum_content_chars(chapter_plan: dict[str, Any] | None) -> int:
        if not chapter_plan:
            return 0
        budget = chapter_plan.get("length_budget")
        raw_minimum: Any = None
        if isinstance(budget, dict):
            raw_minimum = budget.get("minimum_chars")
        if raw_minimum is None:
            raw_minimum = chapter_plan.get("target_word_count")
        try:
            return max(0, int(raw_minimum))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _quality_issues(
        draft: str,
        chapter_plan: dict[str, Any] | None = None,
    ) -> list[str]:
        issues: list[str] = []

        minimum_chars = ChapterWriter._minimum_content_chars(chapter_plan)
        if minimum_chars:
            current_chars = ChapterWriter._effective_content_chars(draft)
            if current_chars < minimum_chars:
                issues.append(
                    "正文有效字数不足："
                    f"当前 {current_chars}，最低 {minimum_chars}。"
                    "必须在既有场景内补足动作、对话、选择过程和余波"
                )

        def matches(pattern: str, flags: int = 0) -> list[str]:
            found: list[str] = []
            for match in re.finditer(pattern, draft, flags):
                excerpt = re.sub(r"\s+", " ", match.group(0)).strip()
                if len(excerpt) > 46:
                    excerpt = excerpt[:43] + "..."
                if excerpt:
                    found.append(excerpt)
            return found

        def format_examples(found: list[str], limit: int = 2) -> str:
            unique = list(dict.fromkeys(found))
            return " / ".join(f"“{item}”" for item in unique[:limit])

        def add_pattern_issue(
            label: str,
            pattern: str,
            limit: int = 0,
            flags: int = 0,
        ) -> None:
            found = matches(pattern, flags)
            if len(found) <= limit:
                return
            examples = format_examples(found)
            issues.append(f"{label}，例如：{examples}")

        if re.search(r"(?m)^##+\s*(?:场景|Scene)", draft, re.I):
            issues.append("正文包含场景规划小标题")

        checks = (
            (
                "存在替读者下结论的解释词",
                r"这(?:说明|意味着)|不难看出|显而易见|(?<!不)显然",
                0,
            ),
            ("模板化顿悟句", r"这一刻.{0,24}(?:明白|意识到)|命运的齿轮", 0),
            ("使用解释性不是而是句式", r"不是[^。！？\n]{0,60}而是", 0),
            (
                "对话包含说明书式定义",
                r"“[^”]{0,30}(?:是一种|职责[，,]?是|用途[，,]?是|一般用于|需要[^”]{0,24}才能)[^”]{0,120}”",
                0,
            ),
            (
                "使用复杂神色或声音情绪标签",
                r"眼神(?:里|中).{0,24}(?:复杂|情绪)|"
                r"声音(?:里|中).{0,16}(?:有|带着|透着).{0,16}(?:惊讶|警戒|困惑|恳切|警惕|认真|犹豫|疲惫|愤怒|委屈|复杂)|"
                r"(?:眼神|目光)(?:里|中)?.{0,16}(?:有|带着|透着|闪过).{0,20}"
                r"(?:惊讶|警戒|困惑|恳切|警惕|认真|犹豫|疲惫|愤怒|委屈|复杂)|"
                r"(?:神色凝重|眼神变了|脸色(?:突然)?变了|说得很平淡|苦笑了一下|"
                r"声音(?:很低|低了些|更低了|有点哑)|"
                r"语气(?:有点|很|变得)(?:硬|冷|急|沉|轻)|"
                r"没有露出[^。！？\n]{0,20}(?:神色|表情))",
                0,
            ),
            (
                "动作后追加作者猜测",
                r"[，,](?:像是|仿佛)(?:怕|担心|生怕|唯恐)[^。！？\n]{0,40}",
                0,
            ),
            (
                "像是、仿佛或某种等模糊套话过多",
                r"像是|仿佛|某种",
                1,
            ),
            (
                "使用模糊注视或作者判断式比喻",
                r"(?:像是|仿佛)有什(?:么|麼)(?:东西|存在)|"
                r"(?:像是|仿佛)在(?:判断|衡量|猜测)[^。！？\n]{0,24}|"
                r"(?:滴水声|风声|雨声)[^。！？\n]{0,24}(?:像|仿佛)[^。！？\n]{0,20}",
                0,
            ),
            (
                "使用换皮后的情绪或电影化套话",
                r"声音里(?:裂开|裂了)一道缝|"
                r"眼睛[^。！？\n]{0,14}亮了一下[^。！？\n]{0,14}暗下去|"
                r"烙在(?:他|她)?(?:的)?视网膜上|"
                r"火光[^。！？\n]{0,28}投下(?:一半|半边)阴影",
                0,
            ),
            (
                "使用摘要式推理代替人物当下反应",
                r"(?:在脑海里)?(?:快速)?(?:梳理|整理|分析)(?:着)?(?:这些)?(?:信息|线索)|"
                r"(?:[^、。！？\n]{1,12}、){2,}[^—。！？\n]{1,18}——这(?:几个|些)",
                0,
            ),
        )
        for label, pattern, limit in checks:
            add_pattern_issue(label, pattern, limit)

        generic_pattern = (
            r"顿了顿|沉默(?:了)?(?:几秒|片刻|很久)|好一会儿没说话|"
            r"皱(?:了皱)?眉|呼吸顿(?:了)?一下|神色凝重|眼神变了|脸色(?:突然)?变了|"
            r"(?:停|顿)了一下|没(?:有)?(?:马上|立刻)(?:回答|接话|开口)|犹豫了一下"
        )
        generic_reactions = matches(generic_pattern)
        if len(generic_reactions) > 2:
            examples = format_examples(generic_reactions, 3)
            issues.append(f"通用反应累计超过两次，例如：{examples}")

        expository_dialogue: list[str] = []
        technical = re.compile(
            r"封印|功法|术式|阵法|法器|灵力|金丹|筑基|元婴|宗门|长老|魔道|体质|灵脉|系统"
        )
        explanatory = re.compile(
            r"如果|因为|所以|因此|不可能|那是|也就是|需要|用于|能够|原理|记载|意味着"
        )
        for quote in re.finditer(r"“([^”]{40,})”", draft):
            speech = quote.group(1)
            technical_count = len(technical.findall(speech))
            if technical_count < 2:
                continue
            if len(speech) < 70 and not explanatory.search(speech):
                continue
            excerpt = re.sub(r"\s+", " ", speech).strip()
            if len(excerpt) > 46:
                excerpt = excerpt[:43] + "..."
            expository_dialogue.append(excerpt)
        if expository_dialogue:
            examples = " / ".join(f"“{item}”" for item in expository_dialogue[:2])
            issues.append(f"对话仍像百科式长说明，应拆成短答和具体证据，例如：{examples}")

        if chapter_plan:
            plan_text = json.dumps(chapter_plan, ensure_ascii=False)
            body_text = re.sub(r"\A# [^\n]*\n?", "", draft, count=1)
            unexpected_names: list[str] = []
            for match in re.finditer(
                r"(?:叫|名叫|名为|自称)(?!(?:我|你|他|她|它|什么|哪个|谁))"
                r"[：:，,\s]*[“‘\"]?"
                r"([\u4e00-\u9fff]{2,6})(?=[”’\"，。！？、的\s])",
                draft,
            ):
                name = match.group(1)
                if name not in plan_text and name not in unexpected_names:
                    unexpected_names.append(name)
            if unexpected_names:
                issues.append(
                    "正文新增章节计划未列出的命名角色或称谓，例如："
                    + " / ".join(f"“{name}”" for name in unexpected_names[:3])
                )

            required_text = " ".join(
                str(item) for item in chapter_plan.get("required_reveals", [])
            )
            identity_terms = (
                "圣女",
                "圣子",
                "宗主",
                "掌门",
                "皇帝",
                "皇后",
                "太子",
                "公主",
                "王爷",
                "凶手",
                "卧底",
                "警察",
                "医生",
                "教师",
            )
            missing_terms = [
                term
                for term in identity_terms
                if term in required_text and term not in body_text
            ]
            if missing_terms:
                issues.append(
                    "遗漏章节计划要求本章明确揭示的身份词："
                    + "、".join(missing_terms)
                )

            leaked_terms: list[str] = []
            for item in chapter_plan.get("withhold", []):
                for term in re.findall(r"[‘“]([^’”]{2,24})[’”]", str(item)):
                    if term in body_text and term not in leaked_terms:
                        leaked_terms.append(term)
            if leaked_terms:
                issues.append(
                    "泄露章节计划要求隐藏的明确词语："
                    + "、".join(leaked_terms[:3])
                )

        tail = draft[-500:]
        if re.search(
            r"像是.{0,30}(?:觉醒|苏醒)|真正的.{0,20}才刚刚开始|"
            r"暴风雨.{0,20}(?:来临|将至)",
            tail,
        ):
            issues.append("结尾使用模板化诗意悬念句")
        cinematic_tail = re.findall(
            r"两人(?:消失|没入|走进|奔入)[^。！？\n]{0,32}(?:黑暗|夜色)|"
            r"只留下|留下身后|身后那处|"
            r"(?:滴水声|风声|雨声)[^。！？\n]{0,32}(?:仍然|依旧)[^。！？\n]{0,20}(?:规律|响着|落下)",
            tail,
        )
        if cinematic_tail:
            excerpt = re.sub(r"\s+", " ", cinematic_tail[0]).strip()
            issues.append(f"结尾使用镜头拉远式空景，应在具体事件处停笔，例如：“{excerpt}”")
        return issues[:10]
