from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agents import (
    ChapterPlanner,
    ChapterWriter,
    CharacterDesigner,
    GENRE_WRITER_AGENT_IDS,
    GENRE_WRITER_LABELS,
    GENRE_WRITER_PROMPTS,
    GenreAnalyst,
    OutlineArchitect,
    SubplotDesigner,
    WorldBuilder,
)
from .artifacts import ArtifactStore
from .llm import LanguageModel
from .relationship_agents import RelationshipArchitect, RelationshipBeatPlanner
from .relationship_store import RelationshipStateStore


class NovelWorkflow:
    """Required-agent orchestrator from project brief through chapter draft."""

    # The writer needs the current scene's facts, not the full long-form plan.
    # This leaves enough room for a Chinese chapter draft on models with a
    # smaller effective context window.
    WRITER_CONTEXT_MAX_CHARS = 22_000
    WRITER_MEMORY_MAX_CHARS = 4_000
    WRITER_PREVIOUS_TAIL_MAX_CHARS = 2_000
    # Chapter planners formerly received every planning artifact in full. On
    # long projects that can exceed a model gateway's practical input window
    # before it has generated even one scene card.
    PLANNER_CONTEXT_MAX_CHARS = 28_000
    PLANNER_MEMORY_MAX_CHARS = 4_000

    def __init__(
        self,
        project_root: str | Path,
        model: LanguageModel,
        prompt_overrides: dict[str, str] | None = None,
        agent_models: dict[str, LanguageModel] | None = None,
        agent_memory_access: dict[str, dict[str, Any]] | None = None,
        memory_pipeline: Any | None = None,
    ) -> None:
        self.store = ArtifactStore(project_root)
        self.model = model
        prompts = prompt_overrides or {}
        models = agent_models or {}
        self.agent_memory_access = agent_memory_access or {}
        self.memory_pipeline = memory_pipeline
        self.relationships = RelationshipStateStore(project_root)
        self.relationships.initialize()
        self.genre_analyst = GenreAnalyst(
            models.get("genre_analysis", model), self.store, prompts.get("genre_analysis")
        )
        self.world_builder = WorldBuilder(
            models.get("world_bible", model), self.store, prompts.get("world_bible")
        )
        self.character_designer = CharacterDesigner(
            models.get("character_bible", model),
            self.store,
            prompts.get("character_bible"),
        )
        self.outline_architect = OutlineArchitect(
            models.get("story_outline", model), self.store, prompts.get("story_outline")
        )
        self.relationship_architect = RelationshipArchitect(
            models.get("relationship_arcs", model),
            self.store,
            self.relationships,
            prompts.get("relationship_arcs"),
        )
        self.subplot_designer = SubplotDesigner(
            models.get("subplot_register", model),
            self.store,
            prompts.get("subplot_register"),
        )
        self.chapter_planner = ChapterPlanner(
            models.get("chapter_plan", model), self.store, prompts.get("chapter_plan")
        )
        self.relationship_beat_planner = RelationshipBeatPlanner(
            models.get("relationship_beat", model),
            self.store,
            prompts.get("relationship_beat"),
        )
        shared_writer_model = models.get("chapter_draft", model)
        self.chapter_writers = {
            genre_key: ChapterWriter(
                models.get(agent_id, shared_writer_model),
                self.store,
                prompts.get("chapter_draft"),
                prompts.get(agent_id, GENRE_WRITER_PROMPTS[genre_key]),
                writer_agent_id=agent_id,
                genre_label=GENRE_WRITER_LABELS[genre_key],
            )
            for genre_key, agent_id in GENRE_WRITER_AGENT_IDS.items()
        }
        self.chapter_writer = self.chapter_writers["urban"]

    def plan(self, force: bool = False) -> dict[str, Any]:
        project = self.store.load_brief()
        self.store.update_stage("required_planning", "running", "Required planning agents")
        try:
            # Planning calls are deliberately sequential. Many OpenAI-compatible
            # gateways limit concurrent long responses, and planning is not CPU-bound.
            genre = self.genre_analyst.run(
                project, {"project_brief": project.to_dict()}, force
            )
            world = self.world_builder.run(
                project, {"project_brief": project.to_dict()}, force
            )

            characters = self.character_designer.run(
                project,
                {"genre_analysis": genre, "world_bible": world},
                force,
            )
            outline = self.outline_architect.run(
                project,
                {
                    "genre_analysis": genre,
                    "world_bible": world,
                    "character_bible": characters,
                },
                force,
            )
            relationships = self.relationship_architect.run(
                project,
                {
                    "genre_analysis": genre,
                    "world_bible": world,
                    "character_bible": characters,
                    "story_outline": outline,
                },
                force,
            )
            subplots = self.subplot_designer.run(
                project,
                {
                    "world_bible": world,
                    "character_bible": characters,
                    "story_outline": outline,
                    "relationship_arcs": relationships,
                },
                force,
            )

            self.store.update_stage(
                "required_planning", "completed", "Core planning artifacts"
            )
            return {
                "genre_analysis": genre,
                "world_bible": world,
                "character_bible": characters,
                "story_outline": outline,
                "relationship_arcs": relationships,
                "subplot_register": subplots,
            }
        except Exception as exc:
            self.store.update_stage("required_planning", "failed", str(exc))
            raise

    def plan_chapter(self, chapter_number: int, force: bool = False) -> Path:
        project = self.store.load_brief()
        self._validate_chapter_number(chapter_number, project.chapter_count)
        if not self.store.has_artifact("subplot_register"):
            self.plan(force=False)
        relationship_arcs = self._ensure_relationship_arcs(project)

        previous_plan = (
            self.store.read_chapter_plan(chapter_number - 1)
            if chapter_number > 1 and self.store.has_chapter_plan(chapter_number - 1)
            else None
        )
        memory_context = None
        if self.memory_pipeline is not None:
            self.memory_pipeline.assert_ready_for_chapter(chapter_number)
            outline = self.store.read_artifact("story_outline")
            chapter_map = outline.get("chapter_map", [])
            purpose = next(
                (
                    item.get("purpose", "")
                    for item in chapter_map
                    if int(item.get("chapter_number", 0)) == chapter_number
                ),
                "",
            )
            memory_context = self.memory_pipeline.retrieve(
                f"为第 {chapter_number} 章规划剧情。章节作用：{purpose}",
                chapter_number=chapter_number,
                access=self.agent_memory_access.get("chapter_plan"),
            )
        relationship_context = self.relationships.context_for_characters()
        planner_inputs = self._build_planner_inputs(
            chapter_number=chapter_number,
            previous_plan=previous_plan,
            relationship_arcs=relationship_arcs,
            relationship_context=relationship_context,
            memory_context=memory_context,
        )
        chapter_plan = self.chapter_planner.run(
            project,
            chapter_number,
            planner_inputs,
            force,
        )
        chapter_characters = self._chapter_character_names(chapter_plan)
        relationship_context = self.relationships.context_for_characters(
            chapter_characters
        )
        previous_beat = (
            self.store.read_relationship_beat(chapter_number - 1)
            if chapter_number > 1
            and self.store.has_relationship_beat(chapter_number - 1)
            else None
        )
        self.relationship_beat_planner.run(
            project,
            chapter_number,
            {
                "relationship_arcs": relationship_arcs,
                "relationship_context": relationship_context,
                "chapter_plan": chapter_plan,
                "previous_relationship_beat": previous_beat,
                "memory_context": memory_context,
            },
            force,
        )
        return self.store.chapter_plan_path(chapter_number)

    def write_chapter(self, chapter_number: int, force: bool = False) -> Path:
        project = self.store.load_brief()
        self._validate_chapter_number(chapter_number, project.chapter_count)
        if not self.store.has_chapter_plan(
            chapter_number
        ) or not self.store.has_relationship_beat(chapter_number):
            self.plan_chapter(chapter_number, force=False)

        previous_plan = (
            self.store.read_chapter_plan(chapter_number - 1)
            if chapter_number > 1
            else None
        )
        previous_tail = ""
        if chapter_number > 1 and self.store.has_chapter(chapter_number - 1):
            previous_tail = self.store.read_chapter(chapter_number - 1)[
                -self.WRITER_PREVIOUS_TAIL_MAX_CHARS :
            ]

        memory_context = None
        writer_agent_id = GENRE_WRITER_AGENT_IDS[project.writing_genre]
        chapter_writer = self.chapter_writers[project.writing_genre]
        chapter_plan = self.store.read_chapter_plan(chapter_number)
        chapter_characters = self._chapter_character_names(chapter_plan)
        relationship_context = self.relationships.context_for_characters(
            chapter_characters
        )
        relationship_beat = self.store.read_relationship_beat(chapter_number)
        if self.memory_pipeline is not None:
            self.memory_pipeline.assert_ready_for_chapter(chapter_number)
            memory_context = self.memory_pipeline.retrieve(
                f"写第 {chapter_number} 章。目标：{chapter_plan.get('objective', '')}；"
                f"冲突：{chapter_plan.get('conflict', '')}；人物：{'、'.join(chapter_characters)}；"
                f"关系节拍：{relationship_beat.get('progression_summary', '')}",
                chapter_number=chapter_number,
                characters=chapter_characters,
                access=self.agent_memory_access.get(
                    writer_agent_id,
                    self.agent_memory_access.get("chapter_draft"),
                ),
            )

        writer_inputs = self._build_writer_inputs(
            chapter_number=chapter_number,
            chapter_plan=chapter_plan,
            chapter_characters=chapter_characters,
            relationship_context=relationship_context,
            relationship_beat=relationship_beat,
            previous_plan=previous_plan,
            previous_tail=previous_tail,
            memory_context=memory_context,
        )
        chapter_writer.run(
            project,
            chapter_number,
            writer_inputs,
            force,
        )
        return self.store.chapter_path(chapter_number)

    def _build_planner_inputs(
        self,
        *,
        chapter_number: int,
        previous_plan: dict[str, Any] | None,
        relationship_arcs: dict[str, Any],
        relationship_context: dict[str, Any],
        memory_context: Any,
    ) -> dict[str, Any]:
        """Build a chapter-local planning pack rather than sending every artifact."""
        characters = self.store.read_artifact("character_bible")
        chapter_names = self._planner_character_names(characters)
        inputs = {
            "world_bible": self._planner_world_context(
                self.store.read_artifact("world_bible")
            ),
            "character_bible": self._writer_character_context(characters, chapter_names),
            "story_outline": self._writer_outline_context(
                self.store.read_artifact("story_outline"), chapter_number
            ),
            "relationship_arcs": self._writer_relationship_context(
                relationship_arcs, chapter_names, chapter_number
            ),
            "relationship_context": self._compact_relationship_state(
                relationship_context
            ),
            "subplot_register": self._writer_subplot_context(
                self.store.read_artifact("subplot_register"),
                chapter_names,
                chapter_number,
            ),
            "previous_chapter_plan": self._compact_previous_plan(previous_plan),
            "memory_context": self._limit_context(
                memory_context, self.PLANNER_MEMORY_MAX_CHARS
            ),
        }
        return self._enforce_planner_context_limit(inputs)

    def _build_writer_inputs(
        self,
        *,
        chapter_number: int,
        chapter_plan: dict[str, Any],
        chapter_characters: list[str],
        relationship_context: dict[str, Any],
        relationship_beat: dict[str, Any],
        previous_plan: dict[str, Any] | None,
        previous_tail: str,
        memory_context: Any,
    ) -> dict[str, Any]:
        """Build a chapter-local source pack for the prose writer."""
        inputs = {
            "world_bible": self._writer_world_context(
                self.store.read_artifact("world_bible"), chapter_plan
            ),
            "character_bible": self._writer_character_context(
                self.store.read_artifact("character_bible"), chapter_characters
            ),
            "story_outline": self._writer_outline_context(
                self.store.read_artifact("story_outline"), chapter_number
            ),
            "relationship_arcs": self._writer_relationship_context(
                self.store.read_artifact("relationship_arcs"), chapter_characters, chapter_number
            ),
            "relationship_context": self._compact_relationship_state(
                relationship_context
            ),
            "relationship_beat": self._compact_relationship_beat(relationship_beat),
            "subplot_register": self._writer_subplot_context(
                self.store.read_artifact("subplot_register"), chapter_characters, chapter_number
            ),
            "chapter_plan": self._writer_chapter_plan(chapter_plan),
            "previous_chapter_plan": self._compact_previous_plan(previous_plan),
            "previous_chapter_tail": previous_tail,
            "memory_context": self._limit_context(memory_context, self.WRITER_MEMORY_MAX_CHARS),
        }
        return self._enforce_writer_context_limit(inputs)

    @staticmethod
    def _chapter_character_names(chapter_plan: dict[str, Any]) -> list[str]:
        names: list[str] = []
        for item in chapter_plan.get("characters", []):
            name = item.get("name") if isinstance(item, dict) else item
            if isinstance(name, str) and name.strip() and name.strip() not in names:
                names.append(name.strip())
        return names

    @staticmethod
    def _planner_character_names(characters: dict[str, Any]) -> list[str]:
        names: list[str] = []
        for key, limit in (("protagonists", 3), ("supporting_cast", 2)):
            for item in characters.get(key, []):
                name = item.get("name") if isinstance(item, dict) else item
                if isinstance(name, str) and name.strip() and name.strip() not in names:
                    names.append(name.strip())
                if len(names) >= limit if key == "protagonists" else len(names) >= 5:
                    break
        return names

    @staticmethod
    def _chapter_entry_number(entry: dict[str, Any]) -> int | None:
        for key in ("chapter_number", "chapter"):
            try:
                return int(entry.get(key))
            except (AttributeError, TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _range_contains(value: Any, chapter_number: int) -> bool:
        if not isinstance(value, list) or len(value) != 2:
            return False
        try:
            return int(value[0]) <= chapter_number <= int(value[1])
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _mentions_any(value: Any, names: list[str]) -> bool:
        if not names:
            return False
        text = json.dumps(value, ensure_ascii=False)
        return any(name in text for name in names)

    def _writer_outline_context(
        self, outline: dict[str, Any], chapter_number: int
    ) -> dict[str, Any]:
        chapter_map = outline.get("chapter_map", [])
        nearby = [
            item
            for item in chapter_map
            if isinstance(item, dict)
            and (number := self._chapter_entry_number(item)) is not None
            and abs(number - chapter_number) <= 1
        ]
        return {
            "logline": outline.get("logline", ""),
            "theme": outline.get("theme", ""),
            "current_and_adjacent_chapters": nearby,
            "current_volume": next(
                (
                    item
                    for item in outline.get("volumes", [])
                    if isinstance(item, dict)
                    and self._range_contains(item.get("chapter_range"), chapter_number)
                ),
                None,
            ),
            "current_act": next(
                (
                    item
                    for item in outline.get("acts", [])
                    if isinstance(item, dict)
                    and self._range_contains(item.get("chapter_range"), chapter_number)
                ),
                None,
            ),
        }

    @staticmethod
    def _writer_world_context(
        world: dict[str, Any], chapter_plan: dict[str, Any]
    ) -> dict[str, Any]:
        plan_text = json.dumps(chapter_plan, ensure_ascii=False)
        locations = [
            location
            for location in world.get("locations", [])
            if isinstance(location, dict)
            and str(location.get("name", "")) in plan_text
        ]
        return {
            "premise": world.get("premise", ""),
            "rules": [
                NovelWorkflow._select_fields(
                    item, "name", "description", "limitations", "cost", "loophole"
                )
                for item in world.get("rules", [])
                if isinstance(item, dict)
            ],
            "factions": [
                NovelWorkflow._select_fields(item, "name", "description", "access")
                for item in world.get("factions", [])
                if isinstance(item, dict)
            ],
            "locations": [
                NovelWorkflow._select_fields(
                    item, "name", "sensory_cues", "access", "stay", "private_interaction"
                )
                for item in (locations or world.get("locations", [])[:3])
                if isinstance(item, dict)
            ],
        }

    @staticmethod
    def _planner_world_context(world: dict[str, Any]) -> dict[str, Any]:
        """Keep the durable rules and a small location set for scene planning."""
        return {
            "premise": world.get("premise", ""),
            "rules": [
                NovelWorkflow._select_fields(
                    item, "name", "description", "limitations", "cost", "loophole"
                )
                for item in world.get("rules", [])[:6]
                if isinstance(item, dict)
            ],
            "factions": [
                NovelWorkflow._select_fields(item, "name", "description", "access")
                for item in world.get("factions", [])[:4]
                if isinstance(item, dict)
            ],
            "locations": [
                NovelWorkflow._select_fields(
                    item, "name", "sensory_cues", "access", "stay", "private_interaction"
                )
                for item in world.get("locations", [])[:5]
                if isinstance(item, dict)
            ],
        }

    def _writer_character_context(
        self, characters: dict[str, Any], names: list[str]
    ) -> dict[str, Any]:
        cast = [
            item
            for key in ("protagonists", "supporting_cast")
            for item in characters.get(key, [])
            if isinstance(item, dict) and str(item.get("name", "")) in names
        ]
        if not cast:
            cast = [
                item for item in characters.get("protagonists", []) if isinstance(item, dict)
            ][:2]
        return {
            "chapter_cast": [
                self._select_fields(
                    item,
                    "name", "age", "occupation", "goal", "fear", "flaw",
                    "knowledge_state", "habits", "voice", "relationship_boundaries",
                )
                for item in cast
            ],
            "relationships": [
                self._select_fields(
                    item,
                    "character_a", "character_b", "dynamic", "misread",
                    "address_terms", "conflict_style", "dialogue_contrast",
                )
                for item in characters.get("relationships", [])
                if isinstance(item, dict)
                and str(item.get("character_a", "")) in names
                and str(item.get("character_b", "")) in names
            ],
            "character_arcs": [
                self._select_fields(item, "character", "arc")
                for item in characters.get("arcs", [])
                if isinstance(item, dict) and self._mentions_any(item, names)
            ],
        }

    def _writer_relationship_context(
        self, relationships: dict[str, Any], names: list[str], chapter_number: int
    ) -> dict[str, Any]:
        result = []
        for item in relationships.get("relationships", []):
            if not isinstance(item, dict) or not self._mentions_any(item, names):
                continue
            eligible_schedules = [
                schedule
                for schedule in item.get("stage_schedule", [])
                if isinstance(schedule, dict)
                and int(schedule.get("from_chapter", 0)) <= chapter_number
            ]
            current_schedule = max(
                eligible_schedules,
                key=lambda schedule: int(schedule.get("from_chapter", 0)),
                default=None,
            )
            result.append(
                {
                    key: item.get(key)
                    for key in (
                        "id", "character_a", "character_b", "relationship_type",
                        "public_status", "private_status", "a_to_b", "b_to_a",
                        "address_terms", "touch_boundary", "arc_summary", "next_milestone",
                        "forbidden_leaps", "genre_signals",
                    )
                }
                | {"current_stage_schedule": current_schedule}
            )
        return {"global_rules": relationships.get("global_rules", []), "relationships": result}

    @staticmethod
    def _compact_relationship_state(state: dict[str, Any]) -> dict[str, Any]:
        pairs = []
        for item in state.get("pairs", []) if isinstance(state, dict) else []:
            if not isinstance(item, dict):
                continue
            pairs.append(
                NovelWorkflow._select_fields(
                    item,
                    "id", "character_a", "character_b", "stage", "stage_label",
                    "target_stage", "public_status", "private_status", "a_to_b",
                    "b_to_a", "address_terms", "touch_boundary", "next_milestone",
                    "forbidden_leaps",
                )
            )
        rules = state.get("plan_rules", []) if isinstance(state, dict) else []
        return {"plan_rules": rules[:5], "pairs": pairs}

    @staticmethod
    def _compact_relationship_beat(beat: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(beat, dict):
            return beat
        return {
            key: beat.get(key)
            for key in (
                "chapter_number", "intensity", "focus_pairs", "behavior_signals",
                "callbacks", "forbidden_leaps", "progression_summary", "stage_change_allowed",
            )
            if beat.get(key) is not None
        }

    def _writer_subplot_context(
        self, subplots: dict[str, Any], names: list[str], chapter_number: int
    ) -> dict[str, Any]:
        active_subplots = []
        for item in subplots.get("subplots", []):
            if not isinstance(item, dict):
                continue
            current_progression = [
                step
                for step in item.get("progression", [])
                if isinstance(step, dict)
                and self._range_contains(step.get("chapter_range"), chapter_number)
            ]
            if current_progression or self._mentions_any(item.get("participants", []), names):
                active_subplots.append(
                    {
                        "id": item.get("id"),
                        "title": item.get("title"),
                        "participants": item.get("participants", []),
                        "current_progression": current_progression,
                    }
                )
        clues = [
            item
            for item in subplots.get("foreshadowing", [])
            if isinstance(item, dict) and chapter_number in item.get("plant_chapters", [])
        ]
        clue_ids = {str(item.get("id")) for item in clues}
        return {
            "active_subplots": active_subplots,
            "current_chapter_foreshadowing": [
                self._select_fields(item, "id", "detail", "who_notices", "who_ignores")
                for item in clues
            ],
            "relevant_payoffs": [
                item
                for item in subplots.get("payoff_matrix", [])
                if isinstance(item, dict) and str(item.get("foreshadowing_id")) in clue_ids
            ],
        }

    @staticmethod
    def _select_fields(item: dict[str, Any], *fields: str) -> dict[str, Any]:
        return {field: item[field] for field in fields if item.get(field) not in (None, "", [], {})}

    @staticmethod
    def _writer_chapter_plan(chapter_plan: dict[str, Any]) -> dict[str, Any]:
        """Keep concrete scene instructions and hard continuity constraints only."""
        return {
            key: chapter_plan.get(key)
            for key in (
                "chapter_number", "title", "objective", "conflict", "characters",
                "required_reveals", "withhold", "continuity_inputs", "ending_hook",
                "target_word_count", "mainline", "subplot", "length_budget", "scene_cards",
                "forbidden_major_additions",
                "dialogue_information_limits", "ending_concrete_residue",
            )
            if chapter_plan.get(key) is not None
        }

    @staticmethod
    def _compact_previous_plan(previous_plan: dict[str, Any] | None) -> dict[str, Any] | None:
        if not previous_plan:
            return None
        return {
            key: previous_plan.get(key)
            for key in (
                "chapter_number", "title", "objective", "required_reveals", "withhold",
                "ending_hook", "emotional_state_after", "ending_aftertaste",
            )
        } | {"last_scene": (previous_plan.get("scene_cards") or [])[-1:]}

    @staticmethod
    def _serialized_size(value: Any) -> int:
        return len(json.dumps(value, ensure_ascii=False, default=str))

    def _limit_context(self, value: Any, maximum: int) -> Any:
        if value is None or self._serialized_size(value) <= maximum:
            return value
        serialized = json.dumps(value, ensure_ascii=False, default=str)
        return {"truncated": True, "content": serialized[:maximum]}

    def _enforce_writer_context_limit(self, inputs: dict[str, Any]) -> dict[str, Any]:
        for key, limit in (
            ("memory_context", self.WRITER_MEMORY_MAX_CHARS),
            ("previous_chapter_tail", self.WRITER_PREVIOUS_TAIL_MAX_CHARS),
            ("world_bible", 2_500),
            ("character_bible", 4_000),
            ("relationship_arcs", 2_000),
            ("subplot_register", 1_500),
        ):
            if self._serialized_size(inputs) <= self.WRITER_CONTEXT_MAX_CHARS:
                break
            inputs[key] = self._limit_context(inputs.get(key), limit)
        if self._serialized_size(inputs) > self.WRITER_CONTEXT_MAX_CHARS:
            raise ValueError("正文上下文仍超过 22000 字符，已停止发送超大请求")
        return inputs

    def _enforce_planner_context_limit(self, inputs: dict[str, Any]) -> dict[str, Any]:
        for key, limit in (
            ("memory_context", self.PLANNER_MEMORY_MAX_CHARS),
            ("previous_chapter_plan", 2_000),
            ("subplot_register", 3_000),
            ("relationship_arcs", 4_000),
            ("relationship_context", 4_000),
            ("character_bible", 5_000),
            ("world_bible", 4_000),
        ):
            if self._serialized_size(inputs) <= self.PLANNER_CONTEXT_MAX_CHARS:
                break
            inputs[key] = self._limit_context(inputs.get(key), limit)
        if self._serialized_size(inputs) > self.PLANNER_CONTEXT_MAX_CHARS:
            raise ValueError("章节规划上下文仍超过 28000 字符，已停止发送超大请求")
        return inputs

    def run(self, force: bool = False) -> list[Path]:
        project = self.store.load_brief()
        self.plan(force=force)
        paths = []
        for chapter_number in range(1, project.chapter_count + 1):
            paths.append(self.write_chapter(chapter_number, force=force))
        self.store.update_stage(
            "required_workflow", "completed", f"{len(paths)} chapter drafts"
        )
        return paths

    def _ensure_relationship_arcs(self, project: Any) -> dict[str, Any]:
        if self.store.has_artifact("relationship_arcs"):
            plan = self.store.read_artifact("relationship_arcs")
            self.relationships.import_plan(plan)
            return plan
        required = (
            "genre_analysis",
            "world_bible",
            "character_bible",
            "story_outline",
        )
        if not all(self.store.has_artifact(name) for name in required):
            return self.plan(force=False)["relationship_arcs"]
        return self.relationship_architect.run(
            project,
            {
                "genre_analysis": self.store.read_artifact("genre_analysis"),
                "world_bible": self.store.read_artifact("world_bible"),
                "character_bible": self.store.read_artifact("character_bible"),
                "story_outline": self.store.read_artifact("story_outline"),
            },
            force=False,
        )

    @staticmethod
    def _validate_chapter_number(chapter_number: int, chapter_count: int) -> None:
        if chapter_number < 1 or chapter_number > chapter_count:
            raise ValueError(
                f"chapter_number must be between 1 and {chapter_count}"
            )
