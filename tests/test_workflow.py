from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from novel_agents.agents import ChapterWriter
from novel_agents.artifacts import ArtifactStore
from novel_agents.llm import MockLanguageModel
from novel_agents.models import ProjectBrief
from novel_agents.workflow import NovelWorkflow


def padded_draft(payload, opening: str = "", ending: str = "") -> str:
    plan = payload.get("chapter_plan") or payload["inputs"]["chapter_plan"]
    budget = plan.get("length_budget", {})
    minimum = int(
        budget.get("minimum_chars", plan.get("target_word_count", 300))
    )
    filler = (
        "主角把材料按顺序放好，核对眼前能够确认的事实，再决定下一步。"
        "阻力仍在，选择也留下了需要承担的代价。"
    )
    parts = [opening] if opening else []
    current = sum(
        1 for character in opening + ending if not character.isspace()
    )
    while current < minimum:
        parts.append(filler)
        current += sum(1 for character in filler if not character.isspace())
    if ending:
        parts.append(ending)
    return f"# {plan['title']}\n\n" + "\n\n".join(parts)


class MissingFieldOnceModel(MockLanguageModel):
    def __init__(self) -> None:
        super().__init__()
        self.failed_once = False

    def generate_json(self, system_prompt, payload):
        if payload["task"] == "genre_analysis" and not self.failed_once:
            self.failed_once = True
            self.calls.append("genre_analysis")
            return {"positioning": "字段不完整"}
        return super().generate_json(system_prompt, payload)


class TruncatedOutlineMapModel(MockLanguageModel):
    """Returns a valid but incomplete map from the first outline response."""

    def generate_json(self, system_prompt, payload):
        result = super().generate_json(system_prompt, payload)
        if payload["task"] == "story_outline":
            result["chapter_map"] = result["chapter_map"][:2]
        return result


class AiTasteOnceModel(MockLanguageModel):
    def __init__(self) -> None:
        super().__init__()
        self.failed_draft_once = False

    def generate_text(self, system_prompt, payload):
        if payload["task"] == "chapter_draft" and not self.failed_draft_once:
            self.failed_draft_once = True
            self.calls.append("chapter_draft")
            return (
                "# Chapter 1\n\n## 场景一\n\n"
                "这一刻他终于明白，命运的齿轮已经转动。"
            )
        return super().generate_text(system_prompt, payload)


class PersistentAiTasteModel(MockLanguageModel):
    def generate_text(self, system_prompt, payload):
        draft = super().generate_text(system_prompt, payload)
        return draft + "\n\n他皱了皱眉。她皱了皱眉。他又皱了皱眉。"


class AlwaysShortModel(MockLanguageModel):
    def generate_text(self, system_prompt, payload):
        self.calls.append(str(payload["task"]))
        return "# Chapter 1\n\n太短。"


class ShortDraftExtensionModel(MockLanguageModel):
    def __init__(self) -> None:
        super().__init__()
        self.extension_payload: dict = {}

    def generate_text(self, system_prompt, payload):
        task = str(payload["task"])
        self.calls.append(task)
        if task == "chapter_draft":
            return "# Chapter 1\n\n" + "字" * 900
        if task == "chapter_draft_humanize":
            return str(payload["draft"])
        if task == "chapter_draft_extend":
            self.extension_payload = dict(payload)
            return "补" * 450
        raise AssertionError(f"Unexpected task: {task}")


class MissingChapterStructureOnceModel(MockLanguageModel):
    def __init__(self) -> None:
        super().__init__()
        self.failed_once = False

    def generate_json(self, system_prompt, payload):
        result = super().generate_json(system_prompt, payload)
        if payload["task"] == "chapter_plan" and not self.failed_once:
            self.failed_once = True
            result.pop("mainline")
        return result


class FourSceneOnceModel(MockLanguageModel):
    def __init__(self) -> None:
        super().__init__()
        self.failed_once = False

    def generate_json(self, system_prompt, payload):
        result = super().generate_json(system_prompt, payload)
        if payload["task"] == "chapter_plan" and not self.failed_once:
            self.failed_once = True
            result["scene_cards"] = result["scene_cards"][:4]
            result["length_budget"]["scene_count"] = 4
        return result


class EmptyRhythmVariationModel(MockLanguageModel):
    """Simulates the production failure: secondary arrays returned empty."""

    def generate_json(self, system_prompt, payload):
        result = super().generate_json(system_prompt, payload)
        if payload["task"] == "chapter_plan":
            result["rhythm_variation"] = []
            result["incidental_details"] = []
            result["subtext"] = []
            result["lived_in_details"] = []
            result["anti_repetition"] = []
            result["forbidden_major_additions"] = []
            result["dialogue_information_limits"] = []
            result["ending_concrete_residue"] = ""
        return result


class RevisionResidueOnceModel(MockLanguageModel):
    def __init__(self) -> None:
        super().__init__()
        self.humanize_calls = 0
        self.last_repair = ""

    def generate_text(self, system_prompt, payload):
        task = str(payload["task"])
        self.calls.append(task)
        if task == "chapter_draft":
            return padded_draft(
                payload,
                opening="他把杯子放回桌上。",
                ending="“走。”",
            )
        self.humanize_calls += 1
        if self.humanize_calls == 1:
            return padded_draft(
                payload,
                opening="她说得很平淡。她显然也听见了门外的脚步。",
                ending=(
                    "两人消失在黑暗里，只留下房间里的滴水声仍然规律地落下。"
                ),
            )
        self.last_repair = str(payload.get("style_repair", ""))
        return padded_draft(
            payload,
            opening="她把杯子放回桌上。门外又响了一步。",
            ending="“走。”",
        )


class PromptCaptureModel(MockLanguageModel):
    def __init__(self) -> None:
        super().__init__()
        self.system_prompts: list[str] = []
        self.payloads: list[dict] = []

    def generate_text(self, system_prompt, payload):
        self.system_prompts.append(system_prompt)
        self.payloads.append(payload)
        return super().generate_text(system_prompt, payload)


class RequiredWorkflowTests(unittest.TestCase):
    def make_project(
        self,
        root: Path,
        chapters: int = 3,
        chapter_word_count: int = 1200,
    ) -> ArtifactStore:
        store = ArtifactStore(root)
        store.initialize(
            ProjectBrief(
                title="Test Novel",
                premise="A courier discovers that every delivered memory changes the past.",
                genre="speculative mystery",
                target_readers="adult mystery readers",
                style="restrained, tense, close third person",
                chapter_count=chapters,
                chapter_word_count=chapter_word_count,
                constraints=["No unexplained resurrection"],
            )
        )
        return store

    def test_full_required_workflow_creates_all_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "novel"
            store = self.make_project(root)
            model = MockLanguageModel()

            paths = NovelWorkflow(root, model).run()

            self.assertEqual(3, len(paths))
            for name in (
                "genre_analysis",
                "world_bible",
                "character_bible",
                "story_outline",
                "relationship_arcs",
                "subplot_register",
            ):
                self.assertTrue(store.has_artifact(name), name)
            for chapter_number in range(1, 4):
                self.assertTrue(store.has_chapter_plan(chapter_number))
                self.assertTrue(store.has_relationship_beat(chapter_number))
                self.assertTrue(store.has_chapter(chapter_number))
                self.assertTrue(store.read_chapter(chapter_number).startswith("# "))
            self.assertEqual(3, model.calls.count("chapter_plan"))
            self.assertEqual(3, model.calls.count("chapter_draft"))

    def test_existing_artifacts_are_reused_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "novel"
            self.make_project(root, chapters=2)
            model = MockLanguageModel()
            workflow = NovelWorkflow(root, model)

            workflow.plan()
            first_call_count = len(model.calls)
            workflow.plan()

            self.assertEqual(first_call_count, len(model.calls))

    def test_outline_fills_truncated_chapter_map_in_bounded_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "novel"
            store = self.make_project(root, chapters=45)
            model = TruncatedOutlineMapModel()

            outline = NovelWorkflow(root, model).plan()["story_outline"]

            self.assertEqual(45, len(outline["chapter_map"]))
            self.assertEqual(
                list(range(1, 46)),
                [entry["chapter_number"] for entry in outline["chapter_map"]],
            )
            self.assertEqual(3, model.calls.count("story_outline_chapter_map"))
            self.assertEqual(
                "completed", store.read_state()["stages"]["story_outline"]["status"]
            )

    def test_outline_normalizes_complete_legacy_chapter_map_without_regeneration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "novel"
            store = self.make_project(root, chapters=3)
            model = MockLanguageModel()
            workflow = NovelWorkflow(root, model)
            workflow.plan()
            outline = store.read_artifact("story_outline")
            outline["chapter_map"] = [
                {
                    "chapter": number,
                    "relationship_shift": f"Relationship change {number}",
                }
                for number in range(1, 4)
            ]
            store.write_artifact("story_outline", outline)
            calls_before = len(model.calls)

            normalized = workflow.plan()["story_outline"]

            self.assertEqual(calls_before, len(model.calls))
            self.assertEqual(
                [1, 2, 3],
                [entry["chapter_number"] for entry in normalized["chapter_map"]],
            )
            self.assertEqual(
                ["Relationship change 1", "Relationship change 2", "Relationship change 3"],
                [entry["purpose"] for entry in normalized["chapter_map"]],
            )

    def test_planning_schema_failure_is_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "novel"
            self.make_project(root, chapters=1)
            model = MissingFieldOnceModel()

            result = NovelWorkflow(root, model).plan()

            self.assertIn("core_promises", result["genre_analysis"])
            self.assertEqual(2, model.calls.count("genre_analysis"))

    def test_chapter_plan_missing_causal_structure_is_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "novel"
            store = self.make_project(root, chapters=1)
            model = MissingChapterStructureOnceModel()

            NovelWorkflow(root, model).plan_chapter(1)

            plan = store.read_chapter_plan(1)
            self.assertIn("mainline", plan)
            self.assertIn("subplot", plan)
            self.assertIn("length_budget", plan)
            self.assertEqual(2, model.calls.count("chapter_plan"))

    def test_four_thousand_character_plan_requires_five_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "novel"
            store = self.make_project(
                root,
                chapters=1,
                chapter_word_count=4000,
            )
            model = FourSceneOnceModel()

            NovelWorkflow(root, model).plan_chapter(1)

            plan = store.read_chapter_plan(1)
            self.assertEqual(5, len(plan["scene_cards"]))
            self.assertEqual(5, plan["length_budget"]["scene_count"])
            self.assertEqual(2, model.calls.count("chapter_plan"))

    def test_chapter_plan_empty_secondary_arrays_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "novel"
            store = self.make_project(root, chapters=1)
            model = EmptyRhythmVariationModel()

            NovelWorkflow(root, model).plan_chapter(1)

            plan = store.read_chapter_plan(1)
            for field in (
                "rhythm_variation",
                "incidental_details",
                "subtext",
                "lived_in_details",
                "anti_repetition",
                "forbidden_major_additions",
                "dialogue_information_limits",
            ):
                self.assertIsInstance(plan[field], list, field)
                self.assertTrue(plan[field], field)
            self.assertTrue(str(plan["ending_concrete_residue"]).strip())
            # Defaults should accept the plan without a schema retry.
            self.assertEqual(1, model.calls.count("chapter_plan"))

    def test_ai_taste_draft_is_regenerated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "novel"
            self.make_project(root, chapters=1)
            model = AiTasteOnceModel()

            path = NovelWorkflow(root, model).write_chapter(1)
            draft = path.read_text(encoding="utf-8")

            self.assertEqual(2, model.calls.count("chapter_draft"))
            self.assertEqual(1, model.calls.count("chapter_draft_humanize"))
            self.assertNotIn("## 场景", draft)
            self.assertNotIn("命运的齿轮", draft)

    def test_style_warning_does_not_fail_chapter_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "novel"
            store = self.make_project(root, chapters=1)

            path = NovelWorkflow(root, PersistentAiTasteModel()).write_chapter(1)

            self.assertTrue(path.is_file())
            stage = store.read_state()["stages"]["chapter_draft_001"]
            self.assertEqual("completed", stage["status"])
            self.assertIn("文风提醒", stage["detail"])

    def test_short_draft_is_extended_with_existing_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "novel"
            store = self.make_project(root, chapters=1)
            model = ShortDraftExtensionModel()

            path = NovelWorkflow(root, model).write_chapter(1)
            draft = path.read_text(encoding="utf-8")

            self.assertGreaterEqual(
                ChapterWriter._effective_content_chars(draft), 1200
            )
            self.assertIn("chapter_draft_extend", model.calls)
            self.assertEqual(300, model.extension_payload["shortfall_chars"])
            self.assertEqual(480, model.extension_payload["required_additional_chars"])
            self.assertEqual("completed", store.read_state()["stages"]["chapter_draft_001"]["status"])

    def test_persistently_short_draft_fails_without_writing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "novel"
            store = self.make_project(root, chapters=1)

            with self.assertRaisesRegex(ValueError, "正文有效字数不足"):
                NovelWorkflow(root, AlwaysShortModel()).write_chapter(1)

            self.assertFalse(store.has_chapter(1))
            stage = store.read_state()["stages"]["chapter_draft_001"]
            self.assertEqual("failed", stage["status"])
            self.assertIn("章节文件未写入", stage["detail"])

    def test_humanize_retries_on_interpretive_and_cinematic_residue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "novel"
            self.make_project(root, chapters=1)
            model = RevisionResidueOnceModel()

            path = NovelWorkflow(root, model).write_chapter(1)
            draft = path.read_text(encoding="utf-8")

            self.assertEqual(2, model.humanize_calls)
            self.assertIn("说得很平淡", model.last_repair)
            self.assertIn("镜头拉远式空景", model.last_repair)
            self.assertNotIn("显然", draft)
            self.assertNotIn("只留下", draft)

    def test_detector_flags_expository_dialogue_and_summary_reasoning(self) -> None:
        draft = (
            "# 第一章\n\n"
            "她眼神里有惊讶，也有警戒，苦笑了一下。水滴声规律得像呼吸。"
            "她看着他，像是在判断他的意图。某种阴影贴在墙上。\n\n"
            "“封印如果由魔道长老施放，就不可能用正道功法解除，那是专门锁住金丹的术式，"
            "需要元婴修为才能破开，我在宗门古籍里见过记载。”\n\n"
            "他在脑海里快速梳理这些信息。圣女、封印、魔道、灵脉——这几个词连在了一起。"
        )

        issues = "；".join(ChapterWriter._quality_issues(draft))

        self.assertIn("百科式长说明", issues)
        self.assertIn("摘要式推理", issues)
        self.assertIn("复杂神色或声音情绪标签", issues)
        self.assertIn("模糊套话过多", issues)
        self.assertIn("作者判断式比喻", issues)

    def test_detector_flags_draft_below_minimum_length(self) -> None:
        plan = {"length_budget": {"minimum_chars": 100}}

        issues = "；".join(ChapterWriter._quality_issues("# 第一章\n\n" + "字" * 99, plan))

        self.assertIn("当前 99，最低 100", issues)

    def test_detector_accepts_draft_at_minimum_length(self) -> None:
        plan = {"length_budget": {"minimum_chars": 100}}

        issues = "；".join(ChapterWriter._quality_issues("# 第一章\n\n" + "字" * 100, plan))

        self.assertNotIn("有效字数不足", issues)

    def test_detector_checks_required_identity_withhold_and_new_names(self) -> None:
        draft = (
            "# 第一章\n\n"
            "“我叫洛清寒。”\n\n"
            "她说自己是天灵体，又让他去找一个叫无尘老人的联系人。"
        )
        plan = {
            "characters": ["洛清寒"],
            "required_reveals": ["洛清寒承认自己是太玄道宗圣女。"],
            "withhold": ["不得说出‘天灵体’一词。"],
        }

        issues = "；".join(ChapterWriter._quality_issues(draft, plan))

        self.assertIn("无尘老人", issues)
        self.assertIn("圣女", issues)
        self.assertIn("天灵体", issues)

    def test_name_detector_ignores_question_and_call_me_phrases(self) -> None:
        draft = (
            "# 第一章\n\n"
            "“你随便叫什么。”\n\n"
            "“上个月他们还叫我圣女。”\n\n"
            "“我叫洛清寒。”"
        )
        plan = {
            "characters": ["洛清寒"],
            "required_reveals": ["洛清寒承认自己是太玄道宗圣女。"],
            "withhold": [],
        }

        issues = "；".join(ChapterWriter._quality_issues(draft, plan))

        self.assertNotIn("命名角色", issues)

    def test_write_rejects_out_of_range_chapter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "novel"
            self.make_project(root, chapters=1)
            workflow = NovelWorkflow(root, MockLanguageModel())

            with self.assertRaisesRegex(ValueError, "between 1 and 1"):
                workflow.write_chapter(2)

    def test_writer_payload_uses_chapter_local_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "novel"
            store = self.make_project(root, chapters=1)
            model = PromptCaptureModel()
            workflow = NovelWorkflow(root, model)
            workflow.plan_chapter(1)

            # Simulate a long production outline. The writer must not receive
            # unrelated chapter entries even when the stored artifact is huge.
            outline = store.read_artifact("story_outline")
            outline["chapter_map"] = [
                {"chapter": number, "purpose": "unrelated planning text" * 30}
                for number in range(1, 326)
            ]
            store.write_artifact("story_outline", outline)
            workflow.write_chapter(1)

            payload = model.payloads[0]["inputs"]
            self.assertLessEqual(
                len(json.dumps(payload, ensure_ascii=False)),
                NovelWorkflow.WRITER_CONTEXT_MAX_CHARS,
            )
            self.assertLessEqual(
                len(payload["story_outline"]["current_and_adjacent_chapters"]), 2
            )
            self.assertEqual(
                workflow._chapter_character_names(store.read_chapter_plan(1)),
                [item["name"] for item in payload["character_bible"]["chapter_cast"]],
            )

    def test_selected_genre_writer_receives_shared_and_genre_prompts(self) -> None:
        cases = (
            ("玄幻修仙", "xuanhuan", "chapter_writer_xuanhuan", "玄幻专用标记"),
            ("校园青春", "campus", "chapter_writer_campus", "校园专用标记"),
            ("现代都市", "urban", "chapter_writer_urban", "都市专用标记"),
        )
        for genre, writing_genre, agent_id, marker in cases:
            with self.subTest(writing_genre=writing_genre), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir) / "novel"
                store = ArtifactStore(root)
                store.initialize(
                    ProjectBrief(
                        title="题材路由测试",
                        premise="主角处理一个会改变关系的现实问题。",
                        genre=genre,
                        writing_genre=writing_genre,
                        target_readers="青年读者",
                        style="自然克制",
                        chapter_count=1,
                        chapter_word_count=1000,
                    )
                )
                base_model = MockLanguageModel()
                writer_model = PromptCaptureModel()
                workflow = NovelWorkflow(
                    root,
                    base_model,
                    prompt_overrides={
                        "chapter_draft": "共享人味规则标记。角色通过动作和不完整对话呈现自己。",
                        agent_id: f"{marker}。只遵守当前题材的现实约束。",
                    },
                    agent_models={agent_id: writer_model},
                )

                workflow.write_chapter(1)

                self.assertIn("chapter_draft", writer_model.calls)
                self.assertNotIn("chapter_draft", base_model.calls)
                self.assertIn("共享人味规则标记", writer_model.system_prompts[0])
                self.assertIn(marker, writer_model.system_prompts[0])
                self.assertEqual(agent_id, writer_model.payloads[0]["writer_agent_id"])
                self.assertEqual(writing_genre, writer_model.payloads[0]["writer_genre"])

    def test_overwrite_clears_generated_material(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "novel"
            store = self.make_project(root, chapters=1)
            NovelWorkflow(root, MockLanguageModel()).run()
            self.assertTrue(store.has_chapter(1))

            store.initialize(store.load_brief(), overwrite=True)

            self.assertFalse(store.has_artifact("story_outline"))
            self.assertFalse(store.has_chapter(1))


if __name__ == "__main__":
    unittest.main()
