from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from novel_agents.artifacts import ArtifactStore
from novel_agents.llm import MockLanguageModel
from novel_agents.models import ProjectBrief
from novel_agents.relationship_store import RelationshipStateStore
from novel_agents.web import AppContext
from novel_agents.workflow import NovelWorkflow


class RelationshipWorkflowModel(MockLanguageModel):
    def __init__(self) -> None:
        super().__init__()
        self.draft_payload = None

    def generate_json(self, system_prompt, payload):
        if payload["task"] == "character_bible":
            self.calls.append("character_bible")
            return {
                "protagonists": [
                    {
                        "name": "顾言",
                        "goal": "完成当前目标",
                        "flaw": "不愿依赖别人",
                        "arc": "学会承担关系中的责任",
                    },
                    {
                        "name": "林夏",
                        "goal": "保护自己的选择",
                        "flaw": "习惯把真实想法藏起来",
                        "arc": "学会清楚表达边界",
                    },
                ],
                "supporting_cast": [],
                "relationships": [],
                "arcs": ["顾言与林夏从陌生合作发展为双向信任"],
            }
        return super().generate_json(system_prompt, payload)

    def generate_text(self, system_prompt, payload):
        if payload["task"] == "chapter_draft":
            self.draft_payload = payload
        return super().generate_text(system_prompt, payload)


class RelationshipSystemTests(unittest.TestCase):
    def test_stage_change_waits_for_confirmation_and_can_be_rolled_back(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RelationshipStateStore(temp_dir)
            pair = store.create_pair(
                {
                    "character_a": "顾言",
                    "character_b": "林夏",
                    "stage": "stranger",
                    "a_to_b": {"guard": 3},
                    "b_to_a": {"guard": 4},
                }
            )

            result = store.apply_extraction(
                1,
                "final-hash",
                {
                    "events": [
                        {
                            "pair_id": pair["id"],
                            "summary": "两人完成第一次有代价的合作",
                            "evidence_quotes": ["她把伞往他那边挪了一点。"],
                            "deltas": {
                                "a_to_b": {"trust": 1, "guard": -1},
                                "b_to_a": {"trust": 1},
                            },
                            "proposed_stage": "acquainted",
                            "confidence": 0.9,
                        }
                    ]
                },
            )

            self.assertEqual(1, result["pending"])
            public = store.public()
            current = public["pairs"][0]
            self.assertEqual("stranger", current["stage"])
            self.assertEqual(1, current["a_to_b"]["trust"])
            event_id = public["timeline"][0]["id"]

            store.decide_event(event_id, "accept")
            self.assertEqual("acquainted", store.public()["pairs"][0]["stage"])

            store.rollback_event(event_id)
            restored = store.public()["pairs"][0]
            self.assertEqual("stranger", restored["stage"])
            self.assertEqual(0, restored["a_to_b"]["trust"])

    def test_schedule_proposal_pending_until_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RelationshipStateStore(temp_dir)
            store.create_pair(
                {
                    "character_a": "沈知微",
                    "character_b": "林予",
                    "stage": "stranger",
                    "stage_schedule": [
                        {
                            "from_chapter": 1,
                            "stage": "stranger",
                            "summary": "开局远观",
                        },
                        {
                            "from_chapter": 3,
                            "stage": "acquainted",
                            "summary": "桥边第一次被对方记住",
                            "required_evidence": ["对方叫出她的名字"],
                        },
                        {
                            "from_chapter": 10,
                            "stage": "familiar",
                            "summary": "形成固定路径重合",
                        },
                    ],
                }
            )

            early = store.apply_schedule_proposals(2)
            self.assertEqual(0, early["proposed"])
            self.assertEqual("stranger", store.public()["pairs"][0]["stage"])

            due = store.apply_schedule_proposals(3)
            self.assertEqual(1, due["proposed"])
            public = store.public()
            self.assertEqual("stranger", public["pairs"][0]["stage"])
            self.assertEqual(1, public["pending_count"])
            event = next(
                item
                for item in public["timeline"]
                if item.get("kind") == "schedule_proposal"
            )
            self.assertEqual("acquainted", event["proposed_stage"])
            self.assertEqual("pending", event["status"])
            self.assertEqual(3, event["schedule_from_chapter"])

            # Idempotent: do not create a second pending for the same step.
            again = store.apply_schedule_proposals(3)
            self.assertEqual(0, again["proposed"])
            self.assertEqual(1, store.public()["pending_count"])

            store.decide_event(event["id"], "accept")
            self.assertEqual("acquainted", store.public()["pairs"][0]["stage"])

            # Next legal step only after current stage advances.
            later = store.apply_schedule_proposals(10)
            self.assertEqual(1, later["proposed"])
            pending = next(
                item
                for item in store.public()["timeline"]
                if item.get("status") == "pending"
            )
            self.assertEqual("familiar", pending["proposed_stage"])

    def test_import_plan_builds_schedule_from_milestones(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RelationshipStateStore(temp_dir)
            store.initialize()
            result = store.import_plan(
                {
                    "global_rules": ["阶段变化需有可见动作"],
                    "relationships": [
                        {
                            "character_a": "顾言",
                            "character_b": "林夏",
                            "initial_stage": "stranger",
                            "target_stage": "dating",
                            "milestones": [
                                {
                                    "chapter_range": [5, 8],
                                    "stage": "acquainted",
                                    "trigger": "第一次共同完成任务",
                                }
                            ],
                        }
                    ],
                }
            )
            self.assertEqual(1, result["added"])
            pair = store.public()["pairs"][0]
            self.assertEqual(1, len(pair["stage_schedule"]))
            self.assertEqual(5, pair["stage_schedule"][0]["from_chapter"])
            self.assertEqual("acquainted", pair["stage_schedule"][0]["stage"])

            proposed = store.apply_schedule_proposals(5)
            self.assertEqual(1, proposed["proposed"])

    def test_workflow_passes_relationship_context_and_beat_to_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "novel"
            artifacts = ArtifactStore(root)
            artifacts.initialize(
                ProjectBrief(
                    title="校园关系测试",
                    premise="两名学生在共同任务中逐渐建立信任。",
                    genre="校园青春",
                    writing_genre="campus",
                    target_readers="青年读者",
                    style="自然克制",
                    chapter_count=1,
                    chapter_word_count=1000,
                )
            )
            model = RelationshipWorkflowModel()

            NovelWorkflow(root, model).write_chapter(1)

            arcs = artifacts.read_artifact("relationship_arcs")
            beat = artifacts.read_relationship_beat(1)
            self.assertEqual(1, len(arcs["relationships"]))
            self.assertEqual("subtle", beat["intensity"])
            self.assertIsNotNone(model.draft_payload)
            inputs = model.draft_payload["inputs"]
            self.assertEqual(
                arcs["relationships"][0]["id"],
                inputs["relationship_context"]["pairs"][0]["id"],
            )
            self.assertEqual(beat, inputs["relationship_beat"])

    def test_finalize_queues_relationship_analysis_without_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = AppContext(temp_dir)
            try:
                project = app.create_project(
                    {
                        "title": "定稿关系测试",
                        "premise": "两个人在共同事件中改变彼此看法。",
                        "genre": "都市",
                        "writing_genre": "urban",
                        "target_readers": "青年读者",
                        "style": "自然克制",
                        "chapter_count": 1,
                        "chapter_word_count": 1000,
                        "memory_enabled": False,
                    }
                )
                project_id = project["id"]
                root = Path(temp_dir) / "projects" / project_id
                pair = app.create_relationship_pair(
                    project_id,
                    {"character_a": "顾言", "character_b": "林夏"},
                )["pair"]
                artifacts = ArtifactStore(root)
                artifacts.write_chapter_plan(
                    1,
                    {
                        "chapter_number": 1,
                        "title": "第一章",
                        "characters": ["顾言", "林夏"],
                    },
                )
                artifacts.write_relationship_beat(
                    1,
                    {
                        "chapter_number": 1,
                        "intensity": "subtle",
                        "focus_pairs": [{"pair_id": pair["id"]}],
                        "behavior_signals": [],
                        "callbacks": [],
                        "forbidden_leaps": [],
                        "progression_summary": "细微建立信任",
                        "stage_change_allowed": False,
                    },
                )
                artifacts.write_chapter(1, "# 第一章\n\n她把伞往他那边挪了一点。")

                result = app.finalize_chapter(
                    project_id, 1, auto_index=False, mock=True
                )

                self.assertIn("relationship_job", result)
                job_id = result["relationship_job"]["id"]
                deadline = time.time() + 5
                job = app.jobs.get(job_id)
                while job["status"] not in {"completed", "failed"} and time.time() < deadline:
                    time.sleep(0.02)
                    job = app.jobs.get(job_id)
                self.assertEqual("completed", job["status"])
            finally:
                app.close(wait=True)


if __name__ == "__main__":
    unittest.main()
