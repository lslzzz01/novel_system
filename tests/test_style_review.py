from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from novel_agents.artifacts import ArtifactStore
from novel_agents.chapter_versions import ChapterVersionStore
from novel_agents.llm import MockLanguageModel
from novel_agents.models import ProjectBrief
from novel_agents.style_agents import StyleAuditor, StyleEditor
from novel_agents.style_review import StyleReviewService


class StyleReviewWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.project_root = Path(self.temp_dir.name) / "project"
        self.store = ArtifactStore(self.project_root)
        self.store.initialize(
            ProjectBrief.from_dict(
                {
                    "title": "质检测试",
                    "premise": "学生修复校园广播站。",
                    "genre": "校园都市",
                    "target_readers": "青年读者",
                    "style": "自然克制",
                    "chapter_count": 1,
                    "chapter_word_count": 1000,
                    "constraints": ["单女主"],
                }
            )
        )
        self.store.write_chapter_plan(
            1,
            {
                "chapter_number": 1,
                "title": "旧广播站",
                "objective": "修好播音设备",
                "conflict": "零件缺失",
                "characters": ["林知夏"],
                "required_reveals": [],
                "withhold": [],
                "ending_hook": "设备里传出旧录音",
                "forbidden_major_additions": [],
            },
        )
        self.versions = ChapterVersionStore(self.project_root)
        self.draft = "# 旧广播站\n\n显然，这说明林知夏早有准备。\n"
        self.versions.write_draft(1, self.draft)
        model = MockLanguageModel()
        self.service = StyleReviewService(
            self.project_root,
            StyleAuditor(model),
            StyleEditor(model),
        )

    def test_audit_and_revision_do_not_overwrite_author_version(self) -> None:
        author = "# 旧广播站\n\n这是作者亲自润色的版本。\n"
        self.versions.write_author(1, author)

        audited = self.service.audit_chapter(1, "draft")
        self.assertTrue(audited["review"]["audit"]["issues"])
        self.assertEqual(author, self.versions.get_text(1, "author"))

        revised = self.service.revise_chapter(1)
        self.assertTrue(revised["review"]["candidate_text"])
        self.assertEqual(author, self.versions.get_text(1, "author"))

    def test_accept_archives_existing_author_version(self) -> None:
        old_author = "# 旧广播站\n\n作者原来的句子。\n"
        self.versions.write_author(1, old_author)
        self.service.audit_chapter(1, "draft")
        revised = self.service.revise_chapter(1)
        candidate = revised["review"]["candidate_text"]

        accepted = self.service.accept_candidate(1)

        self.assertEqual("accepted", accepted["style"]["review"]["status"])
        self.assertEqual(candidate.strip(), self.versions.get_text(1, "author").strip())
        archives = list(self.versions.versions_dir.glob("CH001_author_*.md"))
        self.assertEqual(1, len(archives))
        self.assertEqual(old_author, archives[0].read_text(encoding="utf-8"))
        self.assertFalse(self.versions.final_path(1).exists())

    def test_author_change_after_audit_blocks_acceptance(self) -> None:
        self.service.audit_chapter(1, "draft")
        self.service.revise_chapter(1)
        latest_author = "# 旧广播站\n\n作者在检测后新增的润色。\n"
        self.versions.write_author(1, latest_author)

        public = self.service.public(1)
        self.assertTrue(public["review"]["author_stale"])
        with self.assertRaisesRegex(ValueError, "作者稿"):
            self.service.accept_candidate(1)
        self.assertEqual(latest_author, self.versions.get_text(1, "author"))

    def test_source_change_after_audit_blocks_acceptance(self) -> None:
        self.service.audit_chapter(1, "draft")
        self.service.revise_chapter(1)
        self.versions.write_draft(1, "# 旧广播站\n\n原稿已被人工改动。\n")

        self.assertTrue(self.service.public(1)["review"]["source_stale"])
        with self.assertRaisesRegex(ValueError, "原稿"):
            self.service.accept_candidate(1)

    def test_reject_keeps_source_and_author_unchanged(self) -> None:
        author = "# 旧广播站\n\n保留这份作者稿。\n"
        self.versions.write_author(1, author)
        self.service.audit_chapter(1, "draft")
        self.service.revise_chapter(1)

        rejected = self.service.reject_candidate(1)

        self.assertEqual("rejected", rejected["review"]["status"])
        self.assertEqual(self.draft, self.versions.get_text(1, "draft"))
        self.assertEqual(author, self.versions.get_text(1, "author"))

    def test_explicit_empty_selection_never_means_select_all(self) -> None:
        self.service.audit_chapter(1, "draft")
        with self.assertRaisesRegex(ValueError, "没有符合条件"):
            self.service.revise_chapter(1, issue_ids=[])

    def test_high_continuity_risk_requires_explicit_selection(self) -> None:
        issue = {
            "id": "continuity-1",
            "severity": "high",
            "continuity_risk": "high",
        }
        self.assertEqual([], self.service._select_issues([issue], None, "low"))
        self.assertEqual([issue], self.service._select_issues([issue], ["continuity-1"], "low"))

    def test_audit_payload_includes_chapter_one_two_style_baseline(self) -> None:
        baseline = self.project_root / "style_review" / "baseline.json"
        baseline.parent.mkdir(parents=True, exist_ok=True)
        baseline.write_text(
            '{"source_chapters":[1,2],"style_principles":["克制","动作呈现"]}',
            encoding="utf-8",
        )

        class CapturingModel(MockLanguageModel):
            def __init__(self) -> None:
                super().__init__()
                self.last_payload = None

            def generate_json(self, system_prompt, payload):
                self.last_payload = payload
                return super().generate_json(system_prompt, payload)

        model = CapturingModel()
        service = StyleReviewService(
            self.project_root,
            StyleAuditor(model),
            StyleEditor(model),
        )
        service.audit_chapter(1, "draft")
        self.assertEqual([1, 2], model.last_payload["style_baseline"]["source_chapters"])
        self.assertIn("克制", model.last_payload["style_baseline"]["style_principles"])

    def test_audit_ignores_unlocatable_issue_instead_of_failing_report(self) -> None:
        paragraphs = [{"id": "p001", "text": "# 旧广播站"}]
        raw = {
            "status": "review",
            "score": 80,
            "summary": "发现问题",
            "strengths": [],
            "rewrite_recommended": True,
            "issues": [
                {
                    "id": "bad-location",
                    "category": "style",
                    "severity": "medium",
                    "paragraph_id": "p999",
                    "quote": "模型编造的原文",
                    "reason": "无法定位",
                    "suggestion": "忽略",
                    "continuity_risk": "low",
                }
            ],
        }

        normalized = StyleAuditor._normalize(
            raw,
            "# 旧广播站",
            paragraphs,
            [],
        )
        self.assertEqual([], normalized["issues"])


if __name__ == "__main__":
    unittest.main()
