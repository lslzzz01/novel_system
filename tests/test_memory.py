from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from novel_agents.artifacts import ArtifactStore
from novel_agents.chapter_versions import ChapterVersionStore
from novel_agents.embedding import HashEmbeddingProvider
from novel_agents.llm import MockLanguageModel
from novel_agents.memory_config import MemorySettingsStore
from novel_agents.memory_pipeline import MemoryPipeline
from novel_agents.models import ProjectBrief
from novel_agents.vector_index import NumpyVectorIndex


class StyleMockLanguageModel(MockLanguageModel):
    def generate_json(self, system_prompt, payload):
        result = super().generate_json(system_prompt, payload)
        if payload["task"] == "memory_extract":
            result["style_observations"] = [
                {"feature": "句式", "value": "短句与克制对白交替"}
            ]
        return result


class MemoryPipelineTests(unittest.TestCase):
    def make_project(self, root: Path, chapters: int = 2) -> None:
        ArtifactStore(root).initialize(
            ProjectBrief(
                title="记忆测试",
                premise="学生共同保住校园广播站。",
                genre="校园都市",
                target_readers="青年读者",
                style="自然克制",
                chapter_count=chapters,
                chapter_word_count=1000,
                constraints=["单女主"],
            )
        )
        MemorySettingsStore(root).initialize(
            {
                "embedding": {
                    "provider": "hash_test",
                    "dimensions": 64,
                    "quantization": "fp32",
                },
                "indexing": {"checkpoint_every_chapters": 1},
                "retrieval": {"minimum_similarity": -1, "vector_top_k": 10},
            }
        )

    def pipeline(self, root: Path) -> MemoryPipeline:
        return MemoryPipeline(
            root,
            MockLanguageModel(),
            root / ".models",
            embedder=HashEmbeddingProvider(64),
        )

    def test_only_finalized_text_is_indexed_and_retrieved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            self.make_project(root)
            versions = ChapterVersionStore(root)
            versions.write_draft(1, "# 第一章\n\n这是智能体初稿。")
            versions.write_author(1, "# 第一章\n\n林知夏在旧广播站保存了最后一盘磁带。")
            finalized = versions.finalize(1)
            pipeline = self.pipeline(root)

            result = pipeline.index_chapter(1)
            retrieval = pipeline.retrieve("林知夏保存的磁带", chapter_number=2)

            self.assertEqual("synced", result["status"])
            self.assertGreater(result["chunk_count"], 0)
            self.assertGreater(len(retrieval["retrieved_chunks"]), 0)
            self.assertEqual(finalized["final_hash"], pipeline.memory_store.indexed_hashes()[1])
            self.assertNotIn("智能体初稿", retrieval["retrieved_chunks"][0]["text"])

    def test_modified_final_is_marked_stale(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            self.make_project(root)
            versions = ChapterVersionStore(root)
            versions.write_draft(1, "# 第一章\n\n原始内容。")
            versions.finalize(1)
            pipeline = self.pipeline(root)
            pipeline.index_chapter(1)

            versions.final_path(1).write_text("# 第一章\n\n作者再次修改后的内容。\n", encoding="utf-8")
            overview = pipeline.overview(2)

            self.assertEqual("stale", overview["chapter_statuses"][0]["status"])
            self.assertEqual(1, overview["stale_chapters"])

    def test_strict_mode_blocks_next_chapter_until_sync(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            self.make_project(root)
            versions = ChapterVersionStore(root)
            versions.write_draft(1, "# 第一章\n\n尚未同步的定稿。")
            versions.finalize(1)
            pipeline = self.pipeline(root)

            with self.assertRaisesRegex(RuntimeError, "尚未完成记忆同步"):
                pipeline.assert_ready_for_chapter(2)

            pipeline.index_chapter(1)
            pipeline.assert_ready_for_chapter(2)

    def test_processing_status_is_visible_before_index_hash_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            self.make_project(root)
            versions = ChapterVersionStore(root)
            versions.finalize(1, "# 第一章\n\n等待嵌入的定稿。")
            versions.set_status(1, "embedding")

            self.assertEqual("embedding", versions.status(1)["status"])

    def test_rerank_limit_and_author_style_access_are_applied(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            self.make_project(root)
            MemorySettingsStore(root).update(
                {
                    "chunking": {
                        "target_chars": 100,
                        "max_chars": 140,
                        "overlap_chars": 0,
                    },
                    "retrieval": {
                        "rerank_top_k": 2,
                        "final_context_chunks": 8,
                    },
                    "policy": {
                        "style_learning": True,
                        "style_learning_min_chapters": 1,
                    },
                }
            )
            paragraphs = [
                f"第{number}段里，林知夏记录广播站的线索，并确认当晚的行动安排。" * 3
                for number in range(1, 7)
            ]
            ChapterVersionStore(root).finalize(1, "# 第一章\n\n" + "\n\n".join(paragraphs))
            pipeline = MemoryPipeline(
                root,
                StyleMockLanguageModel(),
                root / ".models",
                embedder=HashEmbeddingProvider(64),
            )
            pipeline.index_chapter(1)

            context = pipeline.retrieve(
                "广播站线索",
                chapter_number=2,
                access={"read_author_style": True},
            )
            no_raw = pipeline.retrieve(
                "只读取结构化记忆",
                chapter_number=2,
                access={"read_raw_passages": False},
            )

            self.assertLessEqual(len(context["retrieved_chunks"]), 2)
            self.assertTrue(context["author_style_observations"])
            self.assertEqual([], no_raw["retrieved_chunks"])
            self.assertEqual([], no_raw["recent_final_chapters"])

    def test_vector_instances_share_a_file_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first = NumpyVectorIndex(temp_dir, 64)
            second = NumpyVectorIndex(temp_dir, 64)
            self.assertIs(first._lock, second._lock)


if __name__ == "__main__":
    unittest.main()
