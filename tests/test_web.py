from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

from novel_agents.prompt_store import PromptStore
from novel_agents.agent_profiles import AgentProfileStore
from novel_agents.artifacts import ArtifactStore
from novel_agents.chapter_versions import ChapterVersionStore
from novel_agents.memory_store import MemoryStore
from novel_agents.settings import ModelSettingsStore
from novel_agents.vector_index import NumpyVectorIndex
from novel_agents.web import AppContext, make_handler

import numpy as np


class LocalConfigurationTests(unittest.TestCase):
    def test_generation_char_count_ignores_whitespace(self) -> None:
        from novel_agents.web import count_generated_chars, format_char_count

        self.assertEqual(4, count_generated_chars("一 二\n三\t四"))
        self.assertEqual("4 字", format_char_count(4))

    def test_model_settings_never_return_the_secret(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ModelSettingsStore(temp_dir)
            public = store.update(
                {
                    "base_url": "https://example.test/v1",
                    "model": "test-model",
                    "temperature": 0.5,
                    "timeout_seconds": 90,
                    "api_key": "secret-value-1234",
                }
            )

            self.assertTrue(public["has_api_key"])
            self.assertEqual("****1234", public["api_key_hint"])
            self.assertNotIn("secret-value-1234", json.dumps(public))

    def test_legacy_model_settings_migrate_to_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            local_dir = Path(temp_dir) / ".novel_agents"
            local_dir.mkdir()
            path = local_dir / "model.json"
            path.write_text(
                json.dumps(
                    {
                        "base_url": "https://legacy.example/v1",
                        "model": "legacy-model",
                        "api_key": "legacy-secret-5678",
                        "temperature": 0.6,
                    }
                ),
                encoding="utf-8",
            )

            store = ModelSettingsStore(temp_dir)
            registry = store.registry_public()
            migrated = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(2, migrated["version"])
            self.assertEqual("default", registry["active_model_id"])
            self.assertEqual("legacy-model", registry["models"][0]["model"])
            self.assertEqual("****5678", registry["models"][0]["api_key_hint"])
            self.assertNotIn("legacy-secret-5678", json.dumps(registry))

    def test_model_registry_selects_the_complete_connection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ModelSettingsStore(temp_dir)
            first = store.create_model(
                {
                    "display_name": "规划模型",
                    "base_url": "https://planner.example/v1",
                    "model": "planner-pro",
                    "api_key": "planner-key",
                    "temperature": 0.2,
                }
            )
            second = store.create_model(
                {
                    "display_name": "正文模型",
                    "base_url": "https://writer.example/v1",
                    "model": "writer-flash",
                    "api_key": "writer-key",
                    "temperature": 0.8,
                }
            )

            self.assertTrue(first["is_default"])
            writer = store.create_client(model_profile_id=second["id"])
            self.assertEqual("https://writer.example/v1", writer.base_url)
            self.assertEqual("writer-flash", writer.model)
            self.assertEqual("writer-key", writer.api_key)

            selected = store.set_active(second["id"])
            self.assertTrue(selected["is_default"])
            self.assertEqual("writer-flash", store.create_client().model)
            with self.assertRaisesRegex(ValueError, "默认模型不能删除"):
                store.delete_model(second["id"])

    def test_builtin_prompt_can_be_overridden_and_reset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = PromptStore(temp_dir)
            original = store.get_agent("story_outline")["prompt"]

            changed = store.save_prompt("story_outline", "这是一个长度足够的自定义大纲提示词，用于自动化测试。")
            self.assertTrue(changed["overridden"])
            self.assertIn("自定义大纲", store.prompt_overrides()["story_outline"])

            restored = store.reset_prompt("story_outline")
            self.assertFalse(restored["overridden"])
            self.assertEqual(original, restored["prompt"])

    def test_custom_agent_registration_reserves_a_hook(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = PromptStore(temp_dir)
            agent = store.create_custom(
                {
                    "id": "continuity_reviewer",
                    "display_name": "一致性审查员",
                    "hook": "after_chapter_draft",
                    "output_mode": "json",
                    "prompt": "检查人物状态、时间线和世界规则是否与当前章节保持一致。",
                }
            )

            self.assertEqual("reserved", agent["runtime_status"])
            self.assertFalse(agent["required"])


class HttpApplicationTests(unittest.TestCase):
    def test_model_library_http_routes_and_agent_delete_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = AppContext(temp_dir)
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                planner = self.request(
                    base + "/api/models",
                    method="POST",
                    body={
                        "display_name": "规划模型",
                        "base_url": "https://planner.example/v1",
                        "model": "planner-pro",
                        "api_key": "planner-secret",
                        "temperature": 0.2,
                        "timeout_seconds": 120,
                    },
                )["model"]
                writer = self.request(
                    base + "/api/models",
                    method="POST",
                    body={
                        "display_name": "正文模型",
                        "base_url": "https://writer.example/v1",
                        "model": "writer-flash",
                        "api_key": "writer-secret",
                        "temperature": 0.8,
                        "timeout_seconds": 180,
                    },
                )["model"]
                registry = self.request(base + "/api/models")
                self.assertEqual(2, len(registry["models"]))
                self.assertNotIn("planner-secret", json.dumps(registry))
                self.assertNotIn("writer-secret", json.dumps(registry))

                profile = self.request(
                    base + "/api/agents/chapter_draft/profile",
                    method="PUT",
                    body={"model_profile_id": writer["id"]},
                )["profile"]
                self.assertEqual(writer["id"], profile["model_profile_id"])
                self.assertEqual("writer-flash", profile["resolved_model"]["model"])

                with self.assertRaises(urllib.error.HTTPError) as used_error:
                    self.request(
                        base + f"/api/models/{quote(writer['id'])}", method="DELETE"
                    )
                self.assertEqual(409, used_error.exception.code)

                self.request(
                    base + "/api/agents/chapter_draft/profile",
                    method="PUT",
                    body={"model_profile_id": None, "model": None},
                )
                selected = self.request(
                    base + "/api/models/default",
                    method="PUT",
                    body={"model_id": writer["id"]},
                )["registry"]
                self.assertEqual(writer["id"], selected["active_model_id"])
                deleted = self.request(
                    base + f"/api/models/{quote(planner['id'])}", method="DELETE"
                )
                self.assertTrue(deleted["deleted"])
            finally:
                server.shutdown()
                server.server_close()
                app.close(wait=True)

    def test_relationship_workbench_http_routes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = AppContext(temp_dir)
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                project = app.create_project(
                    {
                        "title": "关系工作台接口测试",
                        "premise": "两名同学共同完成一次活动。",
                        "genre": "校园",
                        "writing_genre": "campus",
                        "target_readers": "青年读者",
                        "style": "自然克制",
                        "chapter_count": 1,
                        "chapter_word_count": 1000,
                        "memory_enabled": False,
                    }
                )
                root_url = base + f"/api/projects/{quote(project['id'])}/relationships"
                pair = self.request(
                    root_url + "/pairs",
                    method="POST",
                    body={
                        "character_a": "顾言",
                        "character_b": "林夏",
                        "stage": "stranger",
                    },
                )["pair"]
                self.request(
                    root_url + "/settings",
                    method="PUT",
                    body={"auto_update_after_finalize": False},
                )
                changed = self.request(
                    root_url + f"/pairs/{quote(pair['id'])}",
                    method="PUT",
                    body={
                        "stage": "acquainted",
                        "a_to_b": {"trust": 2, "guard": 2},
                        "b_to_a": {"trust": 1, "guard": 3},
                    },
                )["pair"]
                self.assertEqual("acquainted", changed["stage"])
                self.assertEqual(2, changed["a_to_b"]["trust"])

                project_root = Path(temp_dir) / "projects" / project["id"]
                artifacts = ArtifactStore(project_root)
                artifacts.write_chapter_plan(
                    1,
                    {
                        "chapter_number": 1,
                        "title": "第一章",
                        "characters": ["顾言", "林夏"],
                    },
                )
                artifacts.write_chapter(1, "# 第一章\n\n她把活动表递给他。")
                self.request(
                    base
                    + f"/api/projects/{quote(project['id'])}/chapters/1/finalize",
                    method="POST",
                    body={"auto_index": False, "mock": True},
                )
                job = self.request(
                    root_url + "/actions",
                    method="POST",
                    body={"action": "analyze_chapter", "chapter": 1, "mock": True},
                )
                for _ in range(100):
                    status = self.request(base + f"/api/jobs/{job['id']}")
                    if status["status"] in {"completed", "failed"}:
                        break
                    time.sleep(0.02)
                self.assertEqual("completed", status["status"])

                public = self.request(root_url)
                self.assertEqual(1, len(public["pairs"]))
                deleted = self.request(
                    root_url + f"/pairs/{quote(pair['id'])}", method="DELETE"
                )
                self.assertTrue(deleted["deleted"])
            finally:
                server.shutdown()
                server.server_close()
                app.close(wait=True)

    def test_completed_jobs_are_available_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = AppContext(temp_dir)
            project = app.create_project(
                {
                    "title": "任务历史测试",
                    "premise": "学生修复旧广播设备。",
                    "genre": "校园都市",
                    "target_readers": "青年读者",
                    "style": "自然克制",
                    "chapter_count": 1,
                    "chapter_word_count": 1000,
                    "constraints": ["单女主"],
                }
            )
            job = app.start_action(
                project["id"], {"action": "plan", "mock": True, "force": False}
            )
            for _ in range(100):
                status = app.jobs.get(job["id"])
                if status["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.02)
            self.assertEqual("completed", status["status"])
            app.close(wait=True)

            restarted = AppContext(temp_dir)
            try:
                persisted = restarted.jobs.get(job["id"])
                self.assertEqual("completed", persisted["status"])
                self.assertEqual(job["id"], restarted.recent_jobs(project["id"], 1)[0]["id"])
            finally:
                restarted.close(wait=True)

    def test_artifacts_can_be_edited_and_deleted_safely(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = AppContext(temp_dir)
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                project = app.create_project(
                    {
                        "title": "产物编辑测试",
                        "premise": "学生整理广播站档案。",
                        "genre": "校园都市",
                        "target_readers": "青年读者",
                        "style": "自然克制",
                        "chapter_count": 1,
                        "chapter_word_count": 1000,
                        "constraints": ["单女主"],
                    }
                )
                project_root = Path(temp_dir) / "projects" / project["id"]
                store = ArtifactStore(project_root)
                store.write_artifact("story_outline", {"logline": "旧内容"})
                store.write_chapter(1, "# 第一章\n\n旧正文")
                project_url = base + f"/api/projects/{quote(project['id'])}"

                changed_json = self.request(
                    project_url + "/artifact?path=artifacts%2Fstory_outline.json",
                    method="PUT",
                    body={"content": '{"logline":"新内容","chapter_map":[]}'},
                )
                self.assertTrue(changed_json["planning_stale"])
                self.assertEqual("新内容", store.read_artifact("story_outline")["logline"])

                with self.assertRaises(urllib.error.HTTPError) as invalid_json:
                    self.request(
                        project_url + "/artifact?path=artifacts%2Fstory_outline.json",
                        method="PUT",
                        body={"content": "{not valid json}"},
                    )
                self.assertEqual(400, invalid_json.exception.code)

                changed_markdown = self.request(
                    project_url + "/artifact?path=chapters%2FCH001_draft.md",
                    method="PUT",
                    body={"content": "# 第一章\n\n作者修改后的正文。"},
                )
                self.assertFalse(changed_markdown["planning_stale"])
                self.assertIn("作者修改", store.read_chapter(1))

                detail = app.project_detail(project["id"])
                draft_item = next(
                    item
                    for item in detail["artifacts"]
                    if item["path"] == "chapters/CH001_draft.md"
                )
                self.assertGreater(draft_item["char_count"], 0)
                self.assertIn("字", draft_item["char_count_label"])
                opened = self.request(
                    project_url + "/artifact?path=chapters%2FCH001_draft.md"
                )
                self.assertEqual(draft_item["char_count"], opened["char_count"])
                self.assertIn("字", opened["char_count_label"])

                (project_root / "artifacts" / "unsafe.bin").write_bytes(b"unsafe")
                for path in ("..%2Fproject.json", "artifacts%2Funsafe.bin"):
                    with self.assertRaises(urllib.error.HTTPError) as unsafe_error:
                        self.request(
                            project_url + f"/artifact?path={path}",
                            method="PUT",
                            body={"content": "unsafe"},
                        )
                    self.assertEqual(400, unsafe_error.exception.code)

                versions = ChapterVersionStore(project_root)
                versions.finalize(1, "# 第一章\n\n已经定稿并建立记忆。")
                memory = MemoryStore(project_root)
                memory.replace_chapter(
                    1,
                    "hash",
                    "摘要",
                    10,
                    [
                        {
                            "id": "ch001-memory",
                            "text_hash": "hash",
                            "text": "记忆内容",
                        }
                    ],
                    [],
                )
                index = NumpyVectorIndex(project_root / "memory", 512)
                index.upsert(["ch001-memory"], np.ones((1, 512), dtype=np.float32))
                removed_final = self.request(
                    project_url + "/artifact?path=chapters%2FCH001_final.md",
                    method="DELETE",
                )
                self.assertEqual(1, removed_final["memory_chunks_removed"])
                self.assertIsNone(memory.chapter_record(1))
                self.assertEqual(0, index.count())

                removed = self.request(
                    project_url + "/artifact?path=chapters%2FCH001_draft.md",
                    method="DELETE",
                )
                self.assertTrue(removed["deleted"])
                self.assertFalse(store.chapter_path(1).exists())
            finally:
                server.shutdown()
                server.server_close()
                app.close(wait=True)

    def test_health_agents_and_project_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = AppContext(temp_dir)
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                health = self.request(base + "/api/health")
                self.assertTrue(health["ok"])
                agents = self.request(base + "/api/agents")
                self.assertEqual(19, len(agents["agents"]))
                agent_ids = {item["id"] for item in agents["agents"]}
                self.assertTrue(
                    {
                        "relationship_arcs",
                        "relationship_beat",
                        "relationship_audit",
                        "relationship_memory",
                    }.issubset(agent_ids)
                )
                campus_writer = self.request(
                    base + "/api/agents/chapter_writer_campus"
                )["agent"]
                self.assertEqual("genre", campus_writer["activation"])
                self.assertEqual("chapter_draft", campus_writer["profile_parent"])
                self.assertIn("正文硬规则", campus_writer["combined_prompt"])
                self.assertIn("课程、作息、考试", campus_writer["combined_prompt"])

                project = self.request(
                    base + "/api/projects",
                    method="POST",
                    body={
                        "title": "测试校园故事",
                        "premise": "两个学生共同保住校园广播站。",
                        "genre": "校园都市",
                        "writing_genre": "campus",
                        "target_readers": "青年读者",
                        "style": "自然克制",
                        "chapter_count": 2,
                        "chapter_word_count": 1000,
                        "constraints": ["单女主"],
                    },
                )
                self.assertEqual("测试校园故事", project["brief"]["title"])
                self.assertEqual("campus", project["brief"]["writing_genre"])
                self.assertTrue(project["memory"]["enabled"])
                listing = self.request(base + "/api/projects")
                self.assertEqual(1, len(listing["projects"]))

                job = self.request(
                    base + f"/api/projects/{quote(project['id'])}/actions",
                    method="POST",
                    body={"action": "plan", "mock": True, "force": False},
                )
                for _ in range(100):
                    status = self.request(base + f"/api/jobs/{job['id']}")
                    if status["status"] in {"completed", "failed"}:
                        break
                    time.sleep(0.02)
                self.assertEqual("completed", status["status"])

                finalized = self.request(
                    base + f"/api/projects/{quote(project['id'])}/chapters/1/finalize",
                    method="POST",
                    body={
                        "text": "# 第一章\n\n林知夏在广播站保存了最后一盘磁带。",
                        "auto_index": True,
                        "mock": True,
                    },
                )
                memory_job = finalized["job"]
                for _ in range(100):
                    memory_status = self.request(base + f"/api/jobs/{memory_job['id']}")
                    if memory_status["status"] in {"completed", "failed"}:
                        break
                    time.sleep(0.02)
                self.assertEqual("completed", memory_status["status"])
                memory = self.request(
                    base + f"/api/projects/{quote(project['id'])}/memory"
                )
                self.assertEqual(1, memory["indexed_chapters"])
                self.assertGreater(memory["vector_count"], 0)

                profile = self.request(
                    base + "/api/agents/chapter_draft/profile",
                    method="PUT",
                    body={"temperature": 0.8, "max_tokens": 6000},
                )
                self.assertEqual(0.8, profile["profile"]["temperature"])
            finally:
                server.shutdown()
                server.server_close()
                app.close(wait=True)

    def test_style_review_http_workflow_requires_human_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = AppContext(temp_dir)
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                project = app.create_project(
                    {
                        "title": "正文质检接口测试",
                        "premise": "学生修复校园广播站。",
                        "genre": "校园都市",
                        "target_readers": "青年读者",
                        "style": "自然克制",
                        "chapter_count": 1,
                        "chapter_word_count": 1000,
                        "constraints": ["单女主"],
                    }
                )
                project_root = Path(temp_dir) / "projects" / project["id"]
                store = ArtifactStore(project_root)
                store.write_chapter_plan(
                    1,
                    {
                        "chapter_number": 1,
                        "title": "旧广播站",
                        "objective": "修好设备",
                        "conflict": "零件缺失",
                        "characters": ["林知夏"],
                        "required_reveals": [],
                        "withhold": [],
                        "ending_hook": "传出旧录音",
                        "forbidden_major_additions": [],
                    },
                )
                draft = "# 旧广播站\n\n显然，这说明林知夏早有准备。\n"
                store.write_chapter(1, draft)
                style_url = (
                    base
                    + f"/api/projects/{quote(project['id'])}/chapters/1/style"
                )

                initial = self.request(style_url)
                self.assertTrue(initial["available_sources"]["draft"])
                self.assertEqual("not_started", initial["review"]["status"])

                settings = self.request(
                    style_url + "/settings",
                    method="PUT",
                    body={
                        "mode": "detect_only",
                        "auto_revision_rounds": 1,
                        "minimum_severity": "medium",
                        "generate_revision_after_audit": False,
                        "second_audit": True,
                        "protect_author_version": True,
                    },
                )
                self.assertEqual("detect_only", settings["settings"]["mode"])

                def wait_for(job_id: str) -> dict:
                    for _ in range(200):
                        status = self.request(base + f"/api/jobs/{job_id}")
                        if status["status"] in {"completed", "failed"}:
                            return status
                        time.sleep(0.01)
                    self.fail("正文质检任务等待超时")

                audit_job = self.request(
                    style_url + "/actions",
                    method="POST",
                    body={"action": "audit", "source": "draft", "mock": True},
                )
                self.assertEqual("completed", wait_for(audit_job["id"])["status"])
                audited = self.request(style_url)
                issues = audited["review"]["audit"]["issues"]
                self.assertTrue(issues)
                self.assertEqual(draft, store.read_chapter(1))
                self.assertFalse(ChapterVersionStore(project_root).author_path(1).exists())

                revise_job = self.request(
                    style_url + "/actions",
                    method="POST",
                    body={
                        "action": "revise",
                        "issue_ids": [issues[0]["id"]],
                        "second_audit": True,
                        "mock": True,
                    },
                )
                self.assertEqual("completed", wait_for(revise_job["id"])["status"])
                revised = self.request(style_url)
                self.assertTrue(revised["review"]["candidate_text"])
                self.assertFalse(ChapterVersionStore(project_root).author_path(1).exists())

                edited_text = revised["review"]["candidate_text"].rstrip() + "\n\n林知夏拧紧最后一颗螺丝。\n"
                edited = self.request(
                    style_url + "/candidate",
                    method="PUT",
                    body={"text": edited_text},
                )
                self.assertEqual("candidate_edited", edited["review"]["status"])
                self.assertIsNone(edited["review"]["revision_audit"])

                recheck_job = self.request(
                    style_url + "/actions",
                    method="POST",
                    body={"action": "recheck", "mock": True},
                )
                self.assertEqual("completed", wait_for(recheck_job["id"])["status"])
                rechecked = self.request(style_url)
                self.assertIsNotNone(rechecked["review"]["revision_audit"])

                accepted = self.request(
                    style_url + "/accept",
                    method="POST",
                    body={"text": edited_text},
                )
                self.assertEqual("accepted", accepted["style"]["review"]["status"])
                versions = ChapterVersionStore(project_root)
                self.assertEqual(edited_text.strip(), versions.get_text(1, "author").strip())
                self.assertFalse(versions.final_path(1).exists())
                self.assertIsNone(MemoryStore(project_root).chapter_record(1))
            finally:
                server.shutdown()
                server.server_close()
                app.close(wait=True)

    def test_memory_settings_author_pause_and_profile_persist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = AppContext(temp_dir)
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                project = self.request(
                    base + "/api/projects",
                    method="POST",
                    body={
                        "title": "记忆配置测试",
                        "premise": "学生整理广播站历史。",
                        "genre": "校园都市",
                        "target_readers": "青年读者",
                        "style": "自然克制",
                        "chapter_count": 2,
                        "chapter_word_count": 1000,
                        "constraints": ["单女主"],
                    },
                )
                project_url = base + f"/api/projects/{quote(project['id'])}"
                updated = self.request(
                    project_url + "/memory/settings",
                    method="PUT",
                    body={
                        "chunking": {"target_chars": 450},
                        "indexing": {
                            "batch_size": 8,
                            "max_parallel_jobs": 1,
                            "pause_during_writing": True,
                            "checkpoint_every_chapters": 10,
                        },
                    },
                )
                self.assertTrue(updated["requires_rebuild"])

                author = self.request(
                    project_url + "/chapters/1/author",
                    method="PUT",
                    body={"text": "# 第一章\n\n这是作者润色后的版本。"},
                )
                self.assertEqual("author_edit", author["chapter"]["status"])
                detail = self.request(project_url + "/chapters/1")
                self.assertIn("作者润色", detail["author_text"])

                paused = self.request(
                    project_url + "/memory/actions",
                    method="POST",
                    body={"action": "pause"},
                )
                self.assertTrue(paused["runtime"]["manually_paused"])
                resumed = self.request(
                    project_url + "/memory/actions",
                    method="POST",
                    body={"action": "resume"},
                )
                self.assertFalse(resumed["runtime"]["manually_paused"])

                self.request(
                    base + "/api/agents/chapter_draft/profile",
                    method="PUT",
                    body={
                        "temperature": 0.7,
                        "memory_access": {
                            "enabled": True,
                            "read_author_style": True,
                            "top_k": 6,
                        },
                    },
                )
                persisted = AgentProfileStore(temp_dir).get("chapter_draft")
                self.assertEqual(0.7, persisted["temperature"])
                self.assertTrue(persisted["memory_access"]["read_author_style"])
                self.assertEqual(6, persisted["memory_access"]["top_k"])
            finally:
                server.shutdown()
                server.server_close()
                app.close(wait=True)

    def test_finalize_succeeds_when_memory_queue_is_busy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = AppContext(temp_dir)
            release = threading.Event()
            started = threading.Event()
            try:
                project = app.create_project(
                    {
                        "title": "队列测试",
                        "premise": "测试定稿和索引队列。",
                        "genre": "校园都市",
                        "target_readers": "青年读者",
                        "style": "自然克制",
                        "chapter_count": 1,
                        "chapter_word_count": 1000,
                        "constraints": ["单女主"],
                    }
                )

                def blocking_job(record):
                    started.set()
                    release.wait(timeout=5)
                    return "done"

                app.memory_jobs.submit(project["id"], "blocking", blocking_job)
                self.assertTrue(started.wait(timeout=2))
                result = app.finalize_chapter(
                    project["id"],
                    1,
                    "# 第一章\n\n队列繁忙时仍应保存定稿。",
                    auto_index=True,
                    mock=True,
                )

                self.assertIn("indexing_error", result)
                self.assertTrue(Path(result["path"]).is_file())
            finally:
                release.set()
                app.close(wait=True)

    def test_project_and_custom_agent_can_be_updated_and_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = AppContext(temp_dir)
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                project = self.request(
                    base + "/api/projects",
                    method="POST",
                    body={
                        "title": "待修改项目",
                        "premise": "学生调查校园旧档案。",
                        "genre": "校园都市",
                        "target_readers": "青年读者",
                        "style": "自然克制",
                        "chapter_count": 3,
                        "chapter_word_count": 1000,
                        "constraints": ["单女主"],
                    },
                )
                project_url = base + f"/api/projects/{quote(project['id'])}"
                updated = self.request(
                    project_url,
                    method="PUT",
                    body={
                        "title": "修改后的项目",
                        "premise": "学生修复广播站并追查旧录音。",
                        "genre": "校园悬疑",
                        "writing_genre": "campus",
                        "target_readers": "青年读者",
                        "style": "克制、紧凑",
                        "chapter_count": 4,
                        "chapter_word_count": 1200,
                        "constraints": ["单女主", "无超自然"],
                    },
                )
                self.assertEqual("修改后的项目", updated["brief"]["title"])
                self.assertEqual(4, updated["brief"]["chapter_count"])
                self.assertTrue(updated["planning_stale"])

                project_root = Path(temp_dir) / "projects" / project["id"]
                ArtifactStore(project_root).write_chapter_plan(3, {"chapter_number": 3})
                with self.assertRaises(urllib.error.HTTPError) as project_error:
                    self.request(
                        project_url,
                        method="PUT",
                        body={"chapter_count": 2},
                    )
                self.assertEqual(400, project_error.exception.code)

                custom = self.request(
                    base + "/api/agents",
                    method="POST",
                    body={
                        "id": "continuity_reviewer",
                        "display_name": "一致性审查员",
                        "hook": "after_chapter_draft",
                        "output_mode": "json",
                        "prompt": "检查人物状态、时间线和世界规则是否与当前章节保持一致。",
                    },
                )["agent"]
                changed_agent = self.request(
                    base + f"/api/agents/{custom['id']}",
                    method="PUT",
                    body={
                        "display_name": "连续性审校员",
                        "hook": "after_chapter_plan",
                        "output_mode": "text",
                        "prompt": "逐项核对人物状态、时间线、伏笔和世界规则，并输出明确的修改建议。",
                        "enabled": True,
                    },
                )["agent"]
                self.assertEqual("连续性审校员", changed_agent["display_name"])
                self.assertEqual("after_chapter_plan", changed_agent["hook"])
                self.assertTrue(changed_agent["enabled"])

                self.request(
                    base + f"/api/agents/{custom['id']}/profile",
                    method="PUT",
                    body={"temperature": 0.2},
                )
                deleted_agent = self.request(
                    base + f"/api/agents/{custom['id']}", method="DELETE"
                )
                self.assertTrue(deleted_agent["deleted"])
                self.assertIsNone(
                    AgentProfileStore(temp_dir).get(custom["id"])["temperature"]
                )
                with self.assertRaises(urllib.error.HTTPError) as builtin_error:
                    self.request(base + "/api/agents/chapter_draft", method="DELETE")
                self.assertEqual(400, builtin_error.exception.code)

                deleted_project = self.request(project_url, method="DELETE")
                self.assertTrue(deleted_project["deleted"])
                self.assertFalse(project_root.exists())
                self.assertEqual([], self.request(base + "/api/projects")["projects"])
            finally:
                server.shutdown()
                server.server_close()
                app.close(wait=True)

    @staticmethod
    def request(url: str, method: str = "GET", body: dict | None = None) -> dict:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body else None
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
