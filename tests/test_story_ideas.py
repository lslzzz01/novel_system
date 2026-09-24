from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from novel_agents.artifacts import ArtifactStore
from novel_agents.story_ideas import StoryIdeaStore, compose_premise, idea_to_project_values
from novel_agents.web import AppContext, make_handler


class StoryIdeaStoreTests(unittest.TestCase):
    def test_compose_premise_and_create_project_values(self) -> None:
        sections = {
            "core_relationship": "病态暗恋与被需要的温柔",
            "protagonist": "沈知微，安静克制",
            "counterpart": "林予，阳光可靠",
            "opening": "大学教室远观",
            "midgame": "桥遇后靠近",
            "ending": "互相离不开，但保留不安",
            "tone_scale": "初期 L1，常态 L2",
            "narration": "双视角，偏沈知微",
            "freeform": "",
        }
        premise = compose_premise(sections)
        self.assertIn("【核心关系】", premise)
        self.assertIn("沈知微", premise)
        values = idea_to_project_values(
            {
                "title": "失控依恋",
                "sections": sections,
            }
        )
        self.assertEqual("失控依恋", values["title"])
        self.assertIn("病态暗恋", values["premise"])
        # Project defaults remain available for optional create-from-idea API.
        self.assertIn(values["writing_genre"], {"campus", "urban", "xuanhuan"})

    def test_store_persists_ideas(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = StoryIdeaStore(temp_dir)
            created = store.create({"title": "草稿一"})
            updated = store.update(
                created["id"],
                {
                    "title": "重逢",
                    "sections": {
                        "core_relationship": "旧情人重逢",
                        "protagonist": "女主冷静",
                    },
                },
            )
            self.assertEqual("重逢", updated["title"])
            self.assertIn("旧情人重逢", updated["premise"])
            listed = store.list_ideas()
            self.assertEqual(1, len(listed))
            loaded = store.get(created["id"])
            self.assertEqual("女主冷静", loaded["sections"]["protagonist"])
            store.delete(created["id"])
            self.assertEqual([], store.list_ideas())


class StoryIdeaHttpTests(unittest.TestCase):
    def request(self, url: str, method: str = "GET", body: dict | None = None):
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if body is not None else {},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    def test_story_idea_routes_create_project(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = AppContext(temp_dir)
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                created = self.request(
                    base + "/api/story-ideas",
                    method="POST",
                    body={},
                )["idea"]
                idea_id = created["id"]
                saved = self.request(
                    base + f"/api/story-ideas/{idea_id}",
                    method="PUT",
                    body={
                        "title": "重逢草稿",
                        "sections": {
                            "core_relationship": "分手五年后的重逢",
                            "protagonist": "女主更冷静",
                            "counterpart": "男主仍记得细节",
                            "opening": "机场偶遇",
                            "midgame": "合作项目被迫靠近",
                            "ending": "重新选择彼此",
                            "tone_scale": "L1 开局，中段 L2",
                            "narration": "双视角",
                            "freeform": "不要狗血误会连发",
                        },
                    },
                )["idea"]
                self.assertEqual("重逢草稿", saved["title"])
                self.assertIn("【核心关系】", saved["premise"])
                self.assertIn("机场偶遇", saved["premise"])

                listed = self.request(base + "/api/story-ideas")["ideas"]
                self.assertEqual(1, len(listed))

                # Preferred flow: idea only supplies premise; project form owns the rest.
                project = self.request(
                    base + "/api/projects",
                    method="POST",
                    body={
                        "title": "重逢",
                        "premise": saved["premise"],
                        "genre": "都市情感",
                        "writing_genre": "urban",
                        "target_readers": "青年读者",
                        "style": "克制缠绵",
                        "chapter_count": 80,
                        "chapter_word_count": 3500,
                        "constraints": ["单女主"],
                        "memory_enabled": True,
                    },
                )
                self.assertTrue(project["id"])
                self.assertEqual("重逢", project["brief"]["title"])
                self.assertEqual("urban", project["brief"]["writing_genre"])
                self.assertIn("机场偶遇", project["brief"]["premise"])

                project_root = Path(temp_dir) / "projects" / project["id"]
                brief = ArtifactStore(project_root).load_brief()
                self.assertEqual(80, brief.chapter_count)
                self.assertTrue((project_root / "project.json").exists())
            finally:
                server.shutdown()
                app.close(wait=True)


if __name__ == "__main__":
    unittest.main()
