from __future__ import annotations

import unittest

from novel_agents.memory_chunking import chunk_chapter


class MemoryScaleTests(unittest.TestCase):
    def test_seven_hundred_thousand_character_book_can_be_chunked(self) -> None:
        paragraph = "林知夏推开广播站的门，周叙正在修理旧设备。两个人核对当天的采访记录，并把新的线索写进时间表。"
        repeats = 700_000 // (len(paragraph) + 2) + 1
        text = ((paragraph + "\n\n") * repeats)[:700_000]
        chunks = chunk_chapter(
            1,
            text,
            {
                "target_chars": 400,
                "max_chars": 600,
                "overlap_chars": 80,
                "preserve_scene_boundary": True,
                "preserve_dialogue_block": True,
            },
        )

        self.assertEqual(700_000, len(text))
        self.assertGreater(len(chunks), 1000)
        self.assertTrue(all(0 < len(chunk.text) <= 600 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()

