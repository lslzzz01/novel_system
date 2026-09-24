from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from novel_agents.embedding import LocalSentenceTransformerEmbedder


class FakeSentenceTransformer:
    def save_pretrained(self, output_dir: str) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 64), dtype=np.float32)


class EmbeddingPreparationTests(unittest.TestCase):
    def test_int8_prepare_exports_and_reuses_quantized_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = {
                "provider": "local_onnx",
                "model": "BAAI/bge-small-zh-v1.5",
                "model_path": "",
                "device": "cpu",
                "quantization": "int8",
                "dimensions": 64,
            }
            calls = []

            def fake_export(model, quantization_config, model_name_or_path, **kwargs):
                calls.append((quantization_config, kwargs.get("file_suffix")))
                target = Path(model_name_or_path) / "onnx" / "model_int8.onnx"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"fake-onnx")

            package = types.ModuleType("sentence_transformers")
            package.__path__ = []
            backend = types.ModuleType("sentence_transformers.backend")
            backend.export_dynamic_quantized_onnx_model = fake_export
            modules = {
                "sentence_transformers": package,
                "sentence_transformers.backend": backend,
            }
            embedder = LocalSentenceTransformerEmbedder(
                settings, temp_dir, allow_download=True
            )
            fake_model = FakeSentenceTransformer()
            with patch.dict(sys.modules, modules), patch.object(
                embedder, "_load_model", return_value=fake_model
            ):
                state = embedder.prepare()

            self.assertEqual([("avx2", "int8")], calls)
            self.assertEqual("int8", state["quantization_actual"])
            self.assertTrue(
                (Path(state["quantized_model_path"]) / state["quantized_file"]).is_file()
            )

            reused = LocalSentenceTransformerEmbedder(
                settings, temp_dir, allow_download=True
            )
            with patch.dict(sys.modules, modules), patch.object(
                reused, "_load_model", return_value=fake_model
            ):
                second = reused.prepare()
            self.assertEqual(1, len(calls))
            self.assertEqual(state["quantized_file"], second["quantized_file"])


if __name__ == "__main__":
    unittest.main()
