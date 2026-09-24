from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    dimensions: int

    def embed_documents(self, texts: list[str], batch_size: int = 8) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class HashEmbeddingProvider:
    """Deterministic test embedder; never used for production memory."""

    def __init__(self, dimensions: int = 512) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        return np.vstack([self._embed(text) for text in texts]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        normalized = re.sub(r"\s+", "", text.lower())
        units = list(normalized) + [normalized[index : index + 2] for index in range(max(0, len(normalized) - 1))]
        for unit in units:
            digest = hashlib.blake2b(unit.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % self.dimensions
            vector[index] += 1.0 if value & 1 else -1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector


class LocalSentenceTransformerEmbedder:
    _models: dict[str, Any] = {}
    _lock = threading.RLock()

    def __init__(
        self,
        settings: dict[str, Any],
        cache_dir: str | Path,
        allow_download: bool = False,
    ) -> None:
        self.settings = settings
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dimensions = int(settings["dimensions"])
        self.allow_download = allow_download
        self._model = None

    def embed_documents(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        model = self._load_model()
        vectors = model.encode(
            texts,
            batch_size=int(batch_size),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return self._coerce_dimensions(np.asarray(vectors, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        query = text
        if "bge" in str(self.settings["model"]).lower():
            query = "为这个句子生成表示以用于检索相关文章：" + text
        return self.embed_documents([query], batch_size=1)[0]

    def prepare(self) -> dict[str, Any]:
        requested_quantization = str(self.settings["quantization"])
        if requested_quantization not in {"int8", "fp32"}:
            raise RuntimeError("当前本地嵌入仅支持 INT8 或 FP32")
        if self._backend() != "onnx" and requested_quantization != "fp32":
            raise RuntimeError("INT8 量化仅支持 ONNX 后端")

        prepared = self._read_state()
        if self._state_is_compatible(prepared):
            self._model = None
            probe = self.embed_query("本地记忆模型连接测试")
            prepared["dimensions"] = int(probe.shape[0])
            prepared["prepared_at"] = datetime.now(timezone.utc).isoformat()
            self._write_state(prepared)
            return prepared

        model = self._load_model(use_prepared=False)
        quantized_path = ""
        quantized_file = ""
        quantization_actual = "fp32"
        if self._backend() == "onnx" and requested_quantization == "int8":
            try:
                from sentence_transformers.backend import export_dynamic_quantized_onnx_model
            except ImportError as exc:
                raise RuntimeError(
                    "缺少 optimum[onnxruntime]，无法导出 INT8 ONNX 模型"
                ) from exc
            safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(self.settings["model"]))
            revision = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            output_dir = self.cache_dir / "quantized" / safe_name / revision
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(output_dir))
            export_dynamic_quantized_onnx_model(
                model,
                "avx2",
                str(output_dir),
                file_suffix="int8",
            )
            candidates = sorted(output_dir.rglob("*int8*.onnx"))
            if not candidates:
                raise RuntimeError("INT8 导出完成后未找到量化 ONNX 文件")
            quantized_path = str(output_dir)
            quantized_file = candidates[0].relative_to(output_dir).as_posix()
            quantization_actual = "int8"
            preliminary = {
                "model": self.settings["model"],
                "model_reference": self._model_reference(),
                "backend": "onnx",
                "quantization_requested": "int8",
                "quantization_actual": "int8",
                "quantized_model_path": quantized_path,
                "quantized_file": quantized_file,
                "dimensions": self.dimensions,
                "prepared_at": datetime.now(timezone.utc).isoformat(),
            }
            self._write_state(preliminary)
            self._model = None
            model = self._load_model()
        probe = self.embed_query("本地记忆模型连接测试")
        state = {
            "model": self.settings["model"],
            "model_reference": self._model_reference(),
            "backend": self._backend(),
            "quantization_requested": requested_quantization,
            "quantization_actual": quantization_actual,
            "quantized_model_path": quantized_path,
            "quantized_file": quantized_file,
            "dimensions": int(probe.shape[0]),
            "prepared_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_state(state)
        return state

    def _load_model(self, use_prepared: bool = True):
        if self._model is not None:
            return self._model
        reference = self._model_reference()
        model_kwargs = None
        prepared = self._read_state()
        if (
            use_prepared
            and self._state_is_compatible(prepared)
            and prepared.get("quantized_model_path")
        ):
            reference = str(prepared["quantized_model_path"])
            model_kwargs = {"file_name": str(prepared.get("quantized_file", ""))}
        key = json.dumps(
            {
                "reference": reference,
                "backend": self._backend(),
                "device": self.settings["device"],
                "dimensions": self.dimensions,
                "download": self.allow_download,
                "model_kwargs": model_kwargs,
            },
            sort_keys=True,
        )
        with self._lock:
            if key in self._models:
                self._model = self._models[key]
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("缺少 sentence-transformers，无法运行本地嵌入模型") from exc
            try:
                model = SentenceTransformer(
                    reference,
                    device=str(self.settings["device"]),
                    cache_folder=str(self.cache_dir),
                    local_files_only=not self.allow_download,
                    truncate_dim=self.dimensions,
                    backend=self._backend(),
                    model_kwargs=model_kwargs,
                )
            except Exception as exc:
                if not self.allow_download:
                    raise RuntimeError(
                        "本地嵌入模型尚未准备，请先在记忆页面执行“准备模型”"
                    ) from exc
                raise
            self._models[key] = model
            self._model = model
            return model

    def _model_reference(self) -> str:
        model_path = str(self.settings.get("model_path", "")).strip()
        return model_path or str(self.settings["model"])

    def _state_is_compatible(self, state: dict[str, Any]) -> bool:
        if not state:
            return False
        if state.get("model_reference", state.get("model")) != self._model_reference():
            return False
        if state.get("backend") != self._backend():
            return False
        if int(state.get("dimensions", 0) or 0) != self.dimensions:
            return False
        requested = str(self.settings["quantization"])
        if state.get("quantization_actual") != requested:
            return False
        if requested == "int8":
            root = Path(str(state.get("quantized_model_path", "")))
            file_name = str(state.get("quantized_file", ""))
            return bool(file_name) and (root / file_name).is_file()
        return True

    def _backend(self) -> str:
        return "onnx" if self.settings["provider"] == "local_onnx" else "torch"

    def _coerce_dimensions(self, vectors: np.ndarray) -> np.ndarray:
        if vectors.shape[1] == self.dimensions:
            return vectors
        if vectors.shape[1] > self.dimensions:
            return vectors[:, : self.dimensions]
        raise ValueError(
            f"嵌入模型输出 {vectors.shape[1]} 维，低于配置的 {self.dimensions} 维"
        )

    def _write_state(self, state: dict[str, Any]) -> None:
        path = self.cache_dir / "model_state.json"
        fd, temp_name = tempfile.mkstemp(prefix=".model-state.", suffix=".tmp", dir=self.cache_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(state, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temp_name, path)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise

    def _read_state(self) -> dict[str, Any]:
        path = self.cache_dir / "model_state.json"
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError, json.JSONDecodeError):
            return {}


def create_embedder(
    embedding_settings: dict[str, Any],
    cache_dir: str | Path,
    allow_download: bool = False,
) -> EmbeddingProvider:
    if embedding_settings["provider"] == "hash_test":
        return HashEmbeddingProvider(int(embedding_settings["dimensions"]))
    return LocalSentenceTransformerEmbedder(
        embedding_settings,
        cache_dir=cache_dir,
        allow_download=allow_download,
    )
