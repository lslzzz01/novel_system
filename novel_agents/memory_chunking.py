from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class MemoryChunk:
    id: str
    chapter_number: int
    scene_index: int
    chunk_index: int
    text: str
    text_hash: str
    importance: float = 0.5
    characters: list[str] | None = None
    location: str = ""
    timeline: str = ""
    keywords: list[str] | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["characters"] = data["characters"] or []
        data["keywords"] = data["keywords"] or []
        data["metadata"] = data["metadata"] or {}
        return data


def chunk_chapter(
    chapter_number: int,
    text: str,
    settings: dict[str, Any],
) -> list[MemoryChunk]:
    target = int(settings["target_chars"])
    maximum = int(settings["max_chars"])
    overlap = int(settings["overlap_chars"])
    scenes = _split_scenes(text) if settings.get("preserve_scene_boundary", True) else [text]
    result: list[MemoryChunk] = []
    global_index = 1
    for scene_index, scene in enumerate(scenes, start=1):
        units = _split_units(scene, maximum)
        current = ""
        previous_tail = ""
        for unit in units:
            candidate = f"{current}\n\n{unit}".strip() if current else unit.strip()
            if current and len(candidate) > maximum:
                chunk_text = _with_overlap(previous_tail, current, overlap, maximum)
                result.append(_make_chunk(chapter_number, scene_index, global_index, chunk_text))
                global_index += 1
                previous_tail = current[-overlap:] if overlap else ""
                current = unit.strip()
            else:
                current = candidate
            if len(current) >= target:
                chunk_text = _with_overlap(previous_tail, current, overlap, maximum)
                result.append(_make_chunk(chapter_number, scene_index, global_index, chunk_text))
                global_index += 1
                previous_tail = current[-overlap:] if overlap else ""
                current = ""
        if current:
            chunk_text = _with_overlap(previous_tail, current, overlap, maximum)
            result.append(_make_chunk(chapter_number, scene_index, global_index, chunk_text))
            global_index += 1
    return result


def _split_scenes(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").strip()
    lines = normalized.splitlines()
    scenes: list[list[str]] = [[]]
    for line in lines:
        stripped = line.strip()
        boundary = bool(
            re.fullmatch(r"(?:-{3,}|\*{3,}|#{2,}\s+.+|第[一二三四五六七八九十百零\d]+场.*)", stripped)
        )
        if boundary and any(item.strip() for item in scenes[-1]):
            scenes.append([])
        if not (stripped.startswith("# ") and len(scenes) == 1 and not scenes[-1]):
            scenes[-1].append(line)
    return ["\n".join(scene).strip() for scene in scenes if "\n".join(scene).strip()]


def _split_units(scene: str, maximum: int) -> list[str]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", scene) if item.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= maximum:
            units.append(paragraph)
            continue
        sentences = [
            item.strip()
            for item in re.split(r"(?<=[。！？!?；;])", paragraph)
            if item.strip()
        ]
        current = ""
        for sentence in sentences or [paragraph]:
            if len(sentence) > maximum:
                if current:
                    units.append(current)
                    current = ""
                units.extend(sentence[index : index + maximum] for index in range(0, len(sentence), maximum))
                continue
            if current and len(current) + len(sentence) > maximum:
                units.append(current)
                current = sentence
            else:
                current += sentence
        if current:
            units.append(current)
    return units


def _with_overlap(previous_tail: str, text: str, overlap: int, maximum: int) -> str:
    if not overlap or not previous_tail or text.startswith(previous_tail):
        return text.strip()
    available = max(0, maximum - len(text) - 1)
    prefix = previous_tail[-min(overlap, available) :] if available else ""
    return f"{prefix}\n{text}".strip() if prefix else text.strip()


def _make_chunk(
    chapter_number: int,
    scene_index: int,
    chunk_index: int,
    text: str,
) -> MemoryChunk:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return MemoryChunk(
        id=f"CH{chapter_number:03d}-S{scene_index:03d}-C{chunk_index:04d}-{digest[:8]}",
        chapter_number=chapter_number,
        scene_index=scene_index,
        chunk_index=chunk_index,
        text=text,
        text_hash=digest,
    )
