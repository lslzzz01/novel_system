from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


WRITING_GENRES = {"xuanhuan", "campus", "urban"}


def infer_writing_genre(genre: str) -> str:
    normalized = genre.strip().lower()
    keyword_groups = (
        (
            "xuanhuan",
            (
                "玄幻",
                "仙侠",
                "修仙",
                "奇幻",
                "武侠",
                "高武",
                "fantasy",
                "cultivation",
                "wuxia",
            ),
        ),
        (
            "campus",
            (
                "校园",
                "青春",
                "高中",
                "大学",
                "学生",
                "学园",
                "campus",
                "school",
                "college",
            ),
        ),
        (
            "urban",
            (
                "都市",
                "职场",
                "现实",
                "商战",
                "娱乐圈",
                "urban",
                "city",
                "workplace",
            ),
        ),
    )
    for writing_genre, keywords in keyword_groups:
        if any(keyword in normalized for keyword in keywords):
            return writing_genre
    return "urban"


def normalize_writing_genre(value: str, genre: str) -> str:
    normalized = value.strip().lower()
    aliases = {
        "玄幻": "xuanhuan",
        "仙侠": "xuanhuan",
        "校园": "campus",
        "校园小说": "campus",
        "都市": "urban",
        "都市小说": "urban",
        "auto": "",
        "自动": "",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized or infer_writing_genre(genre)


@dataclass(slots=True)
class ProjectBrief:
    title: str
    premise: str
    genre: str
    target_readers: str
    style: str
    chapter_count: int
    chapter_word_count: int = 3000
    language: str = "Simplified Chinese"
    constraints: list[str] = field(default_factory=list)
    writing_genre: str = ""

    def __post_init__(self) -> None:
        context = " ".join((self.title, self.genre, self.premise))
        self.writing_genre = normalize_writing_genre(self.writing_genre, context)

    def validate(self) -> None:
        text_fields = {
            "title": self.title,
            "premise": self.premise,
            "genre": self.genre,
            "target_readers": self.target_readers,
            "style": self.style,
            "language": self.language,
        }
        missing = [name for name, value in text_fields.items() if not value.strip()]
        if missing:
            raise ValueError(f"Missing project fields: {', '.join(missing)}")
        if self.chapter_count < 1:
            raise ValueError("chapter_count must be at least 1")
        if self.chapter_word_count < 300:
            raise ValueError("chapter_word_count must be at least 300")
        if self.writing_genre not in WRITING_GENRES:
            raise ValueError("writing_genre must be xuanhuan, campus, or urban")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectBrief":
        brief = cls(
            title=str(data.get("title", "")),
            premise=str(data.get("premise", "")),
            genre=str(data.get("genre", "")),
            target_readers=str(data.get("target_readers", "")),
            style=str(data.get("style", "")),
            chapter_count=int(data.get("chapter_count", 0)),
            writing_genre=str(data.get("writing_genre", "")),
            chapter_word_count=int(data.get("chapter_word_count", 3000)),
            language=str(data.get("language", "Simplified Chinese")),
            constraints=[str(item) for item in data.get("constraints", [])],
        )
        brief.validate()
        return brief
