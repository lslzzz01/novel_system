from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from .artifacts import ArtifactStore
from .llm import MockLanguageModel, OpenAICompatibleClient
from .models import ProjectBrief
from .settings import load_workspace_env
from .workflow import NovelWorkflow


def _add_project_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project",
        type=Path,
        required=True,
        help="Path to a novel project directory.",
    )


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic local responses to verify the workflow.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("NOVEL_MODEL", ""),
        help="Chat-completions model name. Defaults to NOVEL_MODEL.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("NOVEL_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("NOVEL_TEMPERATURE", "0.4")),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate existing artifacts for the requested operation.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="novel-agents",
        description="Run the required-agent novel workflow.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a novel project.")
    _add_project_argument(init_parser)
    init_parser.add_argument("--title", required=True)
    init_parser.add_argument("--premise", required=True)
    init_parser.add_argument("--genre", required=True)
    init_parser.add_argument(
        "--writing-genre",
        choices=("xuanhuan", "campus", "urban"),
        default="",
        help="正文题材智能体；留空时根据 --genre 自动识别。",
    )
    init_parser.add_argument("--target-readers", required=True)
    init_parser.add_argument("--style", required=True)
    init_parser.add_argument("--chapters", type=int, required=True)
    init_parser.add_argument("--words-per-chapter", type=int, default=3000)
    init_parser.add_argument("--language", default="Simplified Chinese")
    init_parser.add_argument(
        "--constraint",
        action="append",
        default=[],
        help="Repeat this option to add multiple constraints.",
    )
    init_parser.add_argument("--overwrite", action="store_true")

    plan_parser = subparsers.add_parser("plan", help="Run all required planning agents.")
    _add_project_argument(plan_parser)
    _add_model_arguments(plan_parser)

    chapter_plan_parser = subparsers.add_parser(
        "chapter-plan", help="Create one just-in-time chapter specification."
    )
    _add_project_argument(chapter_plan_parser)
    _add_model_arguments(chapter_plan_parser)
    chapter_plan_parser.add_argument("--chapter", type=int, required=True)

    write_parser = subparsers.add_parser("write", help="Write one planned chapter.")
    _add_project_argument(write_parser)
    _add_model_arguments(write_parser)
    write_parser.add_argument("--chapter", type=int, required=True)

    run_parser = subparsers.add_parser("run", help="Plan and write all chapters.")
    _add_project_argument(run_parser)
    _add_model_arguments(run_parser)

    status_parser = subparsers.add_parser("status", help="Show saved workflow state.")
    _add_project_argument(status_parser)
    return parser


def _build_model(args: argparse.Namespace):
    if args.mock:
        return MockLanguageModel()
    api_key = os.getenv("NOVEL_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Set NOVEL_API_KEY (or OPENAI_API_KEY) in the environment or in .env, "
            "or pass --mock to verify locally."
        )
    if not args.model:
        raise ValueError("Set NOVEL_MODEL or pass --model.")
    if not 0 <= args.temperature <= 2:
        raise ValueError("temperature must be between 0 and 2")
    return OpenAICompatibleClient(
        api_key=api_key,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # 模型密钥来自环境变量，<cwd>/.env 是它的本地文件形式。
    load_workspace_env(Path.cwd())
    try:
        if args.command == "init":
            brief = ProjectBrief(
                title=args.title,
                premise=args.premise,
                genre=args.genre,
                target_readers=args.target_readers,
                style=args.style,
                chapter_count=args.chapters,
                writing_genre=args.writing_genre,
                chapter_word_count=args.words_per_chapter,
                language=args.language,
                constraints=args.constraint,
            )
            store = ArtifactStore(args.project)
            store.initialize(brief, overwrite=args.overwrite)
            print(f"Created project: {store.root}")
            return 0

        if args.command == "status":
            store = ArtifactStore(args.project)
            print(json.dumps(store.read_state(), ensure_ascii=False, indent=2))
            return 0

        model = _build_model(args)
        workflow = NovelWorkflow(args.project, model)
        if args.command == "plan":
            workflow.plan(force=args.force)
            print(f"Planning complete: {workflow.store.artifacts_dir}")
            return 0
        if args.command == "chapter-plan":
            path = workflow.plan_chapter(args.chapter, force=args.force)
            print(f"Chapter plan complete: {path}")
            return 0
        if args.command == "write":
            path = workflow.write_chapter(args.chapter, force=args.force)
            print(f"Chapter complete: {path}")
            return 0
        if args.command == "run":
            paths = workflow.run(force=args.force)
            print(f"Required workflow complete: {len(paths)} chapters")
            return 0
    except Exception as exc:
        parser.exit(1, f"error: {exc}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
