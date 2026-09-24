from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator


class RuntimeCoordinator:
    """Coordinates CPU indexing with prose generation on lightweight machines."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._writing: set[str] = set()
        self._manually_paused: set[str] = set()

    @contextmanager
    def writing(self, project_id: str) -> Iterator[None]:
        with self._condition:
            self._writing.add(project_id)
            self._condition.notify_all()
        try:
            yield
        finally:
            with self._condition:
                self._writing.discard(project_id)
                self._condition.notify_all()

    def pause(self, project_id: str) -> None:
        with self._condition:
            self._manually_paused.add(project_id)
            self._condition.notify_all()

    def resume(self, project_id: str) -> None:
        with self._condition:
            self._manually_paused.discard(project_id)
            self._condition.notify_all()

    def wait_for_index_slot(
        self,
        project_id: str,
        pause_during_writing: bool,
        cancelled: threading.Event | None = None,
    ) -> None:
        with self._condition:
            while project_id in self._manually_paused or (
                pause_during_writing and project_id in self._writing
            ):
                if cancelled and cancelled.is_set():
                    raise RuntimeError("索引任务已取消")
                self._condition.wait(timeout=0.5)

    def status(self, project_id: str) -> dict[str, bool]:
        with self._condition:
            return {
                "writing": project_id in self._writing,
                "manually_paused": project_id in self._manually_paused,
            }

