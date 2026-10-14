"""Episodic memory: what happened in each session, persisted in a LangGraph store.

A session is one CodePilot process (a `run`, a `poll` loop or a TUI session).
Each task is recorded as it finishes, so a crash loses nothing, and the
session is closed with a structured summary. The Orchestrator reads the last
three summaries at the start of a task.
"""

import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from langgraph.store.base import BaseStore
from langgraph.store.sqlite import SqliteStore


def open_store(path: Path) -> BaseStore:
    path.parent.mkdir(parents=True, exist_ok=True)
    store = SqliteStore(sqlite3.connect(path, check_same_thread=False, isolation_level=None))
    store.setup()
    return store


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class TaskEpisode:
    issue_number: int
    title: str
    task_type: str
    state: str
    outcome: str
    files_modified: list[str] = field(default_factory=list)
    seconds: float = 0.0
    pr_url: str | None = None


class EpisodicMemory:
    def __init__(self, store: BaseStore, repo: str) -> None:
        self.store = store
        self.namespace = ("codepilot", "sessions", repo.replace("/", "__"))
        self.session_id: str | None = None
        self.started: str | None = None
        self.tasks: list[dict] = []

    def start_session(self) -> str:
        self.started = _now()
        # A random suffix keeps two sessions started in the same second from overwriting each other.
        self.session_id = f"{self.started.replace(':', '')}-{uuid.uuid4().hex[:6]}"
        self.tasks = []
        self._save(ended=None)
        return self.session_id

    def record(self, episode: TaskEpisode) -> None:
        self.tasks.append(episode.__dict__.copy())
        self._save(ended=None)

    def end_session(self) -> dict:
        return self._save(ended=_now())

    def _save(self, ended: str | None) -> dict:
        value = {"session_id": self.session_id, "started": self.started, "ended": ended,
                 "tasks": self.tasks, "summary": summarize(self.started, ended, self.tasks)}
        self.store.put(self.namespace, self.session_id, value)
        return value

    def recent(self, n: int = 3, exclude_current: bool = True) -> list[dict]:
        items = self.store.search(self.namespace, limit=1000)
        sessions = sorted((i.value for i in items), key=lambda v: v["started"], reverse=True)
        if exclude_current:
            sessions = [s for s in sessions if s["session_id"] != self.session_id]
        return sessions[:n]

    def render_recent(self, n: int = 3) -> str:
        sessions = self.recent(n)
        if not sessions:
            return "# Recent sessions\n\nNo previous sessions recorded.\n"
        return "# Recent sessions (most recent first)\n\n" + "\n\n".join(s["summary"] for s in sessions) + "\n"

    def previous_attempts(self, issue_number: int, n: int = 3) -> list[dict]:
        """Earlier attempts at this issue in the last `n` sessions, most recent first."""
        return [t for s in self.recent(n) for t in reversed(s["tasks"]) if t["issue_number"] == issue_number]


def summarize(started: str | None, ended: str | None, tasks: list[dict]) -> str:
    """Deterministic summary: a structured record needs no model to describe it."""
    done = sum(t["state"] == "DONE" for t in tasks)
    failed = sum(t["state"] == "FAILED" for t in tasks)
    header = (f"## Session {started} -> {ended or 'in progress'}\n"
              f"{len(tasks)} task(s): {done} done, {failed} failed")
    lines = []
    for t in tasks:
        files = ", ".join(t["files_modified"]) or "no files"
        result = t["pr_url"] or t["outcome"][:120]
        lines.append(f"- #{t['issue_number']} {t['title']} [{t['task_type']}] {t['state']} in {t['seconds']:.0f}s; "
                     f"{files}; {result}")
    return "\n".join([header, *lines])
