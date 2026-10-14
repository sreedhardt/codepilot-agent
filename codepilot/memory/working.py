"""Working memory: the current task's state, maintained by code and handed to each subagent at spawn time.

Subagents start with a fresh context, so instead of relying on conversation
history, every delegation through the `task` tool gets a compact snapshot of
the task state appended to its instructions. The snapshot is also kept at
/working/working_memory.json (read-only for agents) and archived when the task
reaches DONE or FAILED.
"""

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from langchain.agents.middleware import AgentMiddleware

from codepilot.workspace import TaskWorkspace

PATH_TOKEN = re.compile(r"[\w./-]+\.\w+")


@dataclass
class WorkingMemory:
    issue_number: int
    title: str
    url: str
    task_type: str
    skill: str
    state: str = "TRIAGED"
    repo_map: str = "/working/repo_map.md"
    relevant_files: list[str] = field(default_factory=list)
    diff: str | None = None
    test_result: str | None = None
    retry_count: int = 0
    lessons: str | None = None
    recent_sessions: str | None = None

    def refresh(self, ws: TaskWorkspace, state: str, retry_count: int) -> None:
        """Re-read the task's progress from the workspace files."""
        self.state, self.retry_count = state, retry_count
        relevant = ws.working / "relevant_files.md"
        if relevant.exists():
            found = []
            for token in PATH_TOKEN.findall(relevant.read_text()):
                path = token.lstrip("/")
                if path not in found and (ws.sandbox / path).is_file():
                    found.append(path)
            self.relevant_files = found[:10]
        if (ws.working / "proposed_diff.txt").exists():
            self.diff = "/working/proposed_diff.txt"
        report = ws.working / "test_report.md"
        if report.exists():
            verdict = next((line.strip() for line in report.read_text().splitlines()
                            if re.search(r"\b(PASS|FAIL|passed|failed)\b", line)), None)
            self.test_result = verdict[:160] if verdict else "see /working/test_report.md"
        self.lessons = "/working/lessons.md" if (ws.working / "lessons.md").exists() else None
        self.recent_sessions = "/working/recent_sessions.md" if (ws.working / "recent_sessions.md").exists() else None

    def snapshot(self) -> str:
        data = {k: v for k, v in asdict(self).items() if v not in (None, [], "")}
        return "Working memory (current task state, maintained by CodePilot):\n" + json.dumps(data, indent=1)

    def save(self, ws: TaskWorkspace) -> None:
        ws.write("working_memory.json", json.dumps(asdict(self), indent=1))

    def archive(self, ws: TaskWorkspace) -> None:
        """Keep the final state for the record and clear the live working memory."""
        live = ws.working / "working_memory.json"
        if live.exists():
            live.rename(ws.working / "working_memory.final.json")


class WorkingMemoryMiddleware(AgentMiddleware):
    """Appends the working-memory snapshot to every subagent delegation."""

    def __init__(self, snapshot: Callable[[], str]) -> None:
        super().__init__()
        self.snapshot = snapshot

    def wrap_tool_call(self, request, handler):
        call = request.tool_call
        if call["name"] != "task":
            return handler(request)
        args = dict(call["args"])
        args["description"] = f"{args.get('description', '')}\n\n{self.snapshot()}"
        return handler(request.override(tool_call={**call, "args": args}))
