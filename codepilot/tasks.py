"""Task state machine and the registry of tasks across runs.

States follow the agent pipeline:

    TRIAGED -> EXPLORING -> IMPLEMENTING <-> TESTING -> PR_OPENED -> DONE
                                  any state -> FAILED

The tracker derives state from what agents actually do (which subagent the
Orchestrator delegates to, when the Coder hands off to the Test Agent) and
from the PR tool's recorded outcome, never from an agent's own claims.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path


class TaskState(StrEnum):
    TRIAGED = "TRIAGED"
    EXPLORING = "EXPLORING"
    IMPLEMENTING = "IMPLEMENTING"
    TESTING = "TESTING"
    PR_OPENED = "PR_OPENED"
    DONE = "DONE"
    FAILED = "FAILED"


TRANSITIONS: dict[TaskState, set[TaskState]] = {
    TaskState.TRIAGED: {TaskState.EXPLORING, TaskState.IMPLEMENTING},
    TaskState.EXPLORING: {TaskState.IMPLEMENTING},
    TaskState.IMPLEMENTING: {TaskState.TESTING, TaskState.PR_OPENED, TaskState.DONE},
    TaskState.TESTING: {TaskState.IMPLEMENTING, TaskState.PR_OPENED, TaskState.DONE},
    TaskState.PR_OPENED: {TaskState.DONE},
    TaskState.DONE: set(),
    TaskState.FAILED: set(),
}
TERMINAL = {TaskState.DONE, TaskState.FAILED}

# Which delegation moves the task into which state.
DELEGATION_STATES = {
    ("orchestrator", "repo_explorer"): TaskState.EXPLORING,
    ("orchestrator", "coder"): TaskState.IMPLEMENTING,
    ("coder", "test_agent"): TaskState.TESTING,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class TaskRecord:
    issue_number: int
    title: str
    task_type: str
    skill: str
    workspace: str
    state: TaskState = TaskState.TRIAGED
    history: list[tuple[str, str]] = field(default_factory=list)
    pr_url: str | None = None
    outcome: str = ""
    active_agent: str = ""
    retries_after_fail: int = 0
    cost_usd: float = 0.0
    billing_blocked: bool = False  # failed because the model provider refused calls for billing, not on its merits
    pid: int = field(default_factory=os.getpid)  # the process running the task, to spot tasks orphaned by a crash

    def __post_init__(self) -> None:
        if not self.history:
            self.history.append((self.state.value, _now()))

    @property
    def dry_run(self) -> bool:
        """Finished without opening a PR because PRs were disabled: a preview, not a resolution."""
        return self.state == TaskState.DONE and self.pr_url is None and self.outcome.startswith("DRY RUN")

    def move(self, new: TaskState, note: str = "") -> bool:
        """Apply a transition if it is allowed; FAILED is reachable from any non-terminal state."""
        if new == self.state:
            return False
        allowed = (new == TaskState.FAILED and self.state not in TERMINAL) or new in TRANSITIONS[self.state]
        if not allowed:
            return False
        if self.state == TaskState.TESTING and new == TaskState.IMPLEMENTING:
            self.retries_after_fail += 1
        self.state = new
        self.history.append((new.value, _now()))
        if note:
            self.outcome = note
        return True


class TaskTracker:
    """Updates a TaskRecord from agent events and the workspace's result files."""

    def __init__(self, record: TaskRecord, on_change=None) -> None:
        self.record = record
        self.on_change = on_change or (lambda record: None)

    def observe(self, agent: str, tool_calls: list[dict]) -> None:
        """Called for each agent message; `agent` is the message's agent name."""
        self.record.active_agent = agent
        for call in tool_calls:
            target = call.get("args", {}).get("subagent_type") if call.get("name") == "task" else None
            new = DELEGATION_STATES.get((agent, target))
            if new is None and agent == "coder" and self.record.state == TaskState.TESTING and call.get("name") == "edit_file":
                new = TaskState.IMPLEMENTING  # Coder is fixing after a failed verification round
            if new is not None and self.record.move(new):
                self.on_change(self.record)

    def finish(self, working: Path, error: str | None = None) -> TaskRecord:
        """Settle the final state from the PR tool's recorded outcome."""
        result_file = working / "pr_result.md"
        result = result_file.read_text().strip().splitlines()[0] if result_file.exists() else ""
        if error:
            self.record.move(TaskState.FAILED, error)
        elif result.startswith("PR_OPENED"):
            self.record.pr_url = result.split(maxsplit=1)[1]
            self.record.move(TaskState.PR_OPENED, result)
            self.record.move(TaskState.DONE, result)
        elif result.startswith("DRY RUN"):
            self.record.move(TaskState.DONE, result)
        else:
            self.record.move(TaskState.FAILED, result or "no pull request was attempted")
        self.record.active_agent = ""
        self.on_change(self.record)
        return self.record


class TaskRegistry:
    """Latest record per issue, persisted as JSON so the poller survives restarts."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: dict[int, TaskRecord] = {}
        if path.exists():
            for number, data in json.loads(path.read_text()).items():
                data["state"] = TaskState(data["state"])
                data["history"] = [tuple(h) for h in data["history"]]
                data.setdefault("pid", 0)  # written before PIDs were recorded: owner unknown
                self.records[int(number)] = TaskRecord(**data)

    def save(self, record: TaskRecord) -> None:
        self.records[record.issue_number] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({n: asdict(r) for n, r in sorted(self.records.items())}, indent=1))

    def recover_stale(self) -> list[TaskRecord]:
        """Fail unfinished tasks whose process has exited, so they stop blocking their issue as "in progress".

        They fail rather than restart at once: a task that crashed the process would otherwise crash it again
        on every start. The poller retries them after the usual cooldown for failed tasks.
        """
        recovered = []
        for record in self.records.values():
            if record.state not in TERMINAL and not _alive(record.pid):
                record.move(TaskState.FAILED, f"interrupted: CodePilot stopped while the task was {record.state}")
                record.active_agent = ""
                recovered.append(record)
        if recovered:
            self.save(recovered[0])
        return recovered

    def in_progress(self) -> set[int]:
        return {n for n, r in self.records.items() if r.state not in TERMINAL}

    def get(self, issue_number: int) -> TaskRecord | None:
        return self.records.get(issue_number)


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)  # signal 0 only checks that the process exists
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    return True
