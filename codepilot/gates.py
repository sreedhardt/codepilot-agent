"""Human-in-the-loop approval gates.

A gate pauses the run with a LangGraph interrupt and resumes with the human's
decision. The run must have a checkpointer. Interrupts raised inside subagents
surface on the root agent, so the driver (CLI or TUI) only watches one place.

On resume, the tool that raised the interrupt runs again from its start, so
code before `request_approval` must have no side effects; record decisions and
act only after it returns.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.types import Command, interrupt

LARGE_COMMIT_FILES = 5
FREE_TEST_FAILURES = 2
MAX_TEST_RETRIES = 3  # verification rounds after a failure; no approval can extend this


class Gate(StrEnum):
    PR_TO_DEFAULT_BRANCH = "pr_to_default_branch"
    LARGE_COMMIT = "large_commit"
    GIT_PUSH = "git_push"
    RETRY_AFTER_FAILURES = "retry_after_failures"
    REVIEW_OVERRULE = "review_overrule"
    BLOCKED_OPERATION = "blocked_operation"


@dataclass
class Decision:
    approved: bool
    note: str = ""

    @classmethod
    def from_reply(cls, reply: Any) -> "Decision":
        if isinstance(reply, dict):
            return cls(bool(reply.get("approved")), str(reply.get("note", "")))
        if isinstance(reply, str):
            return cls(reply.strip().lower() in {"approve", "approved", "yes", "y"})
        return cls(bool(reply))


def request_approval(gates: list[Gate], summary: str, details: dict, audit_log: Path | None = None) -> Decision:
    """Pause for a human decision. Returns once the run is resumed with one."""
    reply = interrupt({"gates": [g.value for g in gates], "summary": summary, "details": details})
    decision = Decision.from_reply(reply)
    if audit_log is not None:
        with audit_log.open("a") as f:
            f.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "gates": [g.value for g in gates],
                                "summary": summary, **asdict(decision)}) + "\n")
    return decision


class RetryGateMiddleware(AgentMiddleware):
    """Requires approval before the Coder's third (and later) verification round after failed test runs, and
    refuses any round beyond MAX_TEST_RETRIES retries."""

    def __init__(self, audit_log: Path | None = None, free_failures: int = FREE_TEST_FAILURES) -> None:
        super().__init__()
        self.audit_log = audit_log
        self.free_failures = free_failures
        self.failures = 0

    def wrap_tool_call(self, request, handler):
        call = request.tool_call
        if call["name"] != "task" or call["args"].get("subagent_type") != "test_agent":
            return handler(request)
        if self.failures > MAX_TEST_RETRIES:
            return ToolMessage(
                content=f"STOPPED: the tests failed {self.failures} times and the limit is {MAX_TEST_RETRIES} retries. "
                        "Stop and report the failing tests.",
                tool_call_id=call["id"],
            )
        if self.failures >= self.free_failures:
            decision = request_approval(
                [Gate.RETRY_AFTER_FAILURES],
                f"The Test Agent has reported FAIL {self.failures} times. Allow another fix-and-verify round?",
                {"failed_rounds": self.failures, "test_report": "/working/test_report.md"},
                self.audit_log,
            )
            if not decision.approved:
                return ToolMessage(
                    content="REJECTED by human: no further verification rounds. Stop and report the failing tests.",
                    tool_call_id=call["id"],
                )
        result = handler(request)
        if _tool_result_text(result).lstrip().startswith("FAIL"):
            self.failures += 1
        return result


def _tool_result_text(result: ToolMessage | Command) -> str:
    """Text of a tool result; DeepAgents' `task` tool wraps its ToolMessage in a Command update."""
    if isinstance(result, ToolMessage):
        return str(result.content)
    messages = (result.update or {}).get("messages", []) if isinstance(result, Command) else []
    return str(messages[-1].content) if messages else ""
