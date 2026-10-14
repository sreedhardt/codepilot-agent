"""Run one issue end to end: classify, prepare the workspace, run the agent team, settle the task state.

Front ends (CLI, poller, TUI) supply callbacks for log lines, state changes and
approval decisions; the runner itself never prints or reads input.
"""

import faulthandler
import json
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from codepilot import agents
from codepilot.agents import TaskContext, build_orchestrator
from codepilot.classifier import IssueClassifier, build_classifier
from codepilot.repo_map import EmbeddingIndex, build_repo_map
from codepilot.sandbox import BlockedAction
from codepilot.skills import skill_for
from codepilot.tasks import TaskRecord, TaskRegistry, TaskTracker
from codepilot.workspace import ROOT, RUNS_DIR, TaskWorkspace

CACHE = ROOT / ".codepilot_cache"
REGISTRY_PATH = ROOT / "runs" / "tasks.json"


@dataclass
class RunOptions:
    open_pr: bool = False
    recursion_limit: int = 150
    watchdog_seconds: int | None = 90
    runs_dir: Path = RUNS_DIR


@dataclass
class Callbacks:
    log: Callable[[str], None] = lambda line: None
    state: Callable[[TaskRecord], None] = lambda record: None
    approve: Callable[[dict, TaskWorkspace], dict] = lambda request, ws: {"approved": False, "note": "no approver"}


@dataclass
class RunResult:
    record: TaskRecord
    verdict: str
    seconds: float
    workspace: TaskWorkspace
    blocked: list[BlockedAction] = field(default_factory=list)
    retries: list[tuple[str, str]] = field(default_factory=list)
    approvals: int = 0


def run_issue(
    issue,
    options: RunOptions | None = None,
    callbacks: Callbacks | None = None,
    registry: TaskRegistry | None = None,
    classifier: IssueClassifier | None = None,
) -> RunResult:
    """`issue` is a PyGithub Issue (number, title, body, html_url, user)."""
    options, callbacks = options or RunOptions(), callbacks or Callbacks()
    registry = registry or TaskRegistry(REGISTRY_PATH)
    started, retries_before = time.monotonic(), len(agents.RETRY_LOG)

    classification = (classifier or build_classifier()).classify(issue.title, issue.body or "")
    skill = skill_for(classification.task_type)
    ws = TaskWorkspace(issue.number, runs_dir=options.runs_dir)
    workspace = ws.root.relative_to(ROOT) if ws.root.is_relative_to(ROOT) else ws.root
    record = TaskRecord(issue.number, issue.title, classification.task_type.value, skill.skill_id, str(workspace))

    def on_change(r: TaskRecord) -> None:
        registry.save(r)
        callbacks.state(r)

    tracker = TaskTracker(record, on_change)
    on_change(record)
    callbacks.log(f"#{issue.number} {issue.title} -> {classification.task_type} "
                  f"({classification.confidence:.2f}), {skill.skill_id}, workspace {record.workspace}")

    final_text, error = "", None
    try:
        repo_map = build_repo_map(ws.sandbox, cache_dir=CACHE)
        index = EmbeddingIndex(repo_map, CACHE / "chroma")
        ws.write("repo_map.md", repo_map.render())
        ws.write("task.md", (
            f"# Issue #{issue.number}: {issue.title}\n\n{issue.body}\n\n"
            f"Task type: {classification.task_type}\nSkill: {skill.skill_id} (installed at /skills/{skill.name}/)\n"
            f"Skill workflow: {' -> '.join(skill.workflow_steps)}\n"
        ))
        ctx = TaskContext(issue.number, issue.title, issue.html_url,
                          issue.user.login if issue.user else None, classification.task_type)
        agent = build_orchestrator(ws, repo_map, skill, ctx, index, checkpointer=InMemorySaver(),
                                   open_pr=options.open_pr)
        config = {"recursion_limit": options.recursion_limit, "configurable": {"thread_id": ws.root.name}}
        payload = {"messages": [{"role": "user",
                                 "content": f"Resolve issue #{issue.number}. Start with /working/task.md."}]}
        while True:
            final_text = _stream(agent, payload, config, tracker, callbacks, started, options, final_text)
            pending = agent.get_state(config).interrupts
            if not pending:
                break
            payload = Command(resume={p.id: callbacks.approve(p.value, ws) for p in pending})
    except Exception as e:  # noqa: BLE001 - one broken task must not stop the poller or the TUI
        error = f"{type(e).__name__}: {e}"
        callbacks.log(f"ERROR {error}\n{traceback.format_exc(limit=5)}")
    finally:
        if options.watchdog_seconds:
            faulthandler.cancel_dump_traceback_later()

    tracker.finish(ws.working, error)
    verdicts = [line.strip(" *`") for line in final_text.splitlines() if "VERDICT:" in line]
    result = RunResult(
        record=record,
        verdict=verdicts[-1] if verdicts else "NO VERDICT LINE from the Orchestrator",
        seconds=time.monotonic() - started,
        workspace=ws,
        blocked=list(ws.shell.blocked),
        retries=agents.RETRY_LOG[retries_before:],
        approvals=ws.approvals.read_text().count("\n") if ws.approvals.exists() else 0,
    )
    (ws.root / "task.json").write_text(json.dumps(asdict(record), indent=1))
    return result


def _stream(agent, payload, config, tracker: TaskTracker, callbacks: Callbacks, started: float,
            options: RunOptions, final_text: str) -> str:
    last = None
    if options.watchdog_seconds:
        faulthandler.dump_traceback_later(options.watchdog_seconds, exit=False)
    for namespace, update in agent.stream(payload, config=config, stream_mode="updates", subgraphs=True):
        if options.watchdog_seconds:  # re-arm: dump thread stacks only if progress stops this long
            faulthandler.cancel_dump_traceback_later()
            faulthandler.dump_traceback_later(options.watchdog_seconds, exit=False)
        for node_update in update.values():
            if not isinstance(node_update, dict):
                continue
            for msg in node_update.get("messages", []):
                if not isinstance(msg, AIMessage):
                    continue
                who = msg.name or ("orchestrator" if not namespace else namespace[-1].split(":")[0])
                tracker.observe(who, msg.tool_calls)
                if who == "orchestrator" and msg.text:
                    final_text = msg.text
                calls = ", ".join(
                    f"{c['name']}({c['args'].get('subagent_type', '')})" if c["name"] == "task" else c["name"]
                    for c in msg.tool_calls
                )
                line = f"[{time.monotonic() - started:5.0f}s] {who:14} {calls or (msg.text or '')[:120]}"
                if line != last:
                    callbacks.log(line.replace("\n", " "))
                    last = line
    return final_text
