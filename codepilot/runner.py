"""Run one issue end to end: classify, prepare the workspace, run the agent team, settle the task state.

Front ends (CLI, poller, TUI) supply callbacks for log lines, state changes and
approval decisions; the runner itself never prints or reads input.
"""

import faulthandler
import json
import os
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
from codepilot import github_ops
from codepilot.classifier import PROVIDER_KWARGS, IssueClassifier, build_classifier
from codepilot.costs import UsageTracker, is_billing_error
from codepilot.memory.episodic import EpisodicMemory, TaskEpisode
from codepilot.memory.semantic import SemanticMemory, extract_lesson, render_lessons
from codepilot.memory.working import WorkingMemory
from codepilot.models import Role, build_model, model_spec
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
    recursion_limit: int | None = None  # None: derived from the Orchestrator's model-call cap
    watchdog_seconds: int | None = 90
    runs_dir: Path = RUNS_DIR
    max_cost_usd: float | None = float(os.getenv("CODEPILOT_MAX_TASK_COST_USD", "2.0"))
    max_minutes: float | None = float(os.getenv("CODEPILOT_MAX_TASK_MINUTES", "20"))


@dataclass
class Callbacks:
    log: Callable[[str], None] = lambda line: None
    state: Callable[[TaskRecord], None] = lambda record: None
    approve: Callable[[dict, TaskWorkspace], dict] = lambda request, ws: {"approved": False, "note": "no approver"}
    todos: Callable[[str, list[dict]], None] = lambda agent, todos: None


@dataclass
class Memory:
    """The cross-task memory tiers; working memory is created per task by the runner."""

    episodic: EpisodicMemory | None = None
    semantic: SemanticMemory | None = None


@dataclass
class RunResult:
    record: TaskRecord
    verdict: str
    seconds: float
    workspace: TaskWorkspace
    blocked: list[BlockedAction] = field(default_factory=list)
    retries: list[tuple[str, str]] = field(default_factory=list)
    approvals: int = 0
    lesson: str | None = None
    usage: dict = field(default_factory=dict)


def run_issue(
    issue,
    options: RunOptions | None = None,
    callbacks: Callbacks | None = None,
    registry: TaskRegistry | None = None,
    classifier: IssueClassifier | None = None,
    memory: Memory | None = None,
) -> RunResult:
    """`issue` is a PyGithub Issue (number, title, body, html_url, user)."""
    options, callbacks, memory = options or RunOptions(), callbacks or Callbacks(), memory or Memory()
    registry = registry or TaskRegistry(REGISTRY_PATH)
    started, retries_before = time.monotonic(), len(agents.RETRY_LOG)
    usage = UsageTracker(options.max_cost_usd, None if options.max_minutes is None else options.max_minutes * 60)

    classification = (classifier or build_classifier()).classify(issue.title, issue.body or "")
    skill = skill_for(classification.task_type)
    ws = TaskWorkspace(issue.number, runs_dir=options.runs_dir)
    workspace = ws.root.relative_to(ROOT) if ws.root.is_relative_to(ROOT) else ws.root
    record = TaskRecord(issue.number, issue.title, classification.task_type.value, skill.skill_id, str(workspace))

    working = WorkingMemory(issue.number, issue.title, issue.html_url, classification.task_type.value, skill.skill_id)

    def snapshot() -> str:
        working.refresh(ws, record.state.value, record.retries_after_fail)
        working.save(ws)
        return working.snapshot()

    def on_change(r: TaskRecord) -> None:
        registry.save(r)
        snapshot()
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
            + _previous_attempts(memory, issue.number)
        ))
        _write_memories(memory, ws, issue, callbacks)
        snapshot()
        ctx = TaskContext(issue.number, issue.title, issue.html_url,
                          issue.user.login if issue.user else None, classification.task_type)
        agent = build_orchestrator(ws, repo_map, skill, ctx, index, checkpointer=InMemorySaver(),
                                   open_pr=options.open_pr, snapshot=snapshot)
        limit = options.recursion_limit or agents.step_limit(agent, Role.ORCHESTRATOR)
        config = {"recursion_limit": limit, "configurable": {"thread_id": ws.root.name},
                  "callbacks": [usage]}
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
        record.billing_blocked = is_billing_error(e)
        callbacks.log(f"ERROR {error}\n{traceback.format_exc(limit=5)}")
    finally:
        if options.watchdog_seconds:
            faulthandler.cancel_dump_traceback_later()

    record.cost_usd = round(usage.total_cost, 4)
    tracker.finish(ws.working, error)
    snapshot()
    working.archive(ws)
    lesson = _learn(memory, ws, issue, record, callbacks)
    if memory.episodic is not None:
        files = [c.path for c in github_ops.sandbox_changes(ws.sandbox)]
        memory.episodic.record(TaskEpisode(issue.number, issue.title, record.task_type, record.state.value,
                                           record.outcome, files, round(time.monotonic() - started, 1), record.pr_url))
    verdicts = [line.strip(" *`") for line in final_text.splitlines() if "VERDICT:" in line]
    result = RunResult(
        record=record,
        verdict=verdicts[-1] if verdicts else "NO VERDICT LINE from the Orchestrator",
        seconds=time.monotonic() - started,
        workspace=ws,
        blocked=list(ws.shell.blocked),
        retries=agents.RETRY_LOG[retries_before:],
        approvals=ws.approvals.read_text().count("\n") if ws.approvals.exists() else 0,
        lesson=lesson,
        usage=usage.summary(),
    )
    (ws.root / "task.json").write_text(json.dumps(asdict(record), indent=1))
    return result


def _previous_attempts(memory: Memory, issue_number: int) -> str:
    if memory.episodic is None:
        return ""
    attempts = memory.episodic.previous_attempts(issue_number)
    if not attempts:
        return ""
    lines = [f"- {a['state']}: {a['outcome'][:200]}" for a in attempts]
    return "\nEarlier attempts at this issue (recent sessions):\n" + "\n".join(lines) + "\n"


def _write_memories(memory: Memory, ws: TaskWorkspace, issue, callbacks: Callbacks) -> None:
    if memory.episodic is not None:
        ws.write("recent_sessions.md", memory.episodic.render_recent(3))
    if memory.semantic is not None:
        lessons = memory.semantic.search(f"{issue.title}\n{issue.body or ''}")
        ws.write("lessons.md", render_lessons(lessons))
        callbacks.log(f"memory: {len(lessons)} lesson(s) retrieved"
                      + (f" ({', '.join(l['id'] for l in lessons)})" if lessons else ""))


def _learn(memory: Memory, ws: TaskWorkspace, issue, record: TaskRecord, callbacks: Callbacks) -> str | None:
    """After a PR opens, extract a lesson and store it as a draft until the PR is merged."""
    if memory.semantic is None or not record.pr_url:
        return None
    try:
        provider = model_spec(Role.SUMMARIZER).partition(":")[0]
        model = build_model(Role.SUMMARIZER, **PROVIDER_KWARGS.get(provider, {}))
        read = lambda name: (ws.working / name).read_text() if (ws.working / name).exists() else ""  # noqa: E731
        lesson = extract_lesson(model, read("task.md"), read("proposed_diff.txt"), read("test_report.md"),
                                structured_method="json_schema" if provider == "groq" else None)
        pr_number = int(record.pr_url.rstrip("/").rsplit("/", 1)[1])
        lesson_id = memory.semantic.add_draft(issue.number, issue.title, record.task_type, lesson, pr_number)
        callbacks.log(f"memory: saved draft lesson {lesson_id} (promoted when PR #{pr_number} is merged)")
        return lesson_id
    except Exception as e:  # noqa: BLE001 - a failed lesson extraction must not fail a finished task
        callbacks.log(f"memory: lesson extraction failed: {type(e).__name__}: {e}")
        return None


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
            if node_update.get("todos"):
                # write_todos updates arrive without a message name; attribute them to the agent last seen acting.
                callbacks.todos(tracker.record.active_agent or "orchestrator", list(node_update["todos"]))
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
