"""Command line: run one issue, or poll for issues.

    uv run python -m codepilot run 2            # dry run: everything except opening the PR
    uv run python -m codepilot run 2 --open-pr  # open a real PR (asks for approval first)
    uv run python -m codepilot poll --once      # one polling cycle over ai-assignable issues
    uv run python -m codepilot poll             # keep polling every CODEPILOT_POLL_MINUTES
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from codepilot.workspace import ROOT

load_dotenv(ROOT / ".env")

from codepilot.poller import PollerConfig, poll_forever  # noqa: E402
import os  # noqa: E402

from codepilot.memory.episodic import EpisodicMemory, open_store  # noqa: E402
from codepilot.memory.semantic import SemanticMemory  # noqa: E402
from codepilot.runner import REGISTRY_PATH, Callbacks, Memory, RunOptions, RunResult, run_issue  # noqa: E402
from codepilot.tasks import TaskRecord, TaskRegistry  # noqa: E402
from codepilot.workspace import TaskWorkspace  # noqa: E402


MEMORY_DIR = ROOT / ".codepilot_memory"


def open_memory(repo) -> Memory:
    """Open the persistent memory tiers, start a session, and refresh lesson status from GitHub."""
    name = os.environ["GITHUB_REPOSITORY"]
    memory = Memory(EpisodicMemory(open_store(MEMORY_DIR / "episodes.sqlite"), name),
                    SemanticMemory(MEMORY_DIR / "lessons", name))
    memory.episodic.start_session()

    def pr_state(number: int) -> str:
        pr = repo.get_pull(number)
        return "merged" if pr.merged else pr.state

    for lesson_id, status in memory.semantic.promote(pr_state).items():
        print(f"memory: lesson {lesson_id} -> {status}")
    return memory


def learn_from(workspace: Path) -> None:
    from codepilot.classifier import PROVIDER_KWARGS
    from codepilot.memory.semantic import extract_lesson
    from codepilot.models import Role, build_model, model_spec

    record = json.loads((workspace / "task.json").read_text())
    if not record.get("pr_url"):
        sys.exit(f"{workspace} has no opened PR; lessons are only learned from PRs.")
    working = workspace / "working"
    read = lambda name: (working / name).read_text() if (working / name).exists() else ""  # noqa: E731
    provider = model_spec(Role.SUMMARIZER).partition(":")[0]
    model = build_model(Role.SUMMARIZER, **PROVIDER_KWARGS.get(provider, {}))
    lesson = extract_lesson(model, read("task.md"), read("proposed_diff.txt"), read("test_report.md"),
                            structured_method="json_schema" if provider == "groq" else None)
    memory = SemanticMemory(MEMORY_DIR / "lessons", os.environ["GITHUB_REPOSITORY"])
    pr_number = int(record["pr_url"].rstrip("/").rsplit("/", 1)[1])
    lesson_id = memory.add_draft(record["issue_number"], record["title"], record["task_type"], lesson, pr_number)
    print(f"saved draft lesson {lesson_id} (PR #{pr_number})\n{lesson.model_dump_json(indent=1)}")


def run_tui(args, options: RunOptions, registry: TaskRegistry, repo, memory: Memory) -> None:
    from dataclasses import replace

    from codepilot.tui import CodePilotApp

    config = PollerConfig()
    if args.poll_minutes is not None:
        config.interval_minutes = args.poll_minutes

    def run_one(issue, callbacks: Callbacks, open_pr: bool) -> RunResult:
        return run_issue(issue, replace(options, open_pr=open_pr, watchdog_seconds=None), callbacks, registry,
                         memory=memory)

    app = CodePilotApp(lambda: list(repo.get_issues(state="open")), run_one, registry, config, open_pr=args.open_pr)
    try:
        app.run()
    finally:
        summary = memory.episodic.end_session()
        print(f"memory: session saved ({len(summary['tasks'])} task(s))")


def target_repo():
    from langchain_community.utilities.github import GitHubAPIWrapper

    return GitHubAPIWrapper().github_repo_instance


def console_callbacks(args) -> Callbacks:
    def state(record: TaskRecord) -> None:
        print(f"  STATE #{record.issue_number}: {record.state}", flush=True)

    def approve(request: dict, ws: TaskWorkspace) -> dict:
        print(f"\n  APPROVAL NEEDED [{', '.join(request['gates'])}]\n  {request['summary']}")
        if args.approve_all or args.reject_all:
            print(f"  -> {'approved' if args.approve_all else 'rejected'} (unattended run)")
            return {"approved": bool(args.approve_all), "note": "unattended run"}
        while True:
            choice = input("  approve / reject / inspect > ").strip().lower()
            if choice in ("a", "approve"):
                return {"approved": True}
            if choice in ("r", "reject"):
                return {"approved": False, "note": input("  reason (optional) > ").strip()}
            if choice in ("i", "inspect"):
                for key, value in request["details"].items():
                    if isinstance(value, str) and value.startswith("/working/"):
                        path = ws.working / value.removeprefix("/working/")
                        print(f"\n----- {value} -----\n{path.read_text()[:4000] if path.exists() else '(not written)'}")
                    else:
                        print(f"  {key}: {value}")

    return Callbacks(log=lambda line: print(line, flush=True), state=state, approve=approve)


def print_summary(result: RunResult) -> None:
    r = result.record
    print(f"\n{result.verdict}")
    print(f"#{r.issue_number} {r.state} in {result.seconds:.0f}s | path: {' -> '.join(s for s, _ in r.history)}")
    print(f"Approvals: {result.approvals} | blocked actions: {len(result.blocked)} | model retries: {len(result.retries)}"
          f" | fix rounds after a failed test run: {r.retries_after_fail}")
    for b in result.blocked:
        print(f"  blocked  [{b.agent}] {b.action!r}: {b.reason}")
    for role, error in result.retries:
        print(f"  retry    {role}: {error}")
    if r.pr_url:
        print(f"PR: {r.pr_url}")
    if result.lesson:
        print(f"Lesson: {result.lesson} (draft until the PR is merged)")
    print(f"Workspace: {r.workspace}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="codepilot")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Resolve one issue")
    run.add_argument("issue", type=int)
    poll = sub.add_parser("poll", help="Poll for eligible issues and resolve them one at a time")
    poll.add_argument("--once", action="store_true", help="Run a single polling cycle")
    tui = sub.add_parser("tui", help="Terminal UI: issue feed, active task, agent logs and approvals")
    tui.add_argument("--poll-minutes", type=float, default=None, help="Override CODEPILOT_POLL_MINUTES")
    learn = sub.add_parser("learn", help="Extract a draft lesson from a finished task workspace with an opened PR")
    learn.add_argument("workspace", type=Path, help="e.g. runs/issue-2-20261014-093359")
    for p in (run, poll, tui):
        p.add_argument("--open-pr", action="store_true", help="Open real PRs (default: dry run)")
        p.add_argument("--recursion-limit", type=int, default=150)
        p.add_argument("--watchdog", type=int, default=90, help="Seconds without progress before dumping stacks")
        group = p.add_mutually_exclusive_group()
        group.add_argument("--approve-all", action="store_true", help="Approve every gate without asking")
        group.add_argument("--reject-all", action="store_true", help="Reject every gate without asking")
    args = parser.parse_args()

    if args.command == "learn":
        return learn_from(args.workspace)

    options = RunOptions(open_pr=args.open_pr, recursion_limit=args.recursion_limit, watchdog_seconds=args.watchdog)
    callbacks, registry, repo = console_callbacks(args), TaskRegistry(REGISTRY_PATH), target_repo()
    memory = open_memory(repo)

    def run_one(issue) -> None:
        print_summary(run_issue(issue, options, callbacks, registry, memory=memory))

    if args.command == "tui":
        return run_tui(args, options, registry, repo, memory)

    try:
        if args.command == "run":
            run_one(repo.get_issue(args.issue))
        else:
            poll_forever(lambda: repo.get_issues(state="open"), run_one, registry, PollerConfig(), once=args.once)
    finally:
        summary = memory.episodic.end_session()
        print(f"\nmemory: session saved ({len(summary['tasks'])} task(s))")


if __name__ == "__main__":
    sys.exit(main())
