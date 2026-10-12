"""Run the agent team on one GitHub issue, in a sandbox, without opening a PR.

    uv run python scripts/run_issue.py 2

Prints which agent is active as it streams and leaves the workspace in runs/
for inspection (working memory, diff, test report).
"""

import argparse
import faulthandler
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from langchain_community.utilities.github import GitHubAPIWrapper  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from codepilot.agents import RETRY_LOG, TaskContext, build_orchestrator  # noqa: E402
from codepilot.classifier import build_classifier  # noqa: E402
from codepilot.repo_map import EmbeddingIndex, build_repo_map  # noqa: E402
from codepilot.skills import skill_for  # noqa: E402
from codepilot.workspace import TaskWorkspace  # noqa: E402

CACHE = ROOT / ".codepilot_cache"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("issue", type=int)
    parser.add_argument("--recursion-limit", type=int, default=150)
    parser.add_argument("--open-pr", action="store_true", help="Actually open the PR (default: dry run)")
    approvals = parser.add_mutually_exclusive_group()
    approvals.add_argument("--approve-all", action="store_true", help="Approve every gate without asking")
    approvals.add_argument("--reject-all", action="store_true", help="Reject every gate without asking")
    parser.add_argument("--watchdog", type=int, default=90, help="Seconds without progress before dumping stacks")
    args = parser.parse_args()

    issue = GitHubAPIWrapper().github_repo_instance.get_issue(args.issue)
    classification = build_classifier().classify(issue.title, issue.body or "")
    skill = skill_for(classification.task_type)
    print(f"#{issue.number} {issue.title}\n  -> {classification.task_type} ({classification.confidence:.2f}), {skill.skill_id}")

    ws = TaskWorkspace(issue.number)
    repo_map = build_repo_map(ws.sandbox, cache_dir=CACHE)
    index = EmbeddingIndex(repo_map, CACHE / "chroma")
    ws.write("repo_map.md", repo_map.render())
    ws.write("task.md", (
        f"# Issue #{issue.number}: {issue.title}\n\n{issue.body}\n\n"
        f"Task type: {classification.task_type}\nSkill: {skill.skill_id} (installed at /skills/{skill.name}/)\n"
        f"Skill workflow: {' -> '.join(skill.workflow_steps)}\n"
    ))
    print(f"  workspace: {ws.root.relative_to(ROOT)}\n")

    ctx = TaskContext(issue.number, issue.title, issue.html_url, issue.user.login if issue.user else None,
                      classification.task_type)
    agent = build_orchestrator(ws, repo_map, skill, ctx, index, checkpointer=InMemorySaver(), open_pr=args.open_pr)
    config = {"recursion_limit": args.recursion_limit, "configurable": {"thread_id": ws.root.name}}
    payload = {"messages": [{"role": "user", "content": f"Resolve issue #{issue.number}. Start with /working/task.md."}]}
    started, last, final_text = time.monotonic(), None, ""
    while True:
        # Watchdog: if no agent event arrives for this long, dump every thread's stack to stderr.
        faulthandler.dump_traceback_later(args.watchdog, exit=False)
        last, final_text = stream_run(agent, payload, config, started, last, final_text, args.watchdog)
        faulthandler.cancel_dump_traceback_later()
        pending = agent.get_state(config).interrupts
        if not pending:
            break
        payload = Command(resume={p.id: ask_human(p.value, ws, args) for p in pending})
    report(ws, final_text, started)


def stream_run(agent, payload, config, started, last, final_text, watchdog):
    for namespace, update in agent.stream(payload, config=config, stream_mode="updates", subgraphs=True):
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(watchdog, exit=False)
        for node_update in update.values():
            for msg in (node_update or {}).get("messages", []) if isinstance(node_update, dict) else []:
                if isinstance(msg, AIMessage):
                    who = msg.name or ("orchestrator" if not namespace else namespace[-1].split(":")[0])
                    calls = ", ".join(
                        f"{c['name']}({c['args'].get('subagent_type', '')})" if c["name"] == "task" else c["name"]
                        for c in msg.tool_calls
                    )
                    if who == "orchestrator" and msg.text:
                        final_text = msg.text
                    line = f"[{time.monotonic() - started:5.0f}s] {who:14} {calls or (msg.text or '')[:120]}"
                    if line != last:
                        print(line.replace("\n", " "), flush=True)
                        last = line
    return last, final_text


def ask_human(request: dict, ws: TaskWorkspace, args) -> dict:
    print(f"\n  APPROVAL NEEDED [{', '.join(request['gates'])}]\n  {request['summary']}")
    if args.approve_all or args.reject_all:
        print(f"  -> {'approved' if args.approve_all else 'rejected'} (--{'approve' if args.approve_all else 'reject'}-all)")
        return {"approved": bool(args.approve_all), "note": "unattended run"}
    while True:
        choice = input("  approve / reject / inspect > ").strip().lower()
        if choice in ("a", "approve"):
            return {"approved": True}
        if choice in ("r", "reject"):
            return {"approved": False, "note": input("  reason (optional) > ").strip()}
        if choice in ("i", "inspect"):
            for key, value in request["details"].items():
                path = ws.working / value.removeprefix("/working/") if isinstance(value, str) and value.startswith("/working/") else None
                if path and path.exists():
                    print(f"\n----- {value} -----\n{path.read_text()[:4000]}")
                else:
                    print(f"  {key}: {value}")


def report(ws: TaskWorkspace, final_text: str, started: float) -> None:
    verdicts = [line.strip(" *`") for line in final_text.splitlines() if "VERDICT:" in line]
    print(f"\n{verdicts[-1] if verdicts else 'NO VERDICT LINE from the Orchestrator'}")
    approvals = ws.approvals.read_text().count("\n") if ws.approvals.exists() else 0
    print(f"Done in {time.monotonic() - started:.0f}s. Approvals requested: {approvals}. Shell commands blocked: {len(ws.shell.blocked)}. Model retries: {len(RETRY_LOG)}")
    for command, reason in ws.shell.blocked:
        print(f"  blocked  {command!r}: {reason}")
    for role, error in RETRY_LOG:
        print(f"  retry  {role}: {error}")
    for name in ("relevant_files.md", "test_report.md", "proposed_diff.txt", "pr_plan.md", "pr_result.md", "pr.json"):
        print(f"  /working/{name}: {'present' if (ws.working / name).exists() else 'MISSING'}")


if __name__ == "__main__":
    main()
