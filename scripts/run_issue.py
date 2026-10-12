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
    agent = build_orchestrator(ws, repo_map, skill, ctx, index, open_pr=args.open_pr)
    started, last, final_text = time.monotonic(), None, ""
    # Watchdog: if no agent event arrives for this long, dump every thread's stack to stderr.
    faulthandler.dump_traceback_later(args.watchdog, exit=False)
    for namespace, update in agent.stream(
        {"messages": [{"role": "user", "content": f"Resolve issue #{issue.number}. Start with /working/task.md."}]},
        config={"recursion_limit": args.recursion_limit},
        stream_mode="updates",
        subgraphs=True,
    ):
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(args.watchdog, exit=False)
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
    faulthandler.cancel_dump_traceback_later()
    verdicts = [line.strip(" *`") for line in final_text.splitlines() if "VERDICT:" in line]
    print(f"\n{verdicts[-1] if verdicts else 'NO VERDICT LINE from the Orchestrator'}")
    print(f"Done in {time.monotonic() - started:.0f}s. Shell commands blocked: {len(ws.shell.blocked)}. Model retries: {len(RETRY_LOG)}")
    for command, reason in ws.shell.blocked:
        print(f"  blocked  {command!r}: {reason}")
    for role, error in RETRY_LOG:
        print(f"  retry  {role}: {error}")
    for name in ("relevant_files.md", "test_report.md", "proposed_diff.txt", "pr_plan.md", "pr_result.md", "pr.json"):
        print(f"  /working/{name}: {'present' if (ws.working / name).exists() else 'MISSING'}")


if __name__ == "__main__":
    main()
