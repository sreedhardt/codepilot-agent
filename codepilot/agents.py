"""The agent team for one task: Orchestrator -> Repo Explorer, Coder -> Test Agent.

The Orchestrator is the root deep agent and spawns its subagents through the
DeepAgents `task` tool. The Repo Explorer and Coder are compiled deep agents
with their own filesystem views (see `workspace.py`); the Test Agent is a
subagent of the Coder and shares its sandbox. Agents pass file paths through
`/working/`, never file contents.
"""

import logging
import os
import json
import subprocess
from dataclasses import dataclass
from typing import Literal

from deepagents import CompiledSubAgent, create_deep_agent
from deepagents.middleware.permissions import FilesystemPermission
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    TodoListMiddleware,
)
from langchain.agents.middleware.model_retry import default_retry_on
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.checkpoint.base import BaseCheckpointSaver

from codepilot import github_ops
from codepilot.classifier import TaskType
from codepilot.gates import LARGE_COMMIT_FILES, Gate, RetryGateMiddleware, request_approval
from codepilot.models import Role, build_model, model_spec
from codepilot.repo_map import EmbeddingIndex, RepoMap
from codepilot.skills import Skill
from codepilot.workspace import TaskWorkspace

logger = logging.getLogger(__name__)

# Streaming makes the timeout apply between chunks: a long generation keeps the connection alive, while a
# stalled connection fails after the timeout and is retried instead of blocking for the SDK's 10-minute default.
PROVIDER_KWARGS = {
    "anthropic": {"max_tokens": 16000, "streaming": True, "timeout": 60},
    "groq": {"timeout": 60},
    "google_genai": {},
    "ollama": {},
}
MODEL_CALL_LIMITS = {Role.ORCHESTRATOR: 25, Role.REPO_EXPLORER: 15, Role.CODER: 40, Role.TESTER: 20, Role.PR: 8}


def _model(role: Role) -> BaseChatModel:
    provider = model_spec(role).partition(":")[0]
    return build_model(role, max_retries=3, **PROVIDER_KWARGS.get(provider, {}))


RETRY_LOG: list[tuple[str, str]] = []


def _retry_on(role: Role):
    """Retry any model failure (stalled streams, timeouts, rejected tool calls), recording each one."""
    def should_retry(exc: Exception) -> bool:
        RETRY_LOG.append((role.value, f"{type(exc).__name__}: {str(exc)[:120]}"))
        logger.warning("model call failed for %s, retrying: %s", role.value, type(exc).__name__)
        return default_retry_on(exc)
    return should_retry


def _middleware(role: Role) -> list:
    """Retries for every agent, a fallback model for Groq roles, and a per-run model-call cap as a cost guard.

    Middleware order is outermost first: the fallback only engages after retries on the primary are exhausted.
    """
    middleware: list = []
    if model_spec(role).startswith("groq:"):
        fallback = os.getenv("CODEPILOT_FALLBACK_MODEL", "anthropic:claude-haiku-4-5")
        provider, _, name = fallback.partition(":")
        if provider != "anthropic" or os.getenv("ANTHROPIC_API_KEY"):
            kwargs = PROVIDER_KWARGS.get(provider, {})
            middleware.append(ModelFallbackMiddleware(init_chat_model(name, model_provider=provider, **kwargs)))
    middleware.append(ModelRetryMiddleware(max_retries=2, retry_on=_retry_on(role), on_failure="error", initial_delay=1.0))
    middleware.append(ModelCallLimitMiddleware(run_limit=MODEL_CALL_LIMITS.get(role, 20), exit_behavior="end"))
    return middleware


ORCHESTRATOR_PROMPT = """You are the Orchestrator of CodePilot, a team of agents that resolves one GitHub issue.
You plan and delegate. You never read repository files, edit code or run commands yourself.

Start by reading /working/task.md (the issue, its task type and the selected skill) and /working/repo_map.md.

1. Use write_todos to turn the issue into an implementation checklist that follows the skill's workflow steps.
2. Delegate to `repo_explorer` with the task tool: describe what to look for. It writes /working/relevant_files.md.
3. Delegate to `coder` with the task tool. Your instruction must contain the issue summary, the skill name, and
   the path /working/relevant_files.md. Pass file paths only; never paste file contents into a delegation.
4. When the coder reports back, read /working/test_report.md and /working/proposed_diff.txt.
5. Only if the test report says PASS and the diff is not empty, delegate to `pr_agent` with the task tool to open
   the pull request. Do not ask it to resolve conflicts or retry if it reports FAILED.
6. Read /working/pr_result.md: it is the authoritative outcome; never describe a PR as opened unless its first
   line starts with PR_OPENED. End with one plain-text line, no markdown:
   `VERDICT: RESOLVED - <first line of pr_result.md>` when it starts with PR_OPENED or DRY RUN, otherwise
   `VERDICT: FAILED - <first line of pr_result.md, or your reason>`.
Mark todos complete as you go."""

EXPLORER_PROMPT = """You are the Repo Explorer. Find the repository files relevant to a task, without editing anything.

- Read /working/repo_map.md first: it lists every file with its symbols and a description.
- Use find_relevant_files: strategy "embedding" for behaviour described in prose, "keyword" for exact names.
- Use grep to find call sites of specific functions or APIs, and read_file to confirm relevance.
- Write /working/relevant_files.md: at most 10 repository paths, most relevant first, each with a one-line reason.
  Include the test files that cover the relevant code.
- Reply with the same list. Do not include file contents in your reply."""

CODER_PROMPT = """You are the Coder in CodePilot. You implement one change in a Python repository.

Your file tools see the repository at `/`, shared working memory at `/working/`, and your skill at `/skills/`.
Shell commands already run in the repository root: use relative paths and never `cd`.

1. Read the skill in /skills/ and follow its workflow steps and forbidden actions.
2. Read /working/relevant_files.md, then load files on demand with read_file. Read before editing.
3. Write your change plan with write_todos.
4. Implement with edit_file, making surgical edits. Never rewrite whole files.
5. Run the code you touched with `python -m pytest -q` to make sure it does not crash.
6. Delegate verification to `test_agent` with the task tool. If it reports FAIL, fix the cause and delegate again.
   Stop after 3 failed verification rounds and report what is wrong.
7. Call write_diff_preview, then reply with: root cause, files changed, tests added, verification result.

Shell commands pass through a guardrail: one plain command per call, no pipes, chaining or redirection.
Allowed: python -m pytest, ls, cat, grep, find, head, tail, wc, git status/diff/log/show. No installs.
If a command is blocked, explain what you wanted to do and why instead of retrying variations."""

TESTER_PROMPT = """You are the Test Agent. You verify a change the Coder made, independently.

1. Run `git diff` to see what changed.
2. Check the tests cover the change: the behaviour the issue asks for, its boundaries, and its error paths.
   Add missing tests to the existing test files under /tests/. Do not edit non-test files.
3. Run the full suite with `python -m pytest -q`.
4. Write /working/test_report.md: the command, pass/fail counts, and for each failure the test name and a
   one-line cause taken from the output.
5. Reply starting with `PASS` or `FAIL`, then the counts and failures."""


PR_PROMPT = """You are the PR Agent. You open the pull request for a verified change; you do not change code.

1. Read /working/task.md, /working/test_report.md and /working/proposed_diff.txt.
2. Call open_pull_request exactly once with:
   - summary: one imperative line under 70 characters describing what the change does
   - what_changed: 1-3 short bullets on what changed, naming files or functions
   - why: 1-2 short bullets on why; do not repeat the issue number (CodePilot adds "Closes #n")
   - approach: 2-4 sentences on the root cause and how the change addresses it
3. Reply with the tool's result verbatim and nothing else. If it starts with FAILED, do not call it again."""


@dataclass
class TaskContext:
    """Issue metadata the PR Agent needs; filled by the driver, never by an agent."""

    issue_number: int
    title: str
    url: str
    reporter: str | None
    task_type: TaskType


def make_pr_tool(ws: TaskWorkspace, ctx: TaskContext, open_pr: bool):
    @tool
    def open_pull_request(summary: str, what_changed: list[str], why: list[str], approach: str) -> str:
        """Commit every sandbox change as one commit on a new branch and open a PR to the default branch.

        The branch name, commit format, PR title, body sections and labels are fixed by CodePilot.
        """
        # Idempotent by construction: a second call (models sometimes repeat tool calls) must never open a second PR.
        previous = ws.working / "pr_result.md"
        if previous.exists():
            return "ALREADY CALLED, nothing done. The recorded outcome is: " + previous.read_text().strip()
        result = _open_pull_request(summary, what_changed, why, approach)
        ws.write("pr_result.md", result + "\n")
        return result

    def _open_pull_request(summary: str, what_changed: list[str], why: list[str], approach: str) -> str:
        changes = github_ops.sandbox_changes(ws.sandbox)
        if not changes:
            return "FAILED: the sandbox has no changes; nothing to open a PR for."
        passed, test_output = ws.run_tests()
        if not passed:
            return f"FAILED: the test suite does not pass, so no PR was opened.\n{test_output}"

        plan = github_ops.PullRequestPlan(
            branch=github_ops.branch_name(ctx.issue_number, ctx.title),
            commit_message=github_ops.commit_message(ctx.task_type, ctx.issue_number, summary, what_changed, why),
            title=f"[CodePilot] {ctx.title}",
            body=github_ops.pr_body(ctx.issue_number, ctx.title, ctx.url, summary, approach, changes, test_output),
            base_sha=github_ops.sandbox_head(ws.sandbox),
            changes=changes,
        )
        ws.write("pr_plan.md", f"# Branch\n{plan.branch}\n\n# Commit message\n{plan.commit_message}\n\n"
                                f"# Title\n{plan.title}\n\n# Labels\n{', '.join(github_ops.PR_LABELS)}\n\n"
                                f"# Body\n{plan.body}\n")
        if not open_pr:
            return "DRY RUN: PR not opened (run with --open-pr). Full preview written to /working/pr_plan.md."

        repo = github_ops.target_repository()
        gates = [Gate.PR_TO_DEFAULT_BRANCH] if repo.default_branch in ("main", "master") else []
        if len(changes) > LARGE_COMMIT_FILES:
            gates.append(Gate.LARGE_COMMIT)
        if gates:
            decision = request_approval(
                gates,
                f"Open PR \"{plan.title}\" from {plan.branch} into {repo.default_branch} ({len(changes)} files)?",
                {"branch": plan.branch, "files": [c.path for c in changes],
                 "preview": "/working/pr_plan.md", "diff": "/working/proposed_diff.txt"},
                ws.approvals,
            )
            if not decision.approved:
                return f"REJECTED by human: PR not opened. {decision.note}".strip()
        try:
            url = github_ops.open_pull_request(repo, ws.sandbox, plan, ctx.reporter)
        except github_ops.MergeConflictError as e:
            return f"FAILED: merge conflict, PR not opened and not resolved automatically: {e}"
        ws.write("pr.json", json.dumps({"url": url, "branch": plan.branch, "title": plan.title}))
        return f"PR_OPENED {url}"

    return open_pull_request


def build_pr_agent(ws: TaskWorkspace, ctx: TaskContext, open_pr: bool) -> CompiledSubAgent:
    agent = create_deep_agent(
        model=_model(Role.PR),
        tools=[make_pr_tool(ws, ctx, open_pr)],
        system_prompt=PR_PROMPT,
        backend=ws.orchestrator_backend(agent="pr_agent"),
        permissions=[FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")],
        middleware=_middleware(Role.PR),
        name="pr_agent",
    )
    return CompiledSubAgent(
        name="pr_agent",
        description="Opens the pull request for a verified change: one structured commit on a codepilot/ branch, "
                    "labelled PR to the default branch. Use only after the Test Agent reports PASS.",
        runnable=agent,
    )


def build_explorer(ws: TaskWorkspace, repo_map: RepoMap, index: EmbeddingIndex | None) -> CompiledSubAgent:
    @tool
    def find_relevant_files(query: str, strategy: Literal["keyword", "embedding"] = "embedding", k: int = 10) -> str:
        """Rank repository files by relevance to the query. Returns paths with their repo-map summary."""
        if strategy == "embedding" and index is not None:
            paths = index.search(query, k)
        else:
            paths = repo_map.keyword_search(query, k)
        lines = []
        for p in paths:
            e = repo_map.files[p]
            lines.append(f"{p} - {e.description or ', '.join(e.symbols[:5]) or e.language}")
        return "\n".join(lines) or "No matches."

    agent = create_deep_agent(
        model=_model(Role.REPO_EXPLORER),
        tools=[find_relevant_files],
        system_prompt=EXPLORER_PROMPT,
        backend=ws.explorer_backend(),
        permissions=[
            FilesystemPermission(operations=["write"], paths=["/working/**"], mode="allow"),
            FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
        ],
        middleware=_middleware(Role.REPO_EXPLORER),
        name="repo_explorer",
    )
    return CompiledSubAgent(
        name="repo_explorer",
        description="Finds the repository files relevant to a task and writes them to /working/relevant_files.md.",
        runnable=agent,
    )


def build_tester(ws: TaskWorkspace) -> CompiledSubAgent:
    # Write rules for the Coder and Test Agent live in their GuardedShellBackend (secrets are protected,
    # the Test Agent may only write tests): DeepAgents rejects declarative permissions on executing backends.
    return CompiledSubAgent(
        name="test_agent",
        description="Independently reviews test coverage for the change, adds missing tests, runs the full suite "
                    "and reports PASS or FAIL with failures.",
        runnable=create_deep_agent(
            model=_model(Role.TESTER),
            system_prompt=TESTER_PROMPT,
            backend=ws.tester_backend(),
            middleware=_middleware(Role.TESTER),
            name="test_agent",
        ),
    )


def build_coder(ws: TaskWorkspace) -> CompiledSubAgent:
    @tool
    def write_diff_preview() -> str:
        """Write a unified diff of all changes to /working/proposed_diff.txt for the Orchestrator to review."""
        diff = subprocess.run(["git", "diff"], cwd=ws.sandbox, capture_output=True, text=True).stdout
        untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], cwd=ws.sandbox,
                                   capture_output=True, text=True).stdout.split()
        for path in untracked:
            diff += subprocess.run(["git", "diff", "--no-index", "/dev/null", path], cwd=ws.sandbox,
                                   capture_output=True, text=True).stdout
        ws.write("proposed_diff.txt", diff or "(no changes)\n")
        stat = subprocess.run(["git", "diff", "--stat"], cwd=ws.sandbox, capture_output=True, text=True).stdout
        return f"Wrote /working/proposed_diff.txt\n{stat}" + (f"New files: {untracked}" if untracked else "")

    agent = create_deep_agent(
        model=_model(Role.CODER),
        tools=[write_diff_preview],
        system_prompt=CODER_PROMPT,
        backend=ws.coder_backend(),
        skills=["/skills/"],
        subagents=[build_tester(ws)],
        middleware=[TodoListMiddleware(), RetryGateMiddleware(ws.approvals), *_middleware(Role.CODER)],
        name="coder",
    )
    return CompiledSubAgent(
        name="coder",
        description="Implements the change in the sandbox following the selected skill, has it verified by the "
                    "Test Agent, and writes /working/proposed_diff.txt.",
        runnable=agent,
    )


def build_team(ws: TaskWorkspace, repo_map: RepoMap, ctx: TaskContext, index: EmbeddingIndex | None = None,
               open_pr: bool = False) -> list[CompiledSubAgent]:
    """The Orchestrator's subagents. The Test Agent is the Coder's own subagent."""
    return [build_explorer(ws, repo_map, index), build_coder(ws), build_pr_agent(ws, ctx, open_pr)]


def build_orchestrator(
    ws: TaskWorkspace,
    repo_map: RepoMap,
    skill: Skill,
    ctx: TaskContext,
    index: EmbeddingIndex | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    open_pr: bool = False,
):
    skill.install(ws.skills)
    return create_deep_agent(
        model=_model(Role.ORCHESTRATOR),
        system_prompt=ORCHESTRATOR_PROMPT,
        backend=ws.orchestrator_backend(),
        subagents=build_team(ws, repo_map, ctx, index, open_pr),
        middleware=[TodoListMiddleware(), *_middleware(Role.ORCHESTRATOR)],
        checkpointer=checkpointer,
        name="orchestrator",
    )
