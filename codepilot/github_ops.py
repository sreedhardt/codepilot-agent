"""GitHub operations for the PR Agent: branch, single commit, pull request.

Formatting is deterministic so every PR has the same structure. The commit is
built with the Git Data API (blobs -> tree -> commit -> ref), which lands all
sandbox changes as one commit without giving the sandbox push credentials.
"""

import base64
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from github import Auth, GithubIntegration, InputGitTreeElement
from github.Repository import Repository

from codepilot.classifier import TaskType

PR_LABELS = ["codepilot-generated", "needs-review"]
COMMIT_TYPES = {
    TaskType.BUG_FIX: "fix",
    TaskType.FEATURE_ADDITION: "feat",
    TaskType.DEPENDENCY_UPDATE: "build",
    TaskType.DOCUMENTATION: "docs",
    TaskType.CONFIG_CHANGE: "chore",
}


@dataclass
class Change:
    path: str
    deleted: bool = False


@dataclass
class PullRequestPlan:
    """Everything needed to open the PR, computed before touching GitHub."""

    branch: str
    commit_message: str
    title: str
    body: str
    base_sha: str
    changes: list[Change] = field(default_factory=list)


class MergeConflictError(RuntimeError):
    """The base branch changed the same files since the sandbox was copied."""


def slugify(title: str, max_length: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    if len(slug) > max_length:
        slug = slug[:max_length].rsplit("-", 1)[0]
    return slug or "change"


def branch_name(issue_number: int, title: str) -> str:
    return f"codepilot/issue-{issue_number}-{slugify(title)}"


LEADING_TYPE = re.compile(r"^(fix|feat|feature|docs|build|chore|refactor|test)(\([^)]*\))?\s*:\s*", re.IGNORECASE)


def commit_message(task_type: TaskType, issue_number: int, summary: str, what: list[str], why: list[str]) -> str:
    summary = LEADING_TYPE.sub("", clean(summary)).rstrip(".")
    summary = summary[:1].upper() + summary[1:]
    bullets = [f"- {clean(line)}" for line in [*what, *why] if clean(line)]
    return f"{COMMIT_TYPES[task_type]}(#{issue_number}): {summary}\n\n" + "\n".join([*bullets, f"- Closes #{issue_number}"])


def clean(text: str) -> str:
    """Strip whitespace, a stray wrapping quote pair, and a dangling trailing quote models sometimes add."""
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'`":
        return text[1:-1].strip()
    if text.endswith('"') and text.count('"') % 2 == 1:
        return text[:-1].rstrip()
    return text


def pr_body(issue_number: int, issue_title: str, issue_url: str, summary: str, approach: str,
            changes: list[Change], test_output: str) -> str:
    files = "\n".join(f"- `{c.path}`" + (" (deleted)" if c.deleted else "") for c in changes)
    return (
        f"## Issue\n\n**#{issue_number} {issue_title}**\n\n{clean(summary)}\n\nCloses #{issue_number} ({issue_url})\n\n"
        f"## Approach\n\n{clean(approach)}\n\n"
        f"## Files changed\n\n{files}\n\n"
        f"## Test results\n\n```\n{test_output.strip()}\n```\n\n"
        "---\nOpened by CodePilot. A human must review before merging."
    )


def sandbox_changes(sandbox: Path) -> list[Change]:
    """Modified, added and deleted files in the sandbox working tree, relative to its HEAD."""
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=sandbox,
                         capture_output=True, text=True, check=True).stdout
    changes = []
    for line in out.splitlines():
        status, path = line[:2], line[3:]
        if " -> " in path:  # rename: old path deleted, new path added
            old, path = path.split(" -> ")
            changes.append(Change(old, deleted=True))
        changes.append(Change(path, deleted="D" in status))
    return changes


def sandbox_head(sandbox: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=sandbox, capture_output=True, text=True,
                          check=True).stdout.strip()


def target_repository() -> Repository:
    """The target repository, authenticated as the GitHub App installation."""
    key = os.environ["GITHUB_APP_PRIVATE_KEY"]
    key = Path(key).read_text() if os.path.isfile(key) else key.replace("\\n", "\n")
    owner, name = os.environ["GITHUB_REPOSITORY"].split("/")
    integration = GithubIntegration(auth=Auth.AppAuth(int(os.environ["GITHUB_APP_ID"]), key))
    installation = integration.get_repo_installation(owner, name)
    return integration.get_github_for_installation(installation.id).get_repo(f"{owner}/{name}")


def check_conflicts(repo: Repository, base_sha: str, changes: list[Change]) -> None:
    """Refuse to open a PR when the base branch has since changed any file we changed."""
    head = repo.get_branch(repo.default_branch).commit.sha
    if head == base_sha:
        return
    upstream = {f.filename for f in repo.compare(base_sha, head).files}
    overlap = sorted(upstream & {c.path for c in changes})
    if overlap:
        raise MergeConflictError(
            f"{repo.default_branch} moved from {base_sha[:7]} to {head[:7]} and also changed {overlap}"
        )


def free_branch_name(repo: Repository, branch: str) -> str:
    existing = {b.name for b in repo.get_branches()}
    candidate, n = branch, 2
    while candidate in existing:
        candidate, n = f"{branch}-{n}", n + 1
    return candidate


def open_pull_request(repo: Repository, sandbox: Path, plan: PullRequestPlan, reviewer: str | None) -> str:
    """Create the branch, one commit and the PR. Returns the PR URL."""
    check_conflicts(repo, plan.base_sha, plan.changes)
    base_commit = repo.get_git_commit(plan.base_sha)

    elements = []
    for change in plan.changes:
        if change.deleted:
            elements.append(InputGitTreeElement(change.path, "100644", "blob", sha=None))
            continue
        file = sandbox / change.path
        blob = repo.create_git_blob(base64.b64encode(file.read_bytes()).decode(), "base64")
        mode = "100755" if os.access(file, os.X_OK) else "100644"
        elements.append(InputGitTreeElement(change.path, mode, "blob", sha=blob.sha))
    tree = repo.create_git_tree(elements, base_tree=base_commit.tree)
    commit = repo.create_git_commit(plan.commit_message, tree, [base_commit])

    branch = free_branch_name(repo, plan.branch)
    repo.create_git_ref(f"refs/heads/{branch}", commit.sha)
    pr = repo.create_pull(base=repo.default_branch, head=branch, title=plan.title, body=plan.body)
    pr.add_to_labels(*PR_LABELS)
    if reviewer:
        try:
            pr.create_review_request(reviewers=[reviewer])
        except Exception:  # noqa: BLE001 - e.g. reviewer lacks access; the PR itself is still valid
            pr.create_issue_comment(f"CodePilot could not request a review from @{reviewer}; please review.")
    return pr.html_url

