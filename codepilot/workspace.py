"""Per-task workspace and the filesystem view each agent gets.

    runs/issue-<n>-<timestamp>/
      sandbox/   copy of the target repo; the only place code is edited or run
      working/   shared working memory: task, repo map, relevant files, diff, test reports
      skills/    the single skill selected for this task
      scratch/   the Orchestrator's private scratch space

Every agent sees `/working/` and `/skills/` at the same paths, so they can hand
each other file paths instead of file contents. Only the Coder and Test agent
get the guarded shell; the Orchestrator never sees the repository at all.
"""

import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from deepagents.backends import CompositeBackend, FilesystemBackend

from codepilot.gates import Gate, request_approval
from codepilot.sandbox import GuardedShellBackend, ProtectedFilesBackend

ROOT = Path(__file__).resolve().parents[1]
# Working-memory files written only by CodePilot's code; agents may read them but never change them.
RECORD_FILES = ["/task.md", "/repo_map.md", "/proposed_diff.txt", "/pr_plan.md", "/pr_result.md", "/pr.json",
                "/approvals.jsonl", "/working_memory.json", "/working_memory.final.json", "/recent_sessions.md",
                "/lessons.md", "/review.json", "/review.md"]
RUNS_DIR = ROOT / "runs"


def target_repo() -> Path:
    return Path(os.getenv("CODEPILOT_TARGET_PATH", ROOT.parent / "bookshelf-api")).resolve()


def target_python_bin() -> Path:
    default = target_repo() / ".venv" / "bin"
    return Path(os.getenv("CODEPILOT_TARGET_PYTHON_BIN", default)).resolve()


def require_target_python() -> None:
    """Fail before any work if the target's interpreter is missing.

    Agent shells put this directory first on PATH; without it, `python` would resolve to a system Python
    that lacks the target's dependencies, and every test verdict would be about the wrong environment.
    """
    python = target_python_bin() / "python"
    if not python.exists():
        raise FileNotFoundError(f"no Python interpreter at {python}: create the target repo's .venv or set "
                                "CODEPILOT_TARGET_PYTHON_BIN to its bin directory")


class TaskWorkspace:
    def __init__(self, issue_number: int, runs_dir: Path = RUNS_DIR, source: Path | None = None) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.root = runs_dir / f"issue-{issue_number}-{stamp}"
        self.sandbox = self.root / "sandbox"
        self.working = self.root / "working"
        self.skills = self.root / "skills"
        self.scratch = self.root / "scratch"
        for d in (self.working, self.skills, self.scratch):
            d.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            source or target_repo(), self.sandbox,
            ignore=shutil.ignore_patterns(".venv", "node_modules", ".pytest_cache", "__pycache__", ".env"),
        )
        self.approvals = self.working / "approvals.jsonl"
        self.shell = GuardedShellBackend(
            self.sandbox,
            env={"PATH": f"{target_python_bin()}:/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
            push_approver=self._approve_push,
            agent="coder",
            can_escalate=True,
        )

    def _approve_push(self, command: str) -> bool:
        return request_approval([Gate.GIT_PUSH], f"An agent wants to run `{command}` (network operation).",
                                {"command": command}, self.approvals).approved

    def _shared_routes(self, agent: str) -> dict:
        return {
            "/working/": ProtectedFilesBackend(self.working, RECORD_FILES, self.shell.blocked, agent=agent),
            "/skills/": ProtectedFilesBackend(self.skills, ["/**"], self.shell.blocked,
                                              reason="skills are read-only for agents", agent=agent),
        }

    def orchestrator_backend(self, agent: str = "orchestrator") -> CompositeBackend:
        """Scratch space plus shared working memory; no repository access, no shell."""
        return CompositeBackend(FilesystemBackend(root_dir=self.scratch, virtual_mode=True), self._shared_routes(agent))

    def explorer_backend(self) -> CompositeBackend:
        """The repository (made read-only by permissions) plus shared working memory; no shell."""
        return CompositeBackend(FilesystemBackend(root_dir=self.sandbox, virtual_mode=True), self._shared_routes("repo_explorer"))

    def coder_backend(self) -> CompositeBackend:
        """The repository with the guarded shell, plus shared working memory."""
        return CompositeBackend(self.shell, self._shared_routes("coder"))

    def tester_backend(self) -> CompositeBackend:
        """Same sandbox and shell as the Coder, but only test files are writable."""
        return CompositeBackend(self.shell.restricted(["/tests/**"], agent="test_agent"), self._shared_routes("test_agent"))

    def run_tests(self) -> tuple[bool, str]:
        """Run the full suite in the sandbox (trusted path, not an agent command). Returns (passed, output tail)."""
        try:
            # An absolute interpreter path: a bare "python" would fall through to a system Python on PATH when
            # the target venv is missing, and judge the change with the wrong interpreter.
            result = subprocess.run(
                [str(target_python_bin() / "python"), "-m", "pytest", "-q", "-p", "no:cacheprovider", "--disable-warnings"], cwd=self.sandbox,
                env={"PATH": f"{target_python_bin()}:/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True, text=True, timeout=600,
            )
        except (OSError, subprocess.TimeoutExpired) as e:  # e.g. no interpreter at CODEPILOT_TARGET_PYTHON_BIN
            return False, f"could not run the test suite: {type(e).__name__}: {e}"
        tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-15:])
        return result.returncode == 0, tail

    def diff(self) -> str:
        """Unified diff of every sandbox change against HEAD, including new files."""
        diff = subprocess.run(["git", "diff"], cwd=self.sandbox, capture_output=True, text=True).stdout
        untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], cwd=self.sandbox,
                                   capture_output=True, text=True).stdout.split()
        for path in untracked:
            diff += subprocess.run(["git", "diff", "--no-index", "/dev/null", path], cwd=self.sandbox,
                                   capture_output=True, text=True).stdout
        return diff

    def write(self, name: str, content: str) -> str:
        (self.working / name).write_text(content)
        return f"/working/{name}"
