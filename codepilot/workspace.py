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

from codepilot.sandbox import GuardedShellBackend

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"


def target_repo() -> Path:
    return Path(os.getenv("CODEPILOT_TARGET_PATH", ROOT.parent / "bookshelf-api")).resolve()


def target_python_bin() -> Path:
    default = target_repo() / ".venv" / "bin"
    return Path(os.getenv("CODEPILOT_TARGET_PYTHON_BIN", default)).resolve()


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
        self.shell = GuardedShellBackend(
            self.sandbox,
            env={"PATH": f"{target_python_bin()}:/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
        )

    def _shared_routes(self) -> dict:
        return {
            "/working/": FilesystemBackend(root_dir=self.working, virtual_mode=True),
            "/skills/": FilesystemBackend(root_dir=self.skills, virtual_mode=True),
        }

    def orchestrator_backend(self) -> CompositeBackend:
        """Scratch space plus shared working memory; no repository access, no shell."""
        return CompositeBackend(FilesystemBackend(root_dir=self.scratch, virtual_mode=True), self._shared_routes())

    def explorer_backend(self) -> CompositeBackend:
        """The repository (made read-only by permissions) plus shared working memory; no shell."""
        return CompositeBackend(FilesystemBackend(root_dir=self.sandbox, virtual_mode=True), self._shared_routes())

    def coder_backend(self) -> CompositeBackend:
        """The repository with the guarded shell, plus shared working memory."""
        return CompositeBackend(self.shell, self._shared_routes())

    def tester_backend(self) -> CompositeBackend:
        """Same sandbox and shell as the Coder, but only test files are writable."""
        return CompositeBackend(self.shell.restricted(["/tests/**"]), self._shared_routes())

    def run_tests(self) -> tuple[bool, str]:
        """Run the full suite in the sandbox (trusted path, not an agent command). Returns (passed, output tail)."""
        result = subprocess.run(
            ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--disable-warnings"], cwd=self.sandbox,
            env={"PATH": f"{target_python_bin()}:/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True, text=True, timeout=600,
        )
        tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-15:])
        return result.returncode == 0, tail

    def write(self, name: str, content: str) -> str:
        (self.working / name).write_text(content)
        return f"/working/{name}"
