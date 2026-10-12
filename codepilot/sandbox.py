"""Command guardrail for the Coder's shell.

DeepAgents' LocalShellBackend runs `execute` commands with `shell=True` and no
restrictions; its `virtual_mode` only confines the *file* tools. This backend
puts an allowlist in front of `execute`:

- one plain command per call: no chaining, pipes, redirection or substitution
- only allowlisted programs (test runner, read-only inspection, read-only git)
- no network tools, package installs, deletion, or paths outside the sandbox

This is a policy layer, not isolation: running a test suite executes code the
agent wrote. True isolation needs a container (see README, Known limitations).
"""

import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from deepagents.backends import LocalShellBackend
from deepagents.backends.protocol import EditResult, ExecuteResponse, WriteResult
from wcmatch.glob import BRACE, DOTGLOB, GLOBSTAR, globmatch

BLOCKED_EXIT_CODE = 126

SHELL_METACHARACTERS = (";", "&", "|", "`", "$(", ">", "<", "\n")
READ_ONLY_PROGRAMS = {"ls", "cat", "head", "tail", "grep", "wc", "find", "pwd"}
GIT_READ_ONLY = {"status", "diff", "log", "show"}
FIND_FORBIDDEN = {"-exec", "-execdir", "-delete", "-ok", "-okdir", "-fprint", "-fls"}


@dataclass
class Verdict:
    allowed: bool
    reason: str = ""


def is_git_push(command: str) -> bool:
    """A single plain `git push ...` command (chained commands are rejected separately)."""
    if any(m in command for m in SHELL_METACHARACTERS):
        return False
    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    return argv[:2] == ["git", "push"]


def check_command(command: str, root: Path) -> Verdict:
    """Decide whether a shell command may run inside the sandbox rooted at `root`."""
    if any(m in command for m in SHELL_METACHARACTERS):
        return Verdict(False, "one plain command per call; no ;, &&, |, redirection or $(...)")
    try:
        argv = shlex.split(command)
    except ValueError as e:
        return Verdict(False, f"could not parse command: {e}")
    if not argv:
        return Verdict(False, "empty command")

    for arg in argv[1:]:
        if arg.startswith("~") or ".." in Path(arg).parts:
            return Verdict(False, f"path escapes the sandbox: {arg}")
        if arg.startswith("/") and not Path(arg).resolve().is_relative_to(root.resolve()):
            return Verdict(False, f"path outside the sandbox: {arg}")

    program, args = argv[0], argv[1:]
    if program in {"python", "python3"}:
        if args[:2] == ["-m", "pytest"]:
            return Verdict(True)
        return Verdict(False, "python may only run the test suite: use `python -m pytest ...`")
    if program == "pytest":
        return Verdict(True)
    if program == "git":
        if args and args[0] in GIT_READ_ONLY:
            return Verdict(True)
        return Verdict(False, f"git is read-only here: allowed subcommands are {sorted(GIT_READ_ONLY)}")
    if program == "find" and FIND_FORBIDDEN.intersection(args):
        return Verdict(False, "find may not execute or delete")
    if program in READ_ONLY_PROGRAMS:
        return Verdict(True)
    return Verdict(False, f"`{program}` is not allowed (network, installs and deletion are blocked)")


PROTECTED_FILES = ["/**/.env", "/**/.env.*", "/**/*.secret", "/**/*.pem", "/**/*.key", "/**/*credentials*"]


def _matches(path: str, patterns: list[str]) -> bool:
    path = "/" + path.lstrip("/")
    return globmatch(path, patterns, flags=GLOBSTAR | BRACE | DOTGLOB)


class GuardedShellBackend(LocalShellBackend):
    """LocalShellBackend whose `execute` only runs commands that pass `check_command`.

    It also enforces a write policy on `write_file`/`edit_file`, because DeepAgents
    does not support declarative permissions on backends that can execute commands:
    protected files (secrets) are never writable, and `writable`, when given,
    limits writes to matching paths.

    Blocked attempts are recorded in `self.blocked` as (command or path, reason) pairs.
    """

    def __init__(
        self,
        root_dir: str | Path,
        *,
        env: dict[str, str] | None = None,
        timeout: int = 120,
        writable: list[str] | None = None,
        protected: list[str] = PROTECTED_FILES,
        blocked: list[tuple[str, str]] | None = None,
        push_approver: Callable[[str], bool] | None = None,
    ) -> None:
        super().__init__(root_dir=root_dir, virtual_mode=True, env=env, inherit_env=False, timeout=timeout)
        self.root = Path(root_dir)
        self.writable = writable
        self.protected = protected
        self.blocked = blocked if blocked is not None else []
        self.push_approver = push_approver

    def restricted(self, writable: list[str]) -> "GuardedShellBackend":
        """Same sandbox and audit log, with writes limited to `writable` paths."""
        return GuardedShellBackend(self.root, env=self._env, timeout=self._default_timeout, writable=writable,
                                   protected=self.protected, blocked=self.blocked, push_approver=self.push_approver)

    def _write_denied(self, file_path: str) -> str | None:
        if _matches(file_path, self.protected):
            reason = "protected file: secrets and credentials are never edited"
        elif self.writable is not None and not _matches(file_path, self.writable):
            reason = f"outside this agent's writable paths {self.writable}"
        else:
            return None
        self.blocked.append((f"write {file_path}", reason))
        return f"BLOCKED by CodePilot guardrail: {file_path}: {reason}. File not changed."

    def write(self, file_path: str, content: str) -> WriteResult:
        if error := self._write_denied(file_path):
            return WriteResult(error=error)
        return super().write(file_path, content)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:  # noqa: FBT001, FBT002
        if error := self._write_denied(file_path):
            return EditResult(error=error)
        return super().edit(file_path, old_string, new_string, replace_all)

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        if is_git_push(command):
            # Network operation: never allowed by policy alone, only with a human's approval.
            if self.push_approver is not None and self.push_approver(command):
                return super().execute(command, timeout=timeout)
            reason = "git push requires human approval" + (" and was rejected" if self.push_approver else "")
            self.blocked.append((command, reason))
            return ExecuteResponse(output=f"BLOCKED by CodePilot guardrail: {reason}. Command not run: {command}",
                                   exit_code=BLOCKED_EXIT_CODE)
        verdict = check_command(command, self.root)
        if not verdict.allowed:
            self.blocked.append((command, verdict.reason))
            return ExecuteResponse(
                output=f"BLOCKED by CodePilot guardrail: {verdict.reason}. Command not run: {command}",
                exit_code=BLOCKED_EXIT_CODE,
            )
        return super().execute(command, timeout=timeout)
