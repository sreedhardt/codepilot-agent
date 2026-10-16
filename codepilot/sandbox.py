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

from deepagents.backends import FilesystemBackend, LocalShellBackend
from deepagents.backends.protocol import DeleteResult, EditResult, ExecuteResponse, WriteResult
from wcmatch.glob import BRACE, DOTGLOB, GLOBSTAR, globmatch

BLOCKED_EXIT_CODE = 126

SHELL_METACHARACTERS = (";", "&", "|", "`", "$(", ">", "<", "\n")
READ_ONLY_PROGRAMS = {"ls", "cat", "head", "tail", "grep", "wc", "find", "pwd"}
GIT_READ_ONLY = {"status", "diff", "log", "show"}
FIND_FORBIDDEN = {"-exec", "-execdir", "-delete", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls"}
GIT_FORBIDDEN_OPTIONS = ("--output", "--ext-diff", "--textconv", "--exec-path")  # write files or run programs


@dataclass
class BlockedAction:
    """A guardrail refusal: which agent tried what, and why it was refused."""

    agent: str
    action: str
    reason: str


@dataclass
class Verdict:
    allowed: bool
    reason: str = ""
    hard: bool = False  # never runs, even with a human's approval (it reaches outside the sandbox)


def is_test_command(command: str) -> bool:
    argv = shlex.split(command)
    return bool(argv) and (argv[0] == "pytest" or (argv[0] in {"python", "python3"} and argv[1:3] == ["-m", "pytest"]))


def is_git_push(command: str) -> bool:
    """A single plain `git push ...` command (chained commands are rejected separately)."""
    if any(m in command for m in SHELL_METACHARACTERS):
        return False
    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    return argv[:2] == ["git", "push"]


def escapes_sandbox(command: str, root: Path) -> str | None:
    """Why a command reaches outside the sandbox, or None. Checks every word, including chained commands."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        words = list(lexer)
    except ValueError as e:
        return f"could not parse command: {e}"
    for word in words:
        # Check option values too: --rootdir=/ names a path as surely as / does.
        for part in (word, word.partition("=")[2]) if word.startswith("-") else (word,):
            if part.startswith("~") or ".." in Path(part).parts:
                return f"path escapes the sandbox: {word}"
            if part.startswith("/") and not Path(part).resolve().is_relative_to(root.resolve()):
                return f"path outside the sandbox: {word}"
    return None


def check_command(command: str, root: Path) -> Verdict:
    """Decide whether a shell command may run inside the sandbox rooted at `root`."""
    if escape := escapes_sandbox(command, root):
        return Verdict(False, escape, hard=True)
    if any(m in command for m in SHELL_METACHARACTERS):
        return Verdict(False, "one plain command per call; no ;, &&, |, redirection or $(...)")
    try:
        argv = shlex.split(command)
    except ValueError as e:
        return Verdict(False, f"could not parse command: {e}")
    if not argv:
        return Verdict(False, "empty command")

    program, args = argv[0], argv[1:]
    if program in {"python", "python3"}:
        if args[:2] == ["-m", "pytest"]:
            return Verdict(True)
        return Verdict(False, "python may only run the test suite: use `python -m pytest ...`")
    if program == "pytest":
        return Verdict(True)
    if program == "git":
        if any(a == opt or a.startswith(opt + "=") for a in argv[1:] for opt in GIT_FORBIDDEN_OPTIONS):
            return Verdict(False, "git options that write files or run programs are not allowed")
        if args and args[0] in GIT_READ_ONLY:
            return Verdict(True)
        return Verdict(False, f"git is read-only here: allowed subcommands are {sorted(GIT_READ_ONLY)}")
    if program == "find" and FIND_FORBIDDEN.intersection(args):
        return Verdict(False, "find may not execute or delete")
    if program in READ_ONLY_PROGRAMS:
        return Verdict(True)
    return Verdict(False, f"`{program}` is not allowed (network, installs and deletion are blocked)")


PROTECTED_FILES = ["/**/.env", "/**/.env.*", "/**/*.secret", "/**/*.pem", "/**/*.key", "/**/*credentials*"]
# CI and repository automation run with the repository's secrets, and git's own config can make read-only git
# commands run programs (core.fsmonitor, diff drivers): changes here need a human, never an agent.
AUTOMATION_FILES = ["/.git/**", "/.github/**", "/.gitlab-ci.yml", "/.circleci/**", "/Jenkinsfile",
                    "/azure-pipelines.yml", "/.gitattributes"]


def _matches(path: str, patterns: list[str]) -> bool:
    path = "/" + path.lstrip("/")
    return globmatch(path, patterns, flags=GLOBSTAR | BRACE | DOTGLOB)


class GuardedShellBackend(LocalShellBackend):
    """LocalShellBackend whose `execute` only runs commands that pass `check_command`.

    It also enforces a write policy on `write_file`/`edit_file`, because DeepAgents
    does not support declarative permissions on backends that can execute commands:
    protected files (secrets) are never writable, and `writable`, when given,
    limits writes to matching paths.

    Refusals are appended to `self.blocked` as BlockedAction records, tagged with `agent`.
    """

    def __init__(
        self,
        root_dir: str | Path,
        *,
        env: dict[str, str] | None = None,
        timeout: int = 120,
        writable: list[str] | None = None,
        protected: list[str] = PROTECTED_FILES,
        blocked: list[BlockedAction] | None = None,
        push_approver: Callable[[str], bool] | None = None,
        agent: str = "agent",
        can_escalate: bool = False,
        test_runner: Callable[[list[str]], "object"] | None = None,
    ) -> None:
        super().__init__(root_dir=root_dir, virtual_mode=True, env=env, inherit_env=False, timeout=timeout)
        self.root = Path(root_dir)
        self.writable = writable
        self.protected = protected
        self.blocked = blocked if blocked is not None else []
        self.push_approver = push_approver
        self.agent = agent
        self.can_escalate = can_escalate  # this agent has the ask_human_to_run tool
        self.test_runner = test_runner  # runs test commands in isolation (see codepilot/test_sandbox.py)

    def restricted(self, writable: list[str], agent: str) -> "GuardedShellBackend":
        """Same sandbox and audit log, for another agent, with writes limited to `writable` paths."""
        return GuardedShellBackend(self.root, env=self._env, timeout=self._default_timeout, writable=writable,
                                   protected=self.protected, blocked=self.blocked, push_approver=self.push_approver,
                                   agent=agent, test_runner=self.test_runner)

    def _write_denied(self, file_path: str) -> str | None:
        if _matches(file_path, self.protected):
            reason = "protected file: secrets and credentials are never edited"
        elif _matches(file_path, AUTOMATION_FILES):
            reason = "CI, repository automation and git internals are never edited by agents"
        elif self.writable is not None and not _matches(file_path, self.writable):
            reason = f"outside this agent's writable paths {self.writable}"
        else:
            return None
        self.blocked.append(BlockedAction(self.agent, f"write {file_path}", reason))
        return f"BLOCKED by CodePilot guardrail: {file_path}: {reason}. File not changed."

    def write(self, file_path: str, content: str) -> WriteResult:
        if error := self._write_denied(file_path):
            return WriteResult(error=error)
        return super().write(file_path, content)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:  # noqa: FBT001, FBT002
        if error := self._write_denied(file_path):
            return EditResult(error=error)
        return super().edit(file_path, old_string, new_string, replace_all)

    def delete(self, file_path: str) -> DeleteResult:
        if error := self._write_denied(file_path):
            return DeleteResult(error=error)
        return super().delete(file_path)

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        if is_git_push(command):
            # Network operation: never allowed by policy alone, only with a human's approval.
            if self.push_approver is not None and self.push_approver(command):
                return super().execute(command, timeout=timeout)
            reason = "git push requires human approval" + (" and was rejected" if self.push_approver else "")
            self.blocked.append(BlockedAction(self.agent, command, reason))
            return ExecuteResponse(output=f"BLOCKED by CodePilot guardrail: {reason}. Command not run: {command}",
                                   exit_code=BLOCKED_EXIT_CODE)
        verdict = check_command(command, self.root)
        if not verdict.allowed:
            self.blocked.append(BlockedAction(self.agent, command, verdict.reason))
            hint = ("" if verdict.hard or not self.can_escalate else
                    " If this exact command is truly needed, call ask_human_to_run with it and explain why;"
                    " a human decides.")
            return ExecuteResponse(
                output=f"BLOCKED by CodePilot guardrail: {verdict.reason}. Command not run: {command}.{hint}",
                exit_code=BLOCKED_EXIT_CODE,
            )
        if self.test_runner is not None and is_test_command(command):
            result = self.test_runner(shlex.split(command))
            return ExecuteResponse(output=result.output, exit_code=result.exit_code)
        return super().execute(command, timeout=timeout)

    def run_once(self, command: str, timeout: int | None = None) -> ExecuteResponse:
        """Run a command a human approved past the policy. Still never one that reaches outside the sandbox."""
        if escape := escapes_sandbox(command, self.root):
            return ExecuteResponse(output=f"BLOCKED by CodePilot guardrail: {escape}", exit_code=BLOCKED_EXIT_CODE)
        return super().execute(command, timeout=timeout)


class ProtectedFilesBackend(FilesystemBackend):
    """A filesystem view where matching files can be read but never written, edited or deleted by an agent.

    Used for shared working memory, whose record files (task, PR outcome, approvals log) are written only by
    CodePilot's own code, and for the read-only skills directory.
    """

    def __init__(self, root_dir: str | Path, protected: list[str], blocked: list[BlockedAction] | None = None,
                 reason: str = "record file: written only by CodePilot, read-only for agents", agent: str = "agent") -> None:
        super().__init__(root_dir=root_dir, virtual_mode=True)
        self.protected = protected
        self.blocked = blocked if blocked is not None else []
        self.reason = reason
        self.agent = agent

    def _denied(self, file_path: str) -> str | None:
        if not _matches(file_path, self.protected):
            return None
        self.blocked.append(BlockedAction(self.agent, f"write {file_path}", self.reason))
        return f"BLOCKED by CodePilot guardrail: {file_path}: {self.reason}. File not changed."

    def write(self, file_path: str, content: str) -> WriteResult:
        if error := self._denied(file_path):
            return WriteResult(error=error)
        return super().write(file_path, content)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:  # noqa: FBT001, FBT002
        if error := self._denied(file_path):
            return EditResult(error=error)
        return super().edit(file_path, old_string, new_string, replace_all)

    def delete(self, file_path: str) -> DeleteResult:
        if error := self._denied(file_path):
            return DeleteResult(error=error)
        return super().delete(file_path)
