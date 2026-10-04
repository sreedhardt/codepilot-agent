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
from dataclasses import dataclass
from pathlib import Path

from deepagents.backends import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse

BLOCKED_EXIT_CODE = 126

SHELL_METACHARACTERS = (";", "&", "|", "`", "$(", ">", "<", "\n")
READ_ONLY_PROGRAMS = {"ls", "cat", "head", "tail", "grep", "wc", "find", "pwd"}
GIT_READ_ONLY = {"status", "diff", "log", "show"}
FIND_FORBIDDEN = {"-exec", "-execdir", "-delete", "-ok", "-okdir", "-fprint", "-fls"}


@dataclass
class Verdict:
    allowed: bool
    reason: str = ""


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


class GuardedShellBackend(LocalShellBackend):
    """LocalShellBackend whose `execute` only runs commands that pass `check_command`.

    Blocked attempts are recorded in `self.blocked` as (command, reason) pairs.
    """

    def __init__(self, root_dir: str | Path, *, env: dict[str, str] | None = None, timeout: int = 120) -> None:
        super().__init__(root_dir=root_dir, virtual_mode=True, env=env, inherit_env=False, timeout=timeout)
        self.root = Path(root_dir)
        self.blocked: list[tuple[str, str]] = []

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        verdict = check_command(command, self.root)
        if not verdict.allowed:
            self.blocked.append((command, verdict.reason))
            return ExecuteResponse(
                output=f"BLOCKED by CodePilot guardrail: {verdict.reason}. Command not run: {command}",
                exit_code=BLOCKED_EXIT_CODE,
            )
        return super().execute(command, timeout=timeout)
