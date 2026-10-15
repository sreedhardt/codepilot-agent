"""Secret scan of a change before it leaves the machine as a pull request.

Two checks over the lines a change adds and the PR text:

- exact matches of the secret values CodePilot itself holds (environment variables named like keys, tokens,
  secrets or passwords), which finds a leak of its own credentials with no false positives;
- known credential formats, for secrets that came from anywhere else.

Findings name the location and kind of secret, never the value.
"""

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

PATTERNS: dict[str, re.Pattern] = {
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "Anthropic API key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"),
    "OpenAI API key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{20,}"),
    "Groq API key": re.compile(r"\bgsk_[A-Za-z0-9]{40,}"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{60,})"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}"),
    "Slack token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
}
SECRET_ENV_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)
MIN_SECRET_LENGTH = 12  # shorter values ("true", a port) would match ordinary code


@dataclass(frozen=True)
class Finding:
    location: str
    kind: str

    def __str__(self) -> str:
        return f"{self.location}: {self.kind}"


def held_secrets(environ: dict[str, str] | None = None) -> list[tuple[str, str]]:
    """(variable name, secret) pairs from CodePilot's environment.

    A multi-line value such as a PEM key contributes each of its lines, since the scan goes line by line.
    """
    environ = os.environ if environ is None else environ
    return [(name, part.strip()) for name, value in environ.items() if SECRET_ENV_NAME.search(name)
            for part in value.splitlines() if len(part.strip()) >= MIN_SECRET_LENGTH]


def scan_lines(lines: list[tuple[str, str]], secrets: list[tuple[str, str]]) -> list[Finding]:
    """Scan (location, line) pairs."""
    findings = []
    for location, line in lines:
        findings += [Finding(location, f"value of ${name}") for name, value in secrets if value in line]
        findings += [Finding(location, kind) for kind, pattern in PATTERNS.items() if pattern.search(line)]
    return findings


HUNK = re.compile(r"^@@ -\S+ \+(\d+)")


def added_lines(sandbox: Path) -> list[tuple[str, str]]:
    """(path:line, text) for every line the sandbox adds relative to HEAD; new files contribute all lines."""
    added = []
    diff = subprocess.run(["git", "diff", "HEAD", "-U0", "--no-color", "--no-ext-diff"], cwd=sandbox,
                          capture_output=True, text=True, check=True).stdout
    path, number, in_header = None, 0, False
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            in_header = True
        elif in_header and line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None  # "+++ /dev/null": a deletion adds nothing
        elif match := HUNK.match(line):
            number, in_header = int(match.group(1)), False
        elif path and not in_header and line.startswith("+"):
            added.append((f"{path}:{number}", line[1:]))
            number += 1
    untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], cwd=sandbox,
                               capture_output=True, text=True, check=True).stdout.splitlines()
    for path in untracked:
        try:
            text = (sandbox / path).read_text()
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable: outside what a line scan can judge
        added += [(f"{path}:{n}", line) for n, line in enumerate(text.splitlines(), 1)]
    return added


def scan_change(sandbox: Path, texts: dict[str, str], environ: dict[str, str] | None = None) -> list[Finding]:
    """Scan the sandbox's added lines and the given texts (e.g. the PR body), keyed by a label for reports."""
    lines = added_lines(sandbox)
    for label, text in texts.items():
        lines += [(f"{label}:{n}", line) for n, line in enumerate(text.splitlines(), 1)]
    return scan_lines(lines, held_secrets(environ))
