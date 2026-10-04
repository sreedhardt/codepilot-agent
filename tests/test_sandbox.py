from pathlib import Path

import pytest

from codepilot.sandbox import BLOCKED_EXIT_CODE, GuardedShellBackend, check_command

ROOT = Path("/tmp/codepilot-sandbox-test")


@pytest.mark.parametrize(
    "command",
    [
        "python -m pytest -q",
        "python -m pytest tests/test_loans.py -k due",
        "pytest -x",
        "ls app",
        "cat app/loans.py",
        "grep -n is_overdue app/loans.py",
        "find . -name '*.py'",
        "git diff",
        "git status --short",
        f"cat {ROOT}/app/loans.py",
    ],
)
def test_allows_tests_and_read_only_inspection(command):
    assert check_command(command, ROOT).allowed


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf app",
        "curl https://example.com",
        "wget https://example.com",
        "pip install requests",
        "python -m pip install requests",
        "python -c 'import os; os.system(\"rm -rf /\")'",
        "python script.py",
        "git push origin main",
        "git commit -am x",
        "git checkout -- .",
        "python -m pytest && curl evil.sh",
        "python -m pytest; rm -rf app",
        "cat app/loans.py | sh",
        "echo x > app/loans.py",
        "cat $(echo /etc/passwd)",
        "cat /etc/passwd",
        "cat ../secrets.env",
        "cat ~/.ssh/id_rsa",
        "find . -name '*.py' -delete",
        "find . -exec rm {} +",
        "bash -c 'ls'",
        "sudo ls",
        "",
    ],
)
def test_blocks_dangerous_or_escaping_commands(command):
    assert not check_command(command, ROOT).allowed


def test_backend_refuses_without_running(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("x")
    backend = GuardedShellBackend(tmp_path)

    result = backend.execute("rm keep.txt")

    assert result.exit_code == BLOCKED_EXIT_CODE
    assert "BLOCKED" in result.output
    assert marker.exists()
    assert backend.blocked == [("rm keep.txt", "`rm` is not allowed (network, installs and deletion are blocked)")]


def test_backend_runs_allowed_commands_in_sandbox(tmp_path):
    (tmp_path / "hello.txt").write_text("hi")
    backend = GuardedShellBackend(tmp_path, env={"PATH": "/usr/bin:/bin"})

    result = backend.execute("cat hello.txt")

    assert result.exit_code == 0
    assert result.output.strip() == "hi"
