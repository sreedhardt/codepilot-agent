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


@pytest.mark.parametrize("path", ["/.env", "/config/.env.prod", "/deploy/server.pem", "/app/aws_credentials.json", "/x.key"])
def test_protected_files_are_never_writable(tmp_path, path):
    backend = GuardedShellBackend(tmp_path)
    assert "BLOCKED" in backend.write(path, "x").error
    assert not (tmp_path / path.lstrip("/")).exists()


def test_writable_paths_limit_a_restricted_backend(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "code.py").write_text("a = 1\n")
    coder = GuardedShellBackend(tmp_path)
    tester = coder.restricted(["/tests/**"])

    assert tester.write("/tests/test_new.py", "def test(): pass\n").error is None
    assert "BLOCKED" in tester.edit("/app/code.py", "a = 1", "a = 2").error
    assert (tmp_path / "app" / "code.py").read_text() == "a = 1\n"
    assert coder.edit("/app/code.py", "a = 1", "a = 2").error is None
    assert tester.blocked is coder.blocked and len(coder.blocked) == 1


def test_git_push_is_blocked_without_an_approver(tmp_path):
    backend = GuardedShellBackend(tmp_path)
    result = backend.execute("git push origin main")
    assert result.exit_code == BLOCKED_EXIT_CODE and "requires human approval" in result.output


def test_git_push_runs_only_when_approved(tmp_path):
    asked = []
    backend = GuardedShellBackend(tmp_path, env={"PATH": "/usr/bin:/bin"}, push_approver=lambda c: asked.append(c) or False)
    result = backend.execute("git push origin main")
    assert asked == ["git push origin main"]
    assert "rejected" in result.output and result.exit_code == BLOCKED_EXIT_CODE

    approved = GuardedShellBackend(tmp_path, env={"PATH": "/usr/bin:/bin"}, push_approver=lambda c: True)
    result = approved.execute("git push origin main")
    assert "BLOCKED" not in result.output  # it ran (and fails here: tmp_path is not a repository)


def test_chained_git_push_never_reaches_the_approver(tmp_path):
    backend = GuardedShellBackend(tmp_path, push_approver=lambda c: pytest.fail("approver must not be asked"))
    assert backend.execute("git push && rm -rf app").exit_code == BLOCKED_EXIT_CODE
