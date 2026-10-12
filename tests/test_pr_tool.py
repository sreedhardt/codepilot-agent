import subprocess

import pytest

from codepilot.agents import TaskContext, make_pr_tool
from codepilot.classifier import TaskType
from codepilot.workspace import TaskWorkspace

ARGS = {"summary": "Make search case-insensitive", "what_changed": ["Casefold in search"],
        "why": ["Users type lowercase"], "approach": "Normalize case before comparing."}


@pytest.fixture
def workspace(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "app.py").write_text("def f():\n    return 1\n")
    (repo / "tests" / "test_app.py").write_text("from app import f\n\ndef test_f():\n    assert f() == 1\n")
    for args in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    return TaskWorkspace(7, runs_dir=tmp_path / "runs", source=repo)


def make_tool(ws):
    ctx = TaskContext(7, "Search is case-sensitive", "https://example.com/7", "someone", TaskType.BUG_FIX)
    return make_pr_tool(ws, ctx, open_pr=False)


def test_no_changes_means_no_pr(workspace):
    assert make_tool(workspace).invoke(ARGS).startswith("FAILED: the sandbox has no changes")


def test_failing_suite_means_no_pr(workspace):
    (workspace.sandbox / "app.py").write_text("def f():\n    return 2\n")
    assert make_tool(workspace).invoke(ARGS).startswith("FAILED: the test suite does not pass")


def test_dry_run_writes_preview_and_second_call_does_nothing(workspace):
    (workspace.sandbox / "app.py").write_text("def f():\n    # tidy\n    return 1\n")
    tool = make_tool(workspace)

    first = tool.invoke(ARGS)
    second = tool.invoke(ARGS)

    assert first.startswith("DRY RUN")
    assert second.startswith("ALREADY CALLED") and "DRY RUN" in second
    plan = (workspace.working / "pr_plan.md").read_text()
    assert "codepilot/issue-7-search-is-case-sensitive" in plan
    assert "fix(#7): Make search case-insensitive" in plan
    assert (workspace.working / "pr_result.md").read_text().startswith("DRY RUN")
