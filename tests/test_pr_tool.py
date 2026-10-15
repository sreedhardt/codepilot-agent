import subprocess

import pytest

from codepilot.agents import TaskContext, make_pr_tool
from codepilot.classifier import TaskType
from codepilot.workspace import TaskWorkspace
from tests.fakes import approve

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


def make_tool(ws, reviewed=True):
    """The PR tool, with the sandbox's current change approved by a review unless `reviewed` is False."""
    if reviewed:
        approve(ws)
    ctx = TaskContext(7, "Search is case-sensitive", "https://example.com/7", "someone", TaskType.BUG_FIX)
    return make_pr_tool(ws, ctx, open_pr=False)


def test_no_changes_means_no_pr(workspace):
    assert make_tool(workspace).invoke(ARGS).startswith("FAILED: the sandbox has no changes")


def test_failing_suite_means_no_pr(workspace):
    (workspace.sandbox / "app.py").write_text("def f():\n    return 2\n")
    result = make_tool(workspace).invoke(ARGS)
    assert result.startswith("FAILED: the test suite does not pass") and "1 failed" in result


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


def test_missing_interpreter_is_a_failed_run_not_a_crash(workspace, monkeypatch, tmp_path):
    monkeypatch.setenv("CODEPILOT_TARGET_PYTHON_BIN", str(tmp_path / "no-python-here"))
    (workspace.sandbox / "app.py").write_text("def f():\n    # tidy\n    return 1\n")
    result = make_tool(workspace).invoke(ARGS)
    assert result.startswith("FAILED: the test suite does not pass") and "could not run the test suite" in result


def test_blocked_actions_name_the_agent_that_tried_them(workspace):
    workspace.write("pr_result.md", "DRY RUN\n")

    workspace.tester_backend().write("/app.py", "x = 2\n")
    workspace.orchestrator_backend().edit("/working/pr_result.md", "DRY RUN", "PR_OPENED fake")
    workspace.coder_backend().execute("git commit -am sneaky")

    assert [(b.agent, b.action.split()[0]) for b in workspace.shell.blocked] == [
        ("test_agent", "write"), ("orchestrator", "write"), ("coder", "git")]
    assert (workspace.working / "pr_result.md").read_text() == "DRY RUN\n"


def test_possible_secret_blocks_the_pr_before_any_preview(workspace, monkeypatch):
    held = "held" + "-value-" + "9f3k2m8q"
    monkeypatch.setenv("GITHUB_TOKEN", held)
    (workspace.sandbox / "app.py").write_text(f"def f():\n    # {held}\n    return 1\n")
    result = make_tool(workspace).invoke(ARGS)
    assert result.startswith("BLOCKED: possible secrets") and "app.py:2: value of $GITHUB_TOKEN" in result
    assert held not in result
    assert not (workspace.working / "pr_plan.md").exists()
    assert [b.agent for b in workspace.shell.blocked] == ["pr_agent"]


def test_pr_needs_an_approving_review_of_the_exact_current_change(workspace):
    (workspace.sandbox / "app.py").write_text("def f():\n    # tidy\n    return 1\n")
    unreviewed = make_tool(workspace, reviewed=False).invoke(ARGS)
    assert unreviewed.startswith("NOT OPENED") and not (workspace.working / "pr_result.md").exists()

    tool = make_tool(workspace)  # approved...
    (workspace.sandbox / "app.py").write_text("def f():\n    # tidied again\n    return 1\n")  # ...then edited
    assert tool.invoke(ARGS).startswith("NOT OPENED")

    assert make_tool(workspace).invoke(ARGS).startswith("DRY RUN")  # re-reviewed: proceeds
