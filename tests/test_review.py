import subprocess

import pytest

from codepilot import agents
from codepilot.models import Role
from codepilot.review import Review, review_change
from codepilot.workspace import TaskWorkspace
from tests.fakes import ScriptedModel, tool_call

CHANGES = {"verdict": "request_changes", "summary": "Debug output left in.",
           "findings": [{"file": "app.py", "problem": "print() left in f", "blocking": True}]}


def reviewer(*replies):
    return ScriptedModel(script=[tool_call("Review", reply, f"r{i}") for i, reply in enumerate(replies)])


def test_a_blocking_finding_overrides_an_approve_verdict():
    contradictory = {**CHANGES, "verdict": "approve"}
    assert review_change(reviewer(contradictory), "issue", "diff", "1 passed").verdict == "request_changes"
    clean = {"verdict": "approve", "summary": "ok", "findings": [{"file": "a", "problem": "nit", "blocking": False}]}
    assert review_change(reviewer(clean), "issue", "diff", "1 passed").verdict == "approve"


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "app.py").write_text("def f():\n    return 1\n")
    (repo / "tests" / "test_app.py").write_text("from app import f\n\ndef test_f():\n    assert f() == 1\n")
    for args in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    ws = TaskWorkspace(7, runs_dir=tmp_path / "runs", source=repo)
    ws.write("task.md", "# Issue #7: annotate f\n")
    (ws.sandbox / "app.py").write_text("def f():\n    print('debug')\n    return 1\n")
    return ws


def test_review_rounds_are_recorded_and_capped(workspace, monkeypatch):
    model = reviewer(CHANGES, CHANGES)
    monkeypatch.setattr(agents, "_model", lambda role: model if role == Role.REVIEWER else pytest.fail(role))
    request_review = agents.make_review_tool(workspace)

    first = request_review.invoke({})
    assert first.startswith("VERDICT: REQUEST_CHANGES") and "[blocking] app.py: print() left in f" in first
    assert "round 1 of 2" in first and (workspace.working / "review.md").exists()
    assert request_review.invoke({}).startswith("VERDICT: REQUEST_CHANGES")
    assert request_review.invoke({}).startswith("NO MORE REVIEWS")
    assert model.calls == 2  # the capped call costs nothing


def test_failing_tests_are_not_sent_for_review(workspace, monkeypatch):
    monkeypatch.setattr(agents, "_model", lambda role: pytest.fail("reviewed a failing change"))
    (workspace.sandbox / "app.py").write_text("def f():\n    return 2\n")
    assert agents.make_review_tool(workspace).invoke({}).startswith("Not reviewed: the test suite fails")
