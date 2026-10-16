import subprocess

import pytest

from codepilot import agents
from codepilot.models import Role
from codepilot.review import Review, review_change
from codepilot.workspace import TaskWorkspace
from tests.fakes import ScriptedModel, tool_call

CHECKED = {"requirements": [{"requirement": "annotate f", "met": True}],
           "files": [{"path": "app.py", "related": True, "new_tests_assert_behaviour": True}]}
CHANGES = {**CHECKED, "summary": "Debug output left in.",
           "findings": [{"file": "app.py", "problem": "print() left in f", "blocking": True}]}


def reviewer(*replies):
    return ScriptedModel(script=[tool_call("Review", reply, f"r{i}") for i, reply in enumerate(replies)])


def test_the_verdict_is_decided_by_the_checklist_and_blocking_findings():
    def verdict(**changes):
        return review_change(reviewer({**CHECKED, "summary": "ok", "findings": [], **changes}),
                             "issue", "diff", "1 passed").verdict

    assert verdict() == "approve"
    assert verdict(findings=[{"file": "a", "problem": "nit", "blocking": False}]) == "approve"
    assert verdict(findings=[{"file": "a", "problem": "bug", "blocking": True}]) == "request_changes"
    assert verdict(requirements=[{"requirement": "document /health", "met": False}]) == "request_changes"
    assert verdict(files=[{"path": "scripts/x.py", "related": False, "new_tests_assert_behaviour": True}]) == \
        "request_changes"
    assert verdict(files=[{"path": "tests/t.py", "related": True, "new_tests_assert_behaviour": False}]) == \
        "request_changes"


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
    ws.write("repo_map.md", "app.py [python] :: f()\n")
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


def test_malformed_review_output_is_retried_once_then_reported_without_using_a_round(workspace, monkeypatch):
    malformed = {"requirements": [], "files": [], "findings": []}  # no summary: what Haiku produced once
    good = {**CHECKED, "summary": "ok", "findings": []}
    assert review_change(reviewer(malformed, good), "i", "d", "t").verdict == "approve"

    model = reviewer(malformed, malformed, good)
    monkeypatch.setattr(agents, "_model", lambda role: model)
    request_review = agents.make_review_tool(workspace)
    assert request_review.invoke({}).startswith("REVIEW UNAVAILABLE")
    assert not (workspace.working / "review.json").exists()
    assert "round 1 of 2" in request_review.invoke({})
