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
    assert "round 1 of 2" in first and "last round" not in first and (workspace.working / "review.md").exists()
    second = request_review.invoke({})
    assert "round 2 of 2" in second and "call request_review once more to have a human decide" in second


@pytest.mark.parametrize("approved", [True, False])
def test_after_the_last_round_a_human_decides_without_another_review(workspace, monkeypatch, approved):
    from deepagents import create_deep_agent
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from codepilot.gates import Gate
    from codepilot.review import ReviewRecord

    model = reviewer(CHANGES, CHANGES)
    monkeypatch.setattr(agents, "_model", lambda role: model)
    request_review = agents.make_review_tool(workspace)
    request_review.invoke({}), request_review.invoke({})  # both rounds request changes
    agent = create_deep_agent(model=ScriptedModel(script=[tool_call("request_review", {}, "rr"), AIMessage(content="ok")]),
                              tools=[request_review], checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": f"h-{approved}"}}

    for _ in agent.stream({"messages": [{"role": "user", "content": "review"}]}, cfg, stream_mode="updates"):
        pass
    pending = agent.get_state(cfg).interrupts
    assert pending[0].value["gates"] == [Gate.REVIEW_OVERRULE.value]
    for _ in agent.stream(Command(resume={pending[0].id: {"approved": approved, "note": "n"}}), cfg,
                          stream_mode="updates"):
        pass

    assert model.calls == 2  # resuming re-ran the tool, but the Reviewer was not called again
    tool_reply = [m.content for m in agent.get_state(cfg).values["messages"] if m.type == "tool"][-1]
    assert tool_reply.startswith("APPROVED BY A HUMAN" if approved else "REJECTED: the human agreed")
    assert ReviewRecord(workspace.working / "review.json").approves(workspace.diff()) is approved


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
