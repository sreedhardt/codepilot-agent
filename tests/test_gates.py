import json
import subprocess
from types import SimpleNamespace

import pytest
from deepagents import CompiledSubAgent, create_deep_agent
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from codepilot import github_ops
from codepilot.agents import TaskContext, make_pr_tool
from codepilot.classifier import TaskType
from codepilot.gates import Decision, Gate, RetryGateMiddleware
from codepilot.workspace import TaskWorkspace
from tests.fakes import ScriptedModel, approve, tool_call


def run_until_pause(agent, payload, cfg):
    for _ in agent.stream(payload, cfg, stream_mode="updates", subgraphs=True):
        pass
    return agent.get_state(cfg).interrupts


@pytest.mark.parametrize("reply, approved", [({"approved": True}, True), ({"approved": False}, False),
                                             ("approve", True), ("reject", False)])
def test_decision_parses_replies(reply, approved):
    assert Decision.from_reply(reply).approved is approved


def test_retry_gate_pauses_before_third_round_and_rejection_stops(tmp_path):
    tester = create_deep_agent(model=ScriptedModel(script=[AIMessage(content="FAIL - 1 failed: test_search")]),
                               name="test_agent")
    verify = lambda i: tool_call("task", {"description": "verify", "subagent_type": "test_agent"}, f"v{i}")
    coder = create_deep_agent(
        model=ScriptedModel(script=[verify(1), verify(2), verify(3), AIMessage(content="gave up")]),
        subagents=[CompiledSubAgent(name="test_agent", description="tests", runnable=tester)],
        middleware=[RetryGateMiddleware(tmp_path / "approvals.jsonl")],
        checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "retry"}}

    pending = run_until_pause(coder, {"messages": [{"role": "user", "content": "fix it"}]}, cfg)
    assert [p.value["gates"] for p in pending] == [[Gate.RETRY_AFTER_FAILURES.value]]
    assert pending[0].value["details"]["failed_rounds"] == 2

    run_until_pause(coder, Command(resume={pending[0].id: {"approved": False, "note": "stop"}}), cfg)
    messages = coder.get_state(cfg).values["messages"]
    assert any("REJECTED by human" in str(m.content) for m in messages)
    assert json.loads((tmp_path / "approvals.jsonl").read_text())["approved"] is False


def test_no_approval_allows_more_than_three_retries(tmp_path):
    tester = ScriptedModel(script=[AIMessage(content="FAIL - 1 failed: test_search")])
    verify = lambda i: tool_call("task", {"description": "verify", "subagent_type": "test_agent"}, f"v{i}")
    coder = create_deep_agent(
        model=ScriptedModel(script=[verify(i) for i in range(1, 6)] + [AIMessage(content="gave up")]),
        subagents=[CompiledSubAgent(name="test_agent", description="tests",
                                    runnable=create_deep_agent(model=tester, name="test_agent"))],
        middleware=[RetryGateMiddleware()],
        checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "cap"}}
    pending = run_until_pause(coder, {"messages": [{"role": "user", "content": "fix it"}]}, cfg)
    while pending:  # a human who approves every extra round
        pending = run_until_pause(coder, Command(resume={p.id: {"approved": True} for p in pending}), cfg)

    assert tester.calls == 4  # the first round plus 3 retries
    messages = coder.get_state(cfg).values["messages"]
    assert any(str(m.content).startswith("STOPPED: the tests failed 4 times") for m in messages)


@pytest.fixture
def workspace(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "app.py").write_text("def f():\n    return 1\n")
    (repo / "tests" / "test_app.py").write_text("from app import f\n\ndef test_f():\n    assert f() == 1\n")
    for args in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    ws = TaskWorkspace(7, runs_dir=tmp_path / "runs", source=repo)
    (ws.sandbox / "app.py").write_text("def f():\n    # tidy\n    return 1\n")
    return ws


@pytest.fixture
def fake_github(monkeypatch):
    opened = []
    monkeypatch.setattr(github_ops, "target_repository", lambda: SimpleNamespace(default_branch="main"))
    monkeypatch.setattr(github_ops, "open_pull_request",
                        lambda repo, sandbox, plan, reviewer: opened.append(plan) or "https://example.com/pr/1")
    return opened


def pr_agent_with_checkpointer(ws):
    approve(ws)
    ctx = TaskContext(7, "Search is case-sensitive", "https://example.com/7", "someone", TaskType.BUG_FIX)
    args = {"summary": "Tidy f", "what_changed": ["Comment in f"], "why": ["Readability"], "approach": "Add a comment."}
    return create_deep_agent(
        model=ScriptedModel(script=[tool_call("open_pull_request", args, "pr1"), AIMessage(content="done")]),
        tools=[make_pr_tool(ws, ctx, open_pr=True)],
        checkpointer=InMemorySaver(),
    )


@pytest.mark.parametrize("approved", [True, False])
def test_pr_to_main_waits_for_approval(workspace, fake_github, approved):
    agent, cfg = pr_agent_with_checkpointer(workspace), {"configurable": {"thread_id": f"pr-{approved}"}}

    pending = run_until_pause(agent, {"messages": [{"role": "user", "content": "open the PR"}]}, cfg)
    assert pending[0].value["gates"] == [Gate.PR_TO_DEFAULT_BRANCH.value]
    assert fake_github == []  # nothing opened while waiting

    run_until_pause(agent, Command(resume={pending[0].id: {"approved": approved}}), cfg)
    result = (workspace.working / "pr_result.md").read_text()
    if approved:
        assert result.startswith("PR_OPENED https://example.com/pr/1") and len(fake_github) == 1
    else:
        assert result.startswith("REJECTED by human") and fake_github == []


def test_commit_over_five_files_adds_large_commit_gate(workspace, fake_github):
    for i in range(6):
        (workspace.sandbox / f"extra_{i}.py").write_text("x = 1\n")
    agent, cfg = pr_agent_with_checkpointer(workspace), {"configurable": {"thread_id": "large"}}
    pending = run_until_pause(agent, {"messages": [{"role": "user", "content": "open the PR"}]}, cfg)
    assert set(pending[0].value["gates"]) == {Gate.PR_TO_DEFAULT_BRANCH.value, Gate.LARGE_COMMIT.value}
