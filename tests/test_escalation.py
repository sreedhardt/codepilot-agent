"""A blocked Coder command goes to a human, who can allow exactly one run of it."""

import subprocess

import pytest
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from codepilot.agents import make_escalation_tool
from codepilot.gates import Gate
from codepilot.workspace import TaskWorkspace
from tests.fakes import ScriptedModel, tool_call

APPEND = "echo ran >> count.txt"  # blocked (redirection), harmless, and leaves a trace of each run


@pytest.fixture
def workspace(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("x = 1\n")
    for args in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    return TaskWorkspace(7, runs_dir=tmp_path / "runs", source=repo)


def run(workspace, command, reply):
    agent = create_deep_agent(
        model=ScriptedModel(script=[tool_call("ask_human_to_run", {"command": command, "reason": "needed"}, "a1"),
                                    AIMessage(content="done")]),
        tools=[make_escalation_tool(workspace)], checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t"}}
    for _ in agent.stream({"messages": [{"role": "user", "content": "go"}]}, cfg, stream_mode="updates"):
        pass
    pending = agent.get_state(cfg).interrupts
    if pending and reply is not None:
        for _ in agent.stream(Command(resume={pending[0].id: reply}), cfg, stream_mode="updates"):
            pass
    tool_replies = [m.content for m in agent.get_state(cfg).values["messages"] if m.type == "tool"]
    return pending, tool_replies


def test_approved_command_runs_exactly_once(workspace):
    pending, replies = run(workspace, APPEND, {"approved": True})
    assert pending[0].value["gates"] == [Gate.BLOCKED_OPERATION.value]
    assert pending[0].value["details"]["coder_reason"] == "needed"
    assert replies[-1].startswith("APPROVED by human and run once")
    assert (workspace.sandbox / "count.txt").read_text() == "ran\n"  # resumed, yet run once


def test_rejected_command_never_runs(workspace):
    _, replies = run(workspace, APPEND, {"approved": False, "note": "no"})
    assert replies[-1].startswith("REJECTED by human") and not (workspace.sandbox / "count.txt").exists()


@pytest.mark.parametrize("command, answer", [("cat /etc/passwd", "Cannot be approved"), ("ls", "Not needed"),
                                             ("git push origin main", "Not here")])
def test_some_requests_never_reach_a_human(workspace, command, answer):
    pending, replies = run(workspace, command, None)
    assert not pending  # no approval was requested
    assert replies[-1].startswith(answer)
