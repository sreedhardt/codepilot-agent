"""Every tool each agent can call, pinned. A framework upgrade that adds a default tool fails here first."""

import subprocess

import pytest
from langchain_core.messages import AIMessage

from codepilot import agents
from codepilot.agents import TaskContext, build_orchestrator, build_team, build_tester
from codepilot.classifier import TaskType
from codepilot.repo_map import build_repo_map
from codepilot.skills import skill_for
from codepilot.workspace import TaskWorkspace
from tests.fakes import ScriptedModel

FILE_TOOLS = {"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute", "task"}

EXPECTED = {
    "orchestrator": FILE_TOOLS | {"write_todos", "request_review"},
    "repo_explorer": FILE_TOOLS | {"find_relevant_files"},
    "coder": FILE_TOOLS | {"write_todos", "write_diff_preview", "ask_human_to_run"},
    "test_agent": FILE_TOOLS,
    "pr_agent": FILE_TOOLS | {"open_pull_request"},
}


def tools_of(graph) -> set[str]:
    return set(graph.nodes["tools"].bound.tools_by_name)


@pytest.fixture
def team(tmp_path, monkeypatch):
    monkeypatch.setattr(agents, "_model", lambda role: ScriptedModel(script=[AIMessage(content="x")]))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("x = 1\n")
    for args in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "i"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    ws = TaskWorkspace(1, runs_dir=tmp_path / "runs", source=repo)
    ctx = TaskContext(1, "t", "u", None, TaskType.BUG_FIX)
    repo_map = build_repo_map(ws.sandbox)
    graphs = {sub["name"]: sub["runnable"] for sub in build_team(ws, repo_map, ctx)}
    graphs["test_agent"] = build_tester(ws)["runnable"]
    graphs["orchestrator"] = build_orchestrator(ws, repo_map, skill_for(TaskType.BUG_FIX), ctx)
    return graphs


@pytest.mark.parametrize("agent", sorted(EXPECTED))
def test_agent_tools_are_exactly_the_reviewed_set(team, agent):
    assert tools_of(team[agent]) == EXPECTED[agent]


def test_no_agent_can_touch_github_except_through_open_pull_request(team):
    github_words = {"merge", "push", "approve", "review", "branch", "commit", "pull", "request"}
    for agent, graph in team.items():
        for name in tools_of(graph):
            if github_words & set(name.split("_")):
                # request_review asks the Reviewer model; it never calls GitHub (see codepilot/review.py)
                assert (agent, name) in {("pr_agent", "open_pull_request"), ("orchestrator", "request_review")}


def test_every_agent_with_a_shell_is_told_the_shell_rules():
    from codepilot.agents import CODER_PROMPT, SHELL_RULES, TESTER_PROMPT

    for prompt in (CODER_PROMPT, TESTER_PROMPT):
        assert SHELL_RULES in prompt
