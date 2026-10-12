"""End-to-end runner test with scripted models: the full agent team, no API calls."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from codepilot import agents, runner
from codepilot.classifier import Classification, TaskType
from codepilot.models import Role
from codepilot.tasks import TaskRegistry, TaskState
from tests.fakes import ScriptedModel, tool_call


def delegate(agent: str, i: int) -> AIMessage:
    return tool_call("task", {"description": f"handle the issue ({agent})", "subagent_type": agent}, f"{agent}{i}")


SCRIPTS = {
    Role.ORCHESTRATOR: [delegate("repo_explorer", 1), delegate("coder", 2), delegate("pr_agent", 3),
                        AIMessage(content="VERDICT: RESOLVED - DRY RUN: PR not opened")],
    Role.REPO_EXPLORER: [AIMessage(content="app.py - the function")],
    Role.CODER: [tool_call("edit_file", {"file_path": "/app.py", "old_string": "return 1", "new_string": "return 1  # ok"}, "e1"),
                 delegate("test_agent", 4), AIMessage(content="Fixed; tests pass.")],
    Role.TESTER: [AIMessage(content="PASS - 1 passed")],
    Role.PR: [tool_call("open_pull_request", {"summary": "Annotate f", "what_changed": ["Comment in f"],
                                              "why": ["Clarity"], "approach": "Add a comment."}, "pr1"),
              AIMessage(content="DRY RUN: PR not opened")],
}


class FakeClassifier:
    def classify(self, title, body):
        return Classification(task_type=TaskType.BUG_FIX, confidence=0.9, rationale="test")


@pytest.fixture
def target(tmp_path, monkeypatch):
    repo = tmp_path / "target"
    (repo / "tests").mkdir(parents=True)
    (repo / "app.py").write_text("def f():\n    return 1\n")
    (repo / "tests" / "test_app.py").write_text("from app import f\n\ndef test_f():\n    assert f() == 1\n")
    for args in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    monkeypatch.setenv("CODEPILOT_TARGET_PATH", str(repo))
    monkeypatch.setenv("CODEPILOT_TARGET_PYTHON_BIN", str(Path(sys.executable).parent))
    monkeypatch.setattr(agents, "_model", lambda role: ScriptedModel(script=SCRIPTS[role]))
    monkeypatch.setattr(runner, "CACHE", tmp_path / "cache")
    return repo


def test_full_team_dry_run(target, tmp_path):
    registry, states, logs = TaskRegistry(tmp_path / "tasks.json"), [], []
    issue = SimpleNamespace(number=7, title="Annotate f", body="Please annotate f.",
                            html_url="https://example.com/7", user=SimpleNamespace(login="someone"))

    result = runner.run_issue(issue, runner.RunOptions(watchdog_seconds=None),
                              runner.Callbacks(log=logs.append, state=lambda r: states.append(r.state)),
                              registry, FakeClassifier())

    record = result.record
    assert [s for s, _ in record.history] == ["TRIAGED", "EXPLORING", "IMPLEMENTING", "TESTING", "DONE"]
    assert record.state == TaskState.DONE and record.outcome.startswith("DRY RUN")
    assert result.verdict == "VERDICT: RESOLVED - DRY RUN: PR not opened"
    assert registry.get(7).state == TaskState.DONE
    plan = (result.workspace.working / "pr_plan.md").read_text()
    assert "codepilot/issue-7-annotate-f" in plan and "fix(#7): Annotate f" in plan
    assert any("coder" in line and "edit_file" in line for line in logs)
