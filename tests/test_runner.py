"""End-to-end runner test with scripted models: the full agent team, no API calls."""

import subprocess
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
    monkeypatch.setattr(agents, "_model", lambda role: ScriptedModel(script=SCRIPTS[role]))
    monkeypatch.setattr(runner, "CACHE", tmp_path / "cache")
    return repo


def test_full_team_dry_run(target, tmp_path):
    registry, states, logs = TaskRegistry(tmp_path / "tasks.json"), [], []
    issue = SimpleNamespace(number=7, title="Annotate f", body="Please annotate f.",
                            html_url="https://example.com/7", user=SimpleNamespace(login="someone"))

    result = runner.run_issue(issue, runner.RunOptions(watchdog_seconds=None, runs_dir=tmp_path / "runs"),
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


def test_full_team_run_uses_and_records_memory(target, tmp_path):
    from codepilot.memory.episodic import EpisodicMemory, open_store
    from codepilot.memory.semantic import SemanticMemory

    memory = runner.Memory(EpisodicMemory(open_store(tmp_path / "e.sqlite"), "o/r"), SemanticMemory(tmp_path / "l", "o/r"))
    memory.episodic.start_session()
    issue = SimpleNamespace(number=7, title="Annotate f", body="Please annotate f.",
                            html_url="https://example.com/7", user=SimpleNamespace(login="someone"))

    result = runner.run_issue(issue, runner.RunOptions(watchdog_seconds=None, runs_dir=tmp_path / "runs"),
                              runner.Callbacks(), TaskRegistry(tmp_path / "tasks.json"), FakeClassifier(), memory)

    working = result.workspace.working
    assert (working / "recent_sessions.md").exists() and (working / "lessons.md").exists()
    assert (working / "working_memory.final.json").exists() and not (working / "working_memory.json").exists()
    assert [t["issue_number"] for t in memory.episodic.tasks] == [7]
    assert memory.episodic.tasks[0]["files_modified"] == ["app.py"]
    assert result.lesson is None  # dry run: no PR, so nothing to learn yet


def test_budget_breach_fails_the_task_with_a_clear_reason(target, tmp_path):
    issue = SimpleNamespace(number=7, title="Annotate f", body="Please annotate f.",
                            html_url="https://example.com/7", user=SimpleNamespace(login="someone"))
    result = runner.run_issue(issue, runner.RunOptions(watchdog_seconds=None, runs_dir=tmp_path / "runs", max_minutes=0),
                              runner.Callbacks(), TaskRegistry(tmp_path / "tasks.json"), FakeClassifier())
    assert result.record.state == TaskState.FAILED
    assert "deadline" in result.record.outcome


def test_step_limits_outlast_model_call_caps(target, tmp_path):
    from codepilot.agents import MODEL_CALL_LIMITS, TaskContext, build_team
    from codepilot.repo_map import build_repo_map
    from codepilot.workspace import TaskWorkspace

    ws = TaskWorkspace(1, runs_dir=tmp_path / "runs")
    team = build_team(ws, build_repo_map(ws.sandbox), TaskContext(1, "t", "u", None, TaskType.BUG_FIX))
    roles = {"repo_explorer": Role.REPO_EXPLORER, "coder": Role.CODER, "pr_agent": Role.PR}
    for sub in team:
        runnable = sub["runnable"]
        loop = [n for n in runnable.get_graph().nodes if not n.startswith("__") and not n.endswith(".before_agent")]
        # a run that uses its whole model-call cap must end by the cap, not by GraphRecursionError
        assert runnable.config["recursion_limit"] > MODEL_CALL_LIMITS[roles[sub["name"]]] * len(loop)


BILLING = ("Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': "
           "'Your credit balance is too low to access the Anthropic API.'}}")


class OutOfCreditModel(ScriptedModel):
    def _generate(self, *args, **kwargs):
        from langchain_core.exceptions import ModelInvalidRequestError
        raise ModelInvalidRequestError(BILLING)


def test_retry_predicate_retries_only_transient_failures():
    from langchain_core.exceptions import ModelInvalidRequestError, ModelRateLimitError

    should_retry, before = agents._retry_on(Role.CODER), len(agents.RETRY_LOG)
    assert not should_retry(ModelInvalidRequestError(BILLING))
    assert not should_retry(RuntimeError(BILLING))  # an unwrapped SDK error is still recognised
    assert len(agents.RETRY_LOG) == before  # refused calls are not reported as retries
    assert should_retry(ModelRateLimitError("Error code: 429 - rate limit exceeded"))
    assert len(agents.RETRY_LOG) == before + 1


def test_billing_failure_is_flagged_on_the_record(target, tmp_path, monkeypatch):
    scripted = agents._model
    monkeypatch.setattr(agents, "_model", lambda role: OutOfCreditModel(script=SCRIPTS[role]) if role == Role.CODER
                        else scripted(role))
    registry = TaskRegistry(tmp_path / "tasks.json")
    issue = SimpleNamespace(number=7, title="Annotate f", body="Please annotate f.",
                            html_url="https://example.com/7", user=SimpleNamespace(login="someone"))
    result = runner.run_issue(issue, runner.RunOptions(watchdog_seconds=None, runs_dir=tmp_path / "runs"),
                              runner.Callbacks(), registry, FakeClassifier())
    assert result.record.state == TaskState.FAILED and "credit balance" in result.record.outcome
    assert registry.get(7).billing_blocked
    assert result.retries == []
