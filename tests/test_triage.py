from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from codepilot import runner
from codepilot.classifier import Classification, TaskType
from codepilot.poller import PollerConfig, skip_reason
from codepilot.tasks import TaskRegistry, TaskState
from codepilot.triage import triage


def assessed(decision="accept", complexity=2, confidence=0.9, questions=()):
    return Classification(task_type=TaskType.BUG_FIX, confidence=confidence, rationale="r", decision=decision,
                          complexity=complexity, triage_reason="because", questions=list(questions))


def test_policy_proceeds_only_on_confident_accepted_small_work():
    assert triage(assessed()).proceed
    assert triage(assessed(complexity=4)).proceed
    declined = triage(assessed(decision="reject"))
    assert (declined.proceed, declined.state) == (False, TaskState.DECLINED)
    clarify = triage(assessed(decision="clarify", questions=["Which endpoint?", "What did you expect?"]))
    assert clarify.state == TaskState.NEEDS_INFO and "- Which endpoint?\n- What did you expect?" in clarify.comment
    assert triage(assessed(complexity=5)).state == TaskState.DECLINED  # accepted by the model, too big for policy
    assert triage(assessed(confidence=0.4)).state == TaskState.NEEDS_INFO
    assert "codepilot-needs-info" in clarify.comment


class FakeIssue(SimpleNamespace):
    """The parts of a PyGithub Issue that triage touches."""

    def create_comment(self, body):
        self.comments.append(body)

    def add_to_labels(self, *names):
        self.labels += [SimpleNamespace(name=n) for n in names]

    def remove_from_labels(self, name):
        self.labels = [lbl for lbl in self.labels if lbl.name != name]

    def update(self):
        self.updated_at = self.github_clock  # GitHub stamps its own time on our writes


T0 = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)
CONFIG = PollerConfig(interval_minutes=5, label="ai-assignable", retry_failed_after_hours=24)


def issue():
    return FakeIssue(number=9, title="It's broken", body="fix it", html_url="u", user=None, assignees=[],
                     pull_request=None, labels=[SimpleNamespace(name="ai-assignable")], comments=[],
                     updated_at=T0, github_clock=T0 + timedelta(seconds=5))


def run(i, tmp_path, classification, open_pr):
    classifier = SimpleNamespace(classify=lambda title, body: classification)
    registry = TaskRegistry(tmp_path / "tasks.json")
    result = runner.run_issue(i, runner.RunOptions(open_pr=open_pr, runs_dir=tmp_path / "runs"),
                              runner.Callbacks(), registry, classifier)
    return result, registry


def test_dry_run_stop_posts_nothing_and_is_redone_when_prs_are_enabled(tmp_path):
    i = issue()
    result, registry = run(i, tmp_path, assessed(decision="clarify", questions=["Which page?"]), open_pr=False)
    assert result.workspace is None and result.record.state == TaskState.NEEDS_INFO
    assert i.comments == [] and [lbl.name for lbl in i.labels] == ["ai-assignable"]
    assert skip_reason(i, registry, CONFIG).startswith("needs info")
    assert skip_reason(i, registry, PollerConfig(open_pr=True, label="ai-assignable")) is None


def test_stop_comments_labels_and_waits_for_the_reporter(tmp_path):
    i = issue()
    _, registry = run(i, tmp_path, assessed(decision="clarify", questions=["Which page?"]), open_pr=True)
    assert "- Which page?" in i.comments[0]
    assert "codepilot-needs-info" in {lbl.name for lbl in i.labels}
    opening = PollerConfig(open_pr=True, label="ai-assignable")
    assert skip_reason(i, registry, opening).startswith("needs info")  # our own comment is not a reply
    i.updated_at = T0 + timedelta(minutes=30)  # the reporter answered
    assert skip_reason(i, registry, opening) is None


def test_accepting_later_clears_the_old_triage_label(tmp_path, monkeypatch):
    class StopHere(Exception):
        pass

    def no_workspace(*args, **kwargs):
        raise StopHere  # the agents are covered elsewhere; this test ends where they would start

    i = issue()
    i.labels.append(SimpleNamespace(name="codepilot-needs-info"))
    monkeypatch.setattr(runner, "TaskWorkspace", no_workspace)
    with pytest.raises(StopHere):
        run(i, tmp_path, assessed(), open_pr=True)
    assert [lbl.name for lbl in i.labels] == ["ai-assignable"]
