from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from codepilot.poller import PollerConfig, eligible_issues, poll_forever, skip_reason
from codepilot.tasks import TaskRecord, TaskRegistry, TaskState

CONFIG = PollerConfig(interval_minutes=5, label="ai-assignable", retry_failed_after_hours=24)


def issue(number, labels=("ai-assignable",), assignees=(), pull_request=None):
    return SimpleNamespace(number=number, labels=[SimpleNamespace(name=n) for n in labels],
                           assignees=list(assignees), pull_request=pull_request)


def finished(registry, number, state, hours_ago=0.0, pr_url=None, billing_blocked=False):
    r = TaskRecord(number, "t", "bug_fix", "bug_fix_skill", "runs/x")
    r.move(TaskState.IMPLEMENTING)
    r.move(state, "outcome")
    r.pr_url = pr_url
    r.billing_blocked = billing_blocked
    r.history[-1] = (state.value, (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat())
    registry.save(r)


def test_skip_reasons(tmp_path):
    registry = TaskRegistry(tmp_path / "tasks.json")
    registry.save(TaskRecord(3, "t", "bug_fix", "bug_fix_skill", "runs/x"))  # in progress
    finished(registry, 4, TaskState.DONE, pr_url="https://example.com/pr/4")
    finished(registry, 5, TaskState.FAILED, hours_ago=1)
    finished(registry, 6, TaskState.FAILED, hours_ago=48)

    assert skip_reason(issue(1, labels=()), registry, CONFIG) == "not labelled ai-assignable"
    assert skip_reason(issue(2, assignees=["dev"]), registry, CONFIG) == "assigned to a human"
    assert skip_reason(issue(7, pull_request=object()), registry, CONFIG) == "is a pull request"
    assert skip_reason(issue(3), registry, CONFIG).startswith("already in progress")
    assert skip_reason(issue(4), registry, CONFIG).startswith("already handled")
    assert skip_reason(issue(5), registry, CONFIG).startswith("failed recently")
    assert skip_reason(issue(6), registry, CONFIG) is None  # failed long enough ago to retry
    assert skip_reason(issue(8), registry, CONFIG) is None


def test_eligible_issues_are_sorted_and_logged(tmp_path):
    registry, skipped = TaskRegistry(tmp_path / "tasks.json"), []
    picked = eligible_issues([issue(9), issue(1, labels=()), issue(2)], registry, CONFIG, skipped.append)
    assert [i.number for i in picked] == [2, 9]
    assert skipped == ["skip #1: not labelled ai-assignable"]


def test_poll_once_runs_each_eligible_issue_one_at_a_time(tmp_path):
    registry, ran = TaskRegistry(tmp_path / "tasks.json"), []

    def run_one(i):
        ran.append(i.number)
        finished(registry, i.number, TaskState.DONE)

    poll_forever(lambda: [issue(2), issue(1)], run_one, registry, CONFIG, log=lambda s: None, once=True)
    assert ran == [1, 2]
    poll_forever(lambda: [issue(2), issue(1)], run_one, registry, CONFIG, log=lambda s: None, once=True)
    assert ran == [1, 2]  # second cycle: both already handled


def test_billing_failure_pauses_polling_and_keeps_the_issue_eligible(tmp_path):
    registry, ran, logs = TaskRegistry(tmp_path / "tasks.json"), [], []

    def run_one(i):
        ran.append(i.number)
        finished(registry, i.number, TaskState.FAILED, billing_blocked=True)

    poll_forever(lambda: [issue(1), issue(2)], run_one, registry, CONFIG, log=logs.append)  # not once: must return
    assert ran == [1]  # #2 would fail the same way, so it is not started
    assert logs[-1].startswith("poll: paused")
    assert skip_reason(issue(1), registry, CONFIG) is None  # retried as soon as polling resumes


def test_dry_run_results_are_previews_only_when_opening_prs(tmp_path):
    registry = TaskRegistry(tmp_path / "tasks.json")
    r = TaskRecord(3, "t", "bug_fix", "bug_fix_skill", "runs/x")
    r.move(TaskState.IMPLEMENTING)
    r.move(TaskState.DONE, "DRY RUN: PR not opened (run with --open-pr).")
    registry.save(r)
    finished(registry, 4, TaskState.DONE, pr_url="https://example.com/pr/4")
    opening = PollerConfig(interval_minutes=5, label="ai-assignable", retry_failed_after_hours=24, open_pr=True)

    assert skip_reason(issue(3), registry, CONFIG).startswith("already handled")  # no rerun on every dry-run poll
    assert skip_reason(issue(3), registry, opening) is None
    assert skip_reason(issue(4), registry, opening).startswith("already handled")  # a real PR is never redone
