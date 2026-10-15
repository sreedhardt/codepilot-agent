from codepilot.tasks import TaskRecord, TaskRegistry, TaskState, TaskTracker


def record(**kw) -> TaskRecord:
    return TaskRecord(issue_number=2, title="t", task_type="bug_fix", skill="bug_fix_skill", workspace="runs/x", **kw)


def delegate(agent: str) -> list[dict]:
    return [{"name": "task", "args": {"subagent_type": agent}}]


def states(r: TaskRecord) -> list[str]:
    return [s for s, _ in r.history]


def test_happy_path_follows_delegations_and_pr_outcome(tmp_path):
    changes = []
    tracker = TaskTracker(record(), on_change=lambda r: changes.append(r.state))
    tracker.observe("orchestrator", delegate("repo_explorer"))
    tracker.observe("orchestrator", delegate("coder"))
    tracker.observe("coder", delegate("test_agent"))
    (tmp_path / "pr_result.md").write_text("PR_OPENED https://github.com/o/r/pull/9\n")

    r = tracker.finish(tmp_path)

    assert states(r) == ["TRIAGED", "EXPLORING", "IMPLEMENTING", "TESTING", "PR_OPENED", "DONE"]
    assert r.pr_url == "https://github.com/o/r/pull/9"
    assert changes[-1] == TaskState.DONE


def test_fix_after_failed_verification_goes_back_to_implementing():
    tracker = TaskTracker(record())
    tracker.observe("orchestrator", delegate("coder"))
    tracker.observe("coder", delegate("test_agent"))
    tracker.observe("coder", [{"name": "edit_file", "args": {}}])
    tracker.observe("coder", delegate("test_agent"))
    assert states(tracker.record)[-3:] == ["TESTING", "IMPLEMENTING", "TESTING"]
    assert tracker.record.retries_after_fail == 1


def test_dry_run_is_done_without_pr_opened(tmp_path):
    (tmp_path / "pr_result.md").write_text("DRY RUN: PR not opened\n")
    tracker = TaskTracker(record())
    tracker.observe("orchestrator", delegate("coder"))
    r = tracker.finish(tmp_path)
    assert r.state == TaskState.DONE and "PR_OPENED" not in states(r)


def test_rejected_or_missing_pr_means_failed(tmp_path):
    (tmp_path / "pr_result.md").write_text("REJECTED by human: PR not opened.\n")
    assert TaskTracker(record()).finish(tmp_path).state == TaskState.FAILED
    assert TaskTracker(record()).finish(tmp_path / "nowhere").outcome == "no pull request was attempted"


def test_error_fails_from_any_state(tmp_path):
    tracker = TaskTracker(record())
    tracker.observe("orchestrator", delegate("coder"))
    r = tracker.finish(tmp_path, error="ReadTimeout: stalled")
    assert r.state == TaskState.FAILED and r.outcome == "ReadTimeout: stalled"


def test_illegal_transitions_are_ignored():
    r = record()
    assert not r.move(TaskState.PR_OPENED)
    assert r.move(TaskState.FAILED)
    assert not r.move(TaskState.EXPLORING)


def test_registry_round_trip_and_in_progress(tmp_path):
    path = tmp_path / "tasks.json"
    registry = TaskRegistry(path)
    active, done = record(), TaskRecord(5, "d", "documentation", "documentation_skill", "runs/y")
    done.move(TaskState.IMPLEMENTING)
    done.move(TaskState.DONE)
    registry.save(active)
    registry.save(done)

    reloaded = TaskRegistry(path)
    assert reloaded.in_progress() == {2}
    assert reloaded.get(5).state == TaskState.DONE
    assert states(reloaded.get(5)) == ["TRIAGED", "IMPLEMENTING", "DONE"]


def test_unfinished_tasks_of_exited_processes_are_failed_on_startup(tmp_path):
    import subprocess
    import sys

    exited = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    registry = TaskRegistry(tmp_path / "tasks.json")
    orphan, live, done = record(pid=int(exited.stdout)), record(), record()
    orphan.move(TaskState.EXPLORING)
    orphan.move(TaskState.IMPLEMENTING)
    orphan.active_agent = "coder"
    live.issue_number, done.issue_number = 3, 4
    done.move(TaskState.FAILED, "earlier failure")
    for r in (orphan, live, done):
        registry.save(r)

    reloaded = TaskRegistry(tmp_path / "tasks.json")
    assert [r.issue_number for r in reloaded.recover_stale()] == [2]
    reloaded = TaskRegistry(tmp_path / "tasks.json")  # the recovery was persisted
    assert reloaded.get(2).state == TaskState.FAILED and reloaded.get(2).active_agent == ""
    assert reloaded.get(2).outcome == "interrupted: CodePilot stopped while the task was IMPLEMENTING"
    assert reloaded.get(3).state == TaskState.TRIAGED  # owned by this live process: left alone
    assert reloaded.get(4).outcome == "earlier failure"


def test_records_written_before_pids_count_as_orphaned(tmp_path):
    import json

    (tmp_path / "tasks.json").write_text(json.dumps({"2": {
        "issue_number": 2, "title": "t", "task_type": "bug_fix", "skill": "bug_fix_skill", "workspace": "runs/x",
        "state": "TESTING", "history": [["TRIAGED", "2026-10-10T00:00:00+00:00"]]}}))
    assert [r.issue_number for r in TaskRegistry(tmp_path / "tasks.json").recover_stale()] == [2]
