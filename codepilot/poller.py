"""Issue polling: find open issues CodePilot may take on, and work through them on a schedule."""

import os
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from codepilot.tasks import TERMINAL, TRIAGE_STOPS, TaskRegistry, TaskState

DEFAULT_LABEL = "ai-assignable"


@dataclass
class PollerConfig:
    interval_minutes: float = float(os.getenv("CODEPILOT_POLL_MINUTES", "5"))
    label: str = os.getenv("CODEPILOT_ISSUE_LABEL", DEFAULT_LABEL)
    # Unassigned issues without the label are taken too, under the stricter unlabelled complexity limit.
    take_unlabelled: bool = os.getenv("CODEPILOT_TAKE_UNLABELLED", "true").lower() == "true"
    retry_failed_after_hours: float = float(os.getenv("CODEPILOT_RETRY_FAILED_AFTER_HOURS", "24"))
    open_pr: bool = False  # when PRs are opened, a dry-run result is only a preview and the issue is taken again


def opted_in(issue, label: str = DEFAULT_LABEL) -> bool:
    """Someone asked CodePilot to handle this: it carries the label, or it was typed into the TUI."""
    return getattr(issue, "local", False) or label in {lbl.name for lbl in getattr(issue, "labels", [])}


def skip_reason(issue, registry: TaskRegistry, config: PollerConfig) -> str | None:
    """Why an open issue is not eligible right now, or None if CodePilot may take it."""
    if getattr(issue, "pull_request", None):
        return "is a pull request"
    if issue.assignees:
        return "assigned to a human"
    if not config.take_unlabelled and not opted_in(issue, config.label):
        return f"not labelled {config.label}"
    record = registry.get(issue.number)
    if record is None:
        return None
    if record.state not in TERMINAL:
        return f"already in progress ({record.state})"
    if record.state == TaskState.DONE:
        if config.open_pr and record.dry_run:
            return None
        return f"already handled ({record.pr_url or record.outcome})"
    if record.state in TRIAGE_STOPS:
        if config.open_pr and not record.notified:
            return None  # decided in a dry run: the reporter was never told
        updated = getattr(issue, "updated_at", None)
        if updated and record.issue_updated_at and updated > datetime.fromisoformat(record.issue_updated_at):
            return None  # edited or answered since triage: look again
        return f"{record.state.lower().replace('_', ' ')} ({record.outcome[:80]}); waiting for the issue to change"
    if record.billing_blocked or record.transient:
        return None  # never really attempted; take it again as soon as polling resumes
    failed_at = datetime.fromisoformat(record.history[-1][1])
    if datetime.now(timezone.utc) - failed_at < timedelta(hours=config.retry_failed_after_hours):
        return f"failed recently ({record.outcome[:80]})"
    return None


def eligible_issues(issues: Iterable, registry: TaskRegistry, config: PollerConfig, log=lambda s: None) -> list:
    picked = []
    for issue in issues:
        reason = skip_reason(issue, registry, config)
        if reason:
            log(f"skip #{issue.number}: {reason}")
        else:
            picked.append(issue)
    return sorted(picked, key=lambda i: i.number)


def billing_paused(issue, registry: TaskRegistry, log: Callable[[str], None]) -> bool:
    """After a task, whether it failed on provider billing; every later task would fail the same way."""
    record = registry.get(issue.number)
    if record is None or not record.billing_blocked:
        return False
    log(f"poll: paused - the model provider refused calls for billing on #{issue.number}. "
        "Add credits, then resume; the issue will be retried.")
    return True


def poll_forever(
    list_issues: Callable[[], Iterable],
    run_one: Callable[[object], None],
    registry: TaskRegistry,
    config: PollerConfig,
    log: Callable[[str], None] = print,
    stop: threading.Event | None = None,
    once: bool = False,
) -> None:
    """Each cycle: list open issues, run the eligible ones one at a time, then sleep until the next cycle."""
    stop = stop or threading.Event()
    while not stop.is_set():
        log(f"poll: checking open issues labelled {config.label}"
            + (" and other unassigned issues" if config.take_unlabelled else ""))
        for issue in eligible_issues(list_issues(), registry, config, log):
            if stop.is_set():
                break
            if skip_reason(issue, registry, config):  # re-check: a long task may have changed the registry
                continue
            run_one(issue)
            if billing_paused(issue, registry, log):
                return
        if once:
            return
        log(f"poll: next check in {config.interval_minutes:g} min")
        stop.wait(config.interval_minutes * 60)
