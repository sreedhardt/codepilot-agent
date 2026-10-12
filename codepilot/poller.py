"""Issue polling: find open issues CodePilot may take on, and work through them on a schedule."""

import os
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from codepilot.tasks import TERMINAL, TaskRegistry, TaskState

DEFAULT_LABEL = "ai-assignable"


@dataclass
class PollerConfig:
    interval_minutes: float = float(os.getenv("CODEPILOT_POLL_MINUTES", "5"))
    label: str = os.getenv("CODEPILOT_ISSUE_LABEL", DEFAULT_LABEL)
    retry_failed_after_hours: float = float(os.getenv("CODEPILOT_RETRY_FAILED_AFTER_HOURS", "24"))


def skip_reason(issue, registry: TaskRegistry, config: PollerConfig) -> str | None:
    """Why an open issue is not eligible right now, or None if CodePilot may take it."""
    if getattr(issue, "pull_request", None):
        return "is a pull request"
    if issue.assignees:
        return "assigned to a human"
    if config.label not in {label.name for label in issue.labels}:
        return f"not labelled {config.label}"
    record = registry.get(issue.number)
    if record is None:
        return None
    if record.state not in TERMINAL:
        return f"already in progress ({record.state})"
    if record.state == TaskState.DONE:
        return f"already handled ({record.pr_url or record.outcome})"
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
        log(f"poll: checking open issues labelled {config.label}")
        for issue in eligible_issues(list_issues(), registry, config, log):
            if stop.is_set():
                break
            if skip_reason(issue, registry, config):  # re-check: a long task may have changed the registry
                continue
            run_one(issue)
        if once:
            return
        log(f"poll: next check in {config.interval_minutes:g} min")
        stop.wait(config.interval_minutes * 60)
