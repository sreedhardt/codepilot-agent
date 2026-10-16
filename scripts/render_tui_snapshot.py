"""Render the TUI as an SVG from a recorded evaluation run, for the README.

    uv run python scripts/render_tui_snapshot.py evals/results/suite_20261016_095128 docs/tui.svg

Nothing runs: the app is driven headless with the run's own task records and agent log, and the approval panel
shows the gate the selected issue would raise with --open-pr. Only recorded data is shown.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codepilot.gates import Gate  # noqa: E402
from codepilot.poller import PollerConfig  # noqa: E402
from codepilot.tasks import TaskRegistry  # noqa: E402
from codepilot.tui import CodePilotApp, PendingApproval  # noqa: E402


async def render(suite: Path, out: Path, active: int) -> None:
    registry = TaskRegistry(suite / "tasks.json")
    issues = [SimpleNamespace(number=n, title=r.title, labels=[SimpleNamespace(name="ai-assignable")], assignees=[],
                              pull_request=None, updated_at=None) for n, r in sorted(registry.records.items())]
    app = CodePilotApp(lambda: issues, lambda *a: None, registry, PollerConfig(interval_minutes=5), open_pr=True)
    record = registry.get(active)
    log = (suite / "agent_log.txt").read_text().splitlines()
    start = next(i for i, line in enumerate(log) if line.startswith(f"#{active} "))
    workspace = next((suite / "runs").glob(f"issue-{active}-*"))
    files = len((workspace / "working" / "proposed_diff.txt").read_text().split("diff --git")) - 1
    request = {"gates": [Gate.PR_TO_DEFAULT_BRANCH.value],
               "summary": f"Open PR \"[CodePilot] {record.title}\" from codepilot/issue-{active}-... into main "
                          f"({files} files)?",
               "details": {}}
    async with app.run_test(size=(170, 46)) as pilot:
        await pilot.pause()
        app.show_task(record)
        for line in log[start:start + 60]:
            app.log_line(line)
        app.show_approval(PendingApproval(request, None, asyncio.Event()))
        app.query_one("#issues").focus()  # so the footer lists the app's keys, not the input's
        await pilot.pause()
        out.parent.mkdir(parents=True, exist_ok=True)
        app.save_screenshot(filename=out.name, path=str(out.parent))
        app.stop.set()
        app.poll_now.set()


if __name__ == "__main__":
    asyncio.run(render(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 6))
