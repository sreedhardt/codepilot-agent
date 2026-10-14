"""Terminal UI: issue feed, active task, streaming agent logs and the human approval panel.

The TUI is a front end to the same runner the CLI uses. Polling and agent runs
happen on a background thread; runner callbacks update the panels through
`call_from_thread`. When an approval gate fires, the background thread waits
until the human answers in the approval panel.

    uv run python -m codepilot tui [--open-pr] [--poll-minutes N]
"""

import queue
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import SimpleNamespace

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Input, Label, ListItem, ListView, RichLog, Static, TextArea

from codepilot.poller import PollerConfig, skip_reason
from codepilot.runner import Callbacks, RunResult
from codepilot.tasks import TERMINAL, TaskRecord, TaskRegistry, TaskState
from codepilot.workspace import TaskWorkspace

LOCAL_TASK_START = 10_000  # numbers for free-form tasks typed in the TUI, clear of GitHub issue numbers

STATE_ICONS = {
    TaskState.DONE: "✓", TaskState.FAILED: "✗", TaskState.PR_OPENED: "↑",
    TaskState.TRIAGED: "⟳", TaskState.EXPLORING: "⟳", TaskState.IMPLEMENTING: "⟳", TaskState.TESTING: "⟳",
}


@dataclass
class PendingApproval:
    request: dict
    ws: TaskWorkspace
    answered: threading.Event
    reply: dict | None = None


class NewTaskScreen(ModalScreen[str | None]):
    """Free-form coding task, not tied to a GitHub issue."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("New task: describe the change (first line becomes the title). Enter to start, Esc to cancel.")
            yield Input(placeholder="e.g. Add a /stats endpoint returning the number of active loans", id="task-input")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class InspectScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape", "close", "Close"), Binding("q", "close", "Close")]

    def __init__(self, title: str, content: str) -> None:
        super().__init__()
        self.title_text, self.content = title, content

    def compose(self) -> ComposeResult:
        with Vertical(id="inspect"):
            yield Label(f"{self.title_text}  (Esc to close)")
            yield TextArea(self.content, read_only=True, id="inspect-text")

    def action_close(self) -> None:
        self.dismiss(None)


class CodePilotApp(App):
    TITLE = "CodePilot"
    CSS = """
    Grid { grid-size: 2 2; grid-columns: 1fr 1fr; grid-rows: 1fr 1fr; }
    .panel { border: round $accent; padding: 0 1; }
    #issues, #task, #logs, #approval { height: 100%; }
    #approval-input { dock: bottom; }
    #dialog, #inspect { width: 90%; height: auto; max-height: 90%; border: thick $accent; background: $surface;
                        padding: 1 2; align: center middle; }
    #inspect-text { height: 30; }
    NewTaskScreen, InspectScreen { align: center middle; }
    """
    BINDINGS = [
        Binding("i", "new_task", "New task"),
        Binding("s", "skip_issue", "Skip issue"),
        Binding("p", "poll_now", "Poll now"),
        Binding("l", "toggle_logs", "Logs"),
        Binding("q", "quit_app", "Quit"),
    ]

    def __init__(
        self,
        list_issues: Callable[[], Iterable],
        run_issue: Callable[[object, Callbacks, bool], RunResult],
        registry: TaskRegistry,
        config: PollerConfig,
        open_pr: bool = False,
        on_exit: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self.list_issues, self.run_issue, self.registry, self.config = list_issues, run_issue, registry, config
        self.open_pr, self.on_exit_hook = open_pr, on_exit
        self.issues: list = []
        self.skipped: set[int] = set()
        self.local_tasks: queue.Queue = queue.Queue()
        self.next_local = LOCAL_TASK_START
        self.pending: PendingApproval | None = None
        self.stop = threading.Event()
        self.poll_now = threading.Event()
        self.current: TaskRecord | None = None
        self.todos: tuple[str, list[dict]] | None = None

    # ----- layout -------------------------------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        with Grid():
            yield ListView(id="issues", classes="panel")
            yield Static("No active task.", id="task", classes="panel")
            yield RichLog(id="logs", classes="panel", wrap=True, min_width=20, markup=False, max_lines=5000)
            with Vertical(id="approval", classes="panel"):
                yield Static("No approval needed.", id="approval-text")
                yield Input(placeholder="approve / reject / inspect", id="approval-input", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#issues").border_title = "GitHub Issues"
        self.query_one("#task").border_title = "Active Task"
        self.query_one("#logs").border_title = "Agent Logs"
        self.query_one("#approval").border_title = "Human Approval"
        mode = "opens PRs (with approval)" if self.open_pr else "dry run (no PRs)"
        self.log_line(f"CodePilot started: polling every {self.config.interval_minutes:g} min, {mode}.")
        threading.Thread(target=self._worker, name="codepilot-worker", daemon=True).start()

    # ----- background worker ----------------------------------------------------------------------------
    def _ui(self, fn, *args) -> None:
        """Run a UI update from the worker thread; a no-op once the app is shutting down."""
        if self.stop.is_set() or not self.is_running:
            return
        try:
            self.call_from_thread(fn, *args)
        except Exception:  # noqa: BLE001 - the app closed between the check and the call
            pass

    def _worker(self) -> None:
        while not self.stop.is_set():
            self._drain_local_tasks()
            try:
                self.issues = sorted(self.list_issues(), key=lambda i: i.number)
            except Exception as e:  # noqa: BLE001 - keep polling through network errors
                self._ui(self.log_line, f"poll failed: {type(e).__name__}: {e}")
                self.issues = []
            self._ui(self.refresh_issues)
            for issue in self.issues:
                if self.stop.is_set():
                    return
                self._drain_local_tasks()
                if issue.number in self.skipped or skip_reason(issue, self.registry, self.config):
                    continue
                self._run(issue, self.open_pr)
            self._ui(self.log_line, f"poll: next check in {self.config.interval_minutes:g} min "
                                                 "(press p to poll now)")
            self.poll_now.wait(self.config.interval_minutes * 60)
            self.poll_now.clear()

    def _drain_local_tasks(self) -> None:
        while not self.local_tasks.empty() and not self.stop.is_set():
            self._run(self.local_tasks.get(), open_pr=False)  # a local task has no issue for a PR to close

    def _run(self, issue, open_pr: bool) -> None:
        self.todos = None
        callbacks = Callbacks(
            log=lambda line: self._ui(self.log_line, line),
            state=lambda record: self._ui(self.show_task, record),
            approve=self._wait_for_approval,
            todos=lambda agent, todos: self._ui(self.show_todos, agent, todos),
        )
        try:
            result = self.run_issue(issue, callbacks, open_pr)
            self._ui(self.log_line, f"{result.verdict} ({result.seconds:.0f}s)")
        except Exception as e:  # noqa: BLE001
            self._ui(self.log_line, f"task #{issue.number} crashed: {type(e).__name__}: {e}")
        self._ui(self.refresh_issues)

    def _wait_for_approval(self, request: dict, ws: TaskWorkspace) -> dict:
        pending = PendingApproval(request, ws, threading.Event())
        self.pending = pending
        self._ui(self.show_approval, pending)
        while not pending.answered.wait(0.5):
            if self.stop.is_set():
                return {"approved": False, "note": "CodePilot was closed while waiting"}
        self.pending = None
        self._ui(self.clear_approval)
        return pending.reply or {"approved": False}

    # ----- panel updates (UI thread) ----------------------------------------------------------------------
    def log_line(self, line: str) -> None:
        self.query_one("#logs", RichLog).write(line)

    def refresh_issues(self) -> None:
        view = self.query_one("#issues", ListView)
        index = view.index
        view.clear()
        for issue in self.issues:
            view.append(ListItem(Label(self._issue_label(issue)), name=str(issue.number)))
        if self.issues:
            view.index = min(index or 0, len(self.issues) - 1)

    def _issue_label(self, issue) -> Text:
        record = self.registry.get(issue.number)
        title = issue.title if len(issue.title) <= 42 else issue.title[:41] + "…"
        if issue.number in self.skipped:
            return Text(f"#{issue.number} ⏭ skipped  {title}", style="dim")
        if record is None:
            reason = skip_reason(issue, self.registry, self.config)
            return Text(f"#{issue.number} ● {'open' if not reason else 'not eligible'}  {title}",
                        style="" if not reason else "dim")
        icon = STATE_ICONS.get(record.state, "?")
        detail = f"PR #{record.pr_url.rsplit('/', 1)[1]}" if record.pr_url else record.state.value
        style = {TaskState.DONE: "green", TaskState.FAILED: "red"}.get(record.state, "yellow")
        return Text(f"#{issue.number} {icon} {detail}  {title}", style=style)

    def show_task(self, record: TaskRecord) -> None:
        self.current = record
        self._render_task()
        self.refresh_issues()

    def show_todos(self, agent: str, todos: list[dict]) -> None:
        self.todos = (agent, todos)
        self._render_task()

    def _render_task(self) -> None:
        r = self.current
        if r is None:
            return
        lines = [f"#{r.issue_number}: {r.title}",
                 f"Status: {r.state}" + ("" if r.state in TERMINAL else f"   Agent: {r.active_agent or '…'}"),
                 f"Skill: {r.skill}   Type: {r.task_type}",
                 f"Fix rounds after a failed test run: {r.retries_after_fail}"]
        if r.pr_url:
            lines.append(f"PR: {r.pr_url}")
        if r.state in TERMINAL and r.outcome:
            lines.append(f"Outcome: {r.outcome[:160]}")
        if self.todos:
            agent, todos = self.todos
            lines.append(f"\nTodo ({agent}):")
            marks = {"completed": "[✓]", "in_progress": "[…]", "pending": "[ ]"}
            lines += [f"  {marks.get(t.get('status'), '[ ]')} {t.get('content', '')}" for t in todos[:12]]
        self.query_one("#task", Static).update("\n".join(lines))

    def show_approval(self, pending: PendingApproval) -> None:
        gates = ", ".join(pending.request["gates"])
        self.query_one("#approval-text", Static).update(
            Text(f"⚠ APPROVAL NEEDED [{gates}]\n\n{pending.request['summary']}\n\n"
                 "Type approve, reject (optionally: reject <reason>) or inspect, then Enter.", style="bold yellow"))
        box = self.query_one("#approval-input", Input)
        box.disabled = False
        box.focus()
        self.bell()

    def clear_approval(self) -> None:
        self.query_one("#approval-text", Static).update("No approval needed.")
        box = self.query_one("#approval-input", Input)
        box.value, box.disabled = "", True

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "approval-input" or self.pending is None:
            return
        command, _, note = event.value.strip().partition(" ")
        event.input.value = ""
        if command in ("approve", "a"):
            self._answer({"approved": True})
        elif command in ("reject", "r"):
            self._answer({"approved": False, "note": note.strip()})
        elif command in ("inspect", "i"):
            self.push_screen(InspectScreen("Inspect", self._inspection_text(self.pending)))
        else:
            self.log_line(f"Unknown answer {event.value!r}: type approve, reject or inspect.")

    def _answer(self, reply: dict) -> None:
        pending = self.pending
        if pending is None:
            return
        pending.reply = reply
        self.log_line(f"human: {'approved' if reply['approved'] else 'rejected'} {pending.request['gates']}")
        pending.answered.set()

    @staticmethod
    def _inspection_text(pending: PendingApproval) -> str:
        parts = []
        for key, value in pending.request["details"].items():
            if isinstance(value, str) and value.startswith("/working/"):
                path = pending.ws.working / value.removeprefix("/working/")
                parts.append(f"===== {value} =====\n{path.read_text() if path.exists() else '(not written)'}")
            else:
                parts.append(f"{key}: {value}")
        return "\n\n".join(parts)

    # ----- key actions ------------------------------------------------------------------------------------
    def action_new_task(self) -> None:
        def queue_task(text: str | None) -> None:
            if not text:
                return
            title, _, body = text.partition("\n")
            task = SimpleNamespace(number=self.next_local, title=title[:120], body=body or title, html_url="",
                                   user=None, labels=[], assignees=[], pull_request=None)
            self.next_local += 1
            self.local_tasks.put(task)
            self.log_line(f"queued local task #{task.number}: {task.title} (dry run: no issue for a PR to close)")
            self.poll_now.set()

        self.push_screen(NewTaskScreen(), queue_task)

    def action_skip_issue(self) -> None:
        view = self.query_one("#issues", ListView)
        if view.highlighted_child is None or view.highlighted_child.name is None:
            return
        number = int(view.highlighted_child.name)
        self.skipped ^= {number}  # toggle
        self.log_line(f"issue #{number} {'skipped for this session' if number in self.skipped else 'un-skipped'}")
        self.refresh_issues()

    def action_poll_now(self) -> None:
        self.poll_now.set()

    def action_toggle_logs(self) -> None:
        logs = self.query_one("#logs")
        if self.screen.maximized is logs:
            self.screen.minimize()
        else:
            self.screen.maximize(logs)

    def action_quit_app(self) -> None:
        self.stop.set()
        self.poll_now.set()
        if self.on_exit_hook:
            self.on_exit_hook()
        self.exit()
