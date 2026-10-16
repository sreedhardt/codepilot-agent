"""Triage policy: whether CodePilot works on an issue, decided in code from the classifier's assessment.

The model proposes (decision, complexity, confidence); this policy disposes. An issue proceeds only when it
is accepted, small enough to automate and classified with confidence; otherwise the task ends before any
workspace or agent exists, and the reporter is told why.

Issues someone opted in (labelled for CodePilot, or typed into the TUI) may be larger than unlabelled issues
CodePilot picked up on its own, and only opted-in issues get a comment: CodePilot does not post on issues
nobody asked it to handle.
"""

import os
from dataclasses import dataclass

from codepilot.classifier import Classification
from codepilot.tasks import TaskState

MAX_COMPLEXITY = int(os.getenv("CODEPILOT_MAX_COMPLEXITY", "8"))  # 1-10, for opted-in issues
MAX_UNLABELLED_COMPLEXITY = int(os.getenv("CODEPILOT_MAX_UNLABELLED_COMPLEXITY", "4"))
MIN_CONFIDENCE = float(os.getenv("CODEPILOT_MIN_CONFIDENCE", "0.6"))
LABELS = {TaskState.NEEDS_INFO: "codepilot-needs-info", TaskState.DECLINED: "codepilot-declined"}


@dataclass
class Triage:
    proceed: bool
    state: TaskState  # TRIAGED when proceeding, otherwise the terminal state that ends the task
    reason: str
    comment: str = ""  # what to tell the reporter when not proceeding


def triage(c: Classification, opted_in: bool = True) -> Triage:
    decision = _triage(c, MAX_COMPLEXITY if opted_in else MAX_UNLABELLED_COMPLEXITY)
    if not opted_in:
        decision.comment = ""  # never post on an issue nobody asked CodePilot to handle
    return decision


def _triage(c: Classification, max_complexity: int) -> Triage:
    if c.decision == "reject":
        return _stop(TaskState.DECLINED, c.triage_reason,
                     "CodePilot won't work on this issue automatically.", c.triage_reason)
    if c.decision == "clarify":
        questions = "\n".join(f"- {q}" for q in c.questions) or f"- {c.triage_reason}"
        return _stop(TaskState.NEEDS_INFO, c.triage_reason,
                     "CodePilot needs more detail before it can work on this issue:", questions)
    if c.complexity > max_complexity:
        reason = f"complexity {c.complexity}/10 exceeds the limit of {max_complexity} for automated changes"
        return _stop(TaskState.DECLINED, reason, "This looks too large to change automatically.",
                     f"{c.triage_reason} Splitting it into smaller, separately testable issues would let CodePilot help.")
    if c.confidence < MIN_CONFIDENCE:
        reason = f"task type uncertain ({c.task_type}, confidence {c.confidence:.2f})"
        return _stop(TaskState.NEEDS_INFO, reason, "CodePilot couldn't tell what kind of change this needs:",
                     "- Is this a bug, a new feature, a dependency, documentation or configuration change? "
                     "What should the result look like?")
    return Triage(True, TaskState.TRIAGED, c.triage_reason)


def _stop(state: TaskState, reason: str, lead: str, detail: str) -> Triage:
    comment = (f"{lead}\n\n{detail}\n\nEdit the issue or reply here and CodePilot will look at it again "
               f"on its next check. (Labelled `{LABELS[state]}`.)")
    return Triage(False, state, reason, comment)
