"""Issue classification: decides which Skill and subagents handle an issue.

`IssueClassifier` is the seam: the LLM implementation below is the default,
and a different backend (for example a System One model) can be swapped in
without touching the Orchestrator.
"""

from enum import StrEnum
from typing import Protocol

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from codepilot.models import Role, build_model, model_spec


class TaskType(StrEnum):
    BUG_FIX = "bug_fix"
    FEATURE_ADDITION = "feature_addition"
    DEPENDENCY_UPDATE = "dependency_update"
    DOCUMENTATION = "documentation"
    CONFIG_CHANGE = "config_change"


class Classification(BaseModel):
    task_type: TaskType
    confidence: float = Field(ge=0, le=1, description="How sure you are, from 0 to 1")
    rationale: str = Field(description="One sentence naming the deciding evidence in the issue")


class IssueClassifier(Protocol):
    def classify(self, title: str, body: str) -> Classification: ...


PROMPT = """Classify this GitHub issue into exactly one task type for an automated coding agent.

- bug_fix: existing behaviour is wrong (errors, incorrect results, crashes, regressions).
- feature_addition: new behaviour, endpoint, option or capability that does not exist yet.
- dependency_update: upgrading, pinning or migrating off a library version or deprecated library API.
- documentation: only docs, README, docstrings or examples change; no behaviour change.
- config_change: making existing hard-coded values configurable, or changing settings, env vars or tooling config.

Classify by the work required, not by the words used: a docs example that is wrong is documentation;
a bug fixed by upgrading a library is dependency_update. Deprecation warnings are not bugs: replacing
deprecated APIs (from a library, framework or the Python standard library) is dependency_update.

Issue title: {title}

Issue body:
{body}"""

# Provider-specific settings that keep a short structured answer cheap and non-empty.
PROVIDER_KWARGS = {
    "groq": {"reasoning_effort": "low", "max_tokens": 2048},
    "anthropic": {"max_tokens": 1024},
}


class LLMIssueClassifier:
    def __init__(self, model: BaseChatModel, *, structured_method: str | None = None) -> None:
        kwargs = {"method": structured_method} if structured_method else {}
        self._chain = model.with_structured_output(Classification, **kwargs)

    def classify(self, title: str, body: str) -> Classification:
        return self._chain.invoke(PROMPT.format(title=title, body=body or "(empty)"))


def build_classifier() -> IssueClassifier:
    provider = model_spec(Role.CLASSIFIER).partition(":")[0]
    model = build_model(Role.CLASSIFIER, **PROVIDER_KWARGS.get(provider, {}))
    # Groq's gpt-oss models support strict JSON-schema output, which guarantees a valid label.
    return LLMIssueClassifier(model, structured_method="json_schema" if provider == "groq" else None)
