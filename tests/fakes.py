"""A scripted chat model for testing agent wiring without API calls."""

from collections.abc import Iterator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedModel(BaseChatModel):
    """Returns the scripted messages in order; tools are accepted and ignored."""

    script: list[AIMessage]
    model: str = "scripted"  # reported to callbacks as ls_model_name, so costs can be priced in tests
    calls: int = 0
    received: list = []  # messages seen on each call, for assertions

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.received.append(list(messages))
        message = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator:
        raise NotImplementedError


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def approve(ws) -> None:
    """Record an approving review of the workspace's current diff, as the Reviewer would."""
    from codepilot.review import Review, ReviewRecord

    ReviewRecord(ws.working / "review.json").add(Review(requirements=[], files=[], findings=[], summary="ok"), ws.diff())
