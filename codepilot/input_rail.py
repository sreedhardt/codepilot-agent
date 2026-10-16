"""Input rail: NeMo Guardrails screens an issue's text before triage.

Issue text is the main way an outsider can reach CodePilot's agents, so text that addresses the agent, tries
to override its instructions or asks for secrets is stopped before any agent or classifier sees it. The rail
is one of several layers: triage rejects such issues too, the Reviewer treats diffs as untrusted, and the
guardrails in code hold even if every model is fooled (tests/test_prompt_injection.py).
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from codepilot.models import Role, build_model, model_spec

CONFIG_DIR = Path(__file__).parent / "rails"
PROVIDER_KWARGS = {"groq": {"reasoning_effort": "low", "max_tokens": 512}, "anthropic": {"max_tokens": 16}}


@dataclass
class RailVerdict:
    blocked: bool
    reason: str = ""


class InputRail(Protocol):
    def screen(self, title: str, body: str) -> RailVerdict: ...


class NemoInputRail:
    def __init__(self) -> None:
        from nemoguardrails import LLMRails, RailsConfig  # heavy import: only when the rail is used

        provider = model_spec(Role.INPUT_RAIL).partition(":")[0]
        llm = build_model(Role.INPUT_RAIL, **PROVIDER_KWARGS.get(provider, {}))
        self._rails = LLMRails(RailsConfig.from_path(str(CONFIG_DIR)), llm=llm)

    def screen(self, title: str, body: str) -> RailVerdict:
        result = self._rails.generate(
            messages=[{"role": "user", "content": f"{title}\n\n{body}"}],
            options={"rails": {"input": True, "output": False, "dialog": False, "retrieval": False},
                     "log": {"activated_rails": True}},
        )
        stopped = [r.name for r in result.log.activated_rails if r.stop]
        return RailVerdict(bool(stopped), f"blocked by the NeMo input rail ({', '.join(stopped)})" if stopped else "")


class NoRail:
    def screen(self, title: str, body: str) -> RailVerdict:
        return RailVerdict(False)


def build_input_rail() -> InputRail:
    return NoRail() if os.getenv("CODEPILOT_INPUT_RAIL", "on").lower() == "off" else NemoInputRail()
