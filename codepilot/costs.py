"""Token and cost tracking per agent, and per-task budgets that stop runaway work.

One callback handler attached to the root run sees every model call, including
those inside subagents (each call carries the calling agent's name). Before a
call it checks the task's deadline and cost so far; exceeding either raises
BudgetExceeded, which ends the task as FAILED.
"""

import threading
import time
from dataclasses import asdict, dataclass

from langchain_core.callbacks import BaseCallbackHandler

# USD per million tokens (input, output). Anthropic cache reads bill at 0.1x input, 5-minute cache writes at
# 1.25x and 1-hour writes at 2x. Groq models run on the free tier here, so they cost nothing.
PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-5-5": (4.00, 20.00),
    "openai/gpt-oss-120b": (0.0, 0.0),
    "openai/gpt-oss-20b": (0.0, 0.0),
}


class BudgetExceeded(RuntimeError):
    """The task ran past its deadline or cost budget."""


@dataclass
class AgentUsage:
    model: str = ""
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost_usd: float = 0.0
    priced: bool = True


def call_cost(model: str, usage: dict) -> tuple[float, bool]:
    """Cost of one call from its usage metadata; (0, False) when the model has no known price."""
    price = PRICES.get(model)
    if price is None:
        return 0.0, False
    price_in, price_out = price
    details = usage.get("input_token_details") or {}
    read = details.get("cache_read") or 0
    write_5m = (details.get("cache_creation") or 0) + (details.get("ephemeral_5m_input_tokens") or 0)
    write_1h = details.get("ephemeral_1h_input_tokens") or 0
    uncached = max(usage.get("input_tokens", 0) - read - write_5m - write_1h, 0)
    cost = (uncached + 0.1 * read + 1.25 * write_5m + 2.0 * write_1h) * price_in + usage.get("output_tokens", 0) * price_out
    return cost / 1_000_000, True


class UsageTracker(BaseCallbackHandler):
    raise_error = True  # a budget breach must stop the run, not be logged and ignored

    def __init__(self, max_cost_usd: float | None = None, max_seconds: float | None = None) -> None:
        self.max_cost_usd, self.max_seconds = max_cost_usd, max_seconds
        self.started = time.monotonic()
        self.agents: dict[str, AgentUsage] = {}
        self._calls: dict = {}
        self._lock = threading.Lock()

    @property
    def total_cost(self) -> float:
        return sum(a.cost_usd for a in self.agents.values())

    def check_budget(self) -> None:
        elapsed = time.monotonic() - self.started
        if self.max_seconds is not None and elapsed > self.max_seconds:
            raise BudgetExceeded(f"task deadline of {self.max_seconds / 60:g} min exceeded ({elapsed / 60:.1f} min)")
        if self.max_cost_usd is not None and self.total_cost > self.max_cost_usd:
            raise BudgetExceeded(f"task cost budget of ${self.max_cost_usd:.2f} exceeded (${self.total_cost:.3f})")

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs) -> None:
        metadata = metadata or {}
        with self._lock:
            self._calls[run_id] = (metadata.get("lc_agent_name") or "agent", metadata.get("ls_model_name") or "")
        self.check_budget()

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        agent, model = self._calls.pop(run_id, ("agent", ""))
        message = getattr(response.generations[0][0], "message", None) if response.generations and response.generations[0] else None
        usage = getattr(message, "usage_metadata", None) or {}
        model = model or (response.llm_output or {}).get("model_name") or (response.llm_output or {}).get("model") or ""
        cost, priced = call_cost(model, usage)
        details = usage.get("input_token_details") or {}
        with self._lock:
            a = self.agents.setdefault(agent, AgentUsage(model=model))
            a.model = a.model or model
            a.calls += 1
            a.input_tokens += usage.get("input_tokens", 0)
            a.output_tokens += usage.get("output_tokens", 0)
            a.cache_read += details.get("cache_read") or 0
            a.cache_write += ((details.get("cache_creation") or 0) + (details.get("ephemeral_5m_input_tokens") or 0)
                              + (details.get("ephemeral_1h_input_tokens") or 0))
            a.cost_usd += cost
            a.priced = a.priced and priced

    def summary(self) -> dict:
        return {
            "total_cost_usd": round(self.total_cost, 4),
            "seconds": round(time.monotonic() - self.started, 1),
            "agents": {name: asdict(u) | {"cost_usd": round(u.cost_usd, 4)} for name, u in sorted(self.agents.items())},
        }
