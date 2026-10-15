import pytest
from deepagents import CompiledSubAgent, create_deep_agent
from langchain_core.messages import AIMessage

from codepilot.costs import BudgetExceeded, UsageTracker, call_cost
from tests.fakes import ScriptedModel, tool_call


def usage(inp, out, read=0, write=0):
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out,
            "input_token_details": {"cache_read": read, "ephemeral_5m_input_tokens": write}}


def test_cost_prices_cache_reads_and_writes():
    # Sonnet 5: $2 in / $10 out per million. 1,000 input = 700 uncached + 200 cache read + 100 cache write.
    cost, priced = call_cost("claude-sonnet-5", usage(1_000, 500, read=200, write=100))
    expected = (700 * 2 + 200 * 2 * 0.1 + 100 * 2 * 1.25 + 500 * 10) / 1_000_000
    assert priced and cost == pytest.approx(expected)


def test_unknown_models_are_flagged_not_guessed():
    assert call_cost("some-new-model", usage(1_000, 100)) == (0.0, False)


def test_free_tier_models_cost_nothing():
    assert call_cost("openai/gpt-oss-120b", usage(50_000, 2_000)) == (0.0, True)


def team(child_usage, parent_usage):
    child = create_deep_agent(model=ScriptedModel(model="claude-sonnet-5", script=[
        AIMessage(content="done", usage_metadata=child_usage)]), name="coder")
    return create_deep_agent(
        model=ScriptedModel(model="claude-haiku-4-5", script=[
            AIMessage(content="", usage_metadata=parent_usage,
                      tool_calls=[{"name": "task", "args": {"description": "x", "subagent_type": "coder"},
                                   "id": "t1", "type": "tool_call"}]),
            AIMessage(content="ok", usage_metadata=parent_usage)]),
        subagents=[CompiledSubAgent(name="coder", description="c", runnable=child)], name="orchestrator")


def test_usage_is_attributed_to_each_agent_including_subagents():
    tracker = UsageTracker()
    team(usage(2_000, 100), usage(1_000, 50)).invoke({"messages": [{"role": "user", "content": "go"}]},
                                                   {"callbacks": [tracker]})
    summary = tracker.summary()["agents"]
    assert summary["orchestrator"]["calls"] == 2 and summary["orchestrator"]["model"] == "claude-haiku-4-5"
    assert summary["coder"]["calls"] == 1 and summary["coder"]["input_tokens"] == 2_000
    assert tracker.total_cost == pytest.approx((2 * (1_000 * 1 + 50 * 5) + 2_000 * 2 + 100 * 10) / 1_000_000)


def test_cost_budget_stops_the_run():
    tracker = UsageTracker(max_cost_usd=0.001)  # the first orchestrator call alone costs $0.00125
    with pytest.raises(BudgetExceeded, match="cost budget"):
        team(usage(2_000, 100), usage(1_000, 50)).invoke({"messages": [{"role": "user", "content": "go"}]},
                                                       {"callbacks": [tracker]})
    assert tracker.summary()["agents"].keys() == {"orchestrator"}  # stopped before the coder spent anything


def test_deadline_stops_the_run_before_any_spend():
    tracker = UsageTracker(max_seconds=0)
    with pytest.raises(BudgetExceeded, match="deadline"):
        team(usage(10, 1), usage(10, 1)).invoke({"messages": [{"role": "user", "content": "go"}]},
                                               {"callbacks": [tracker]})
    assert tracker.total_cost == 0


def test_billing_errors_are_recognised_through_the_cause_chain():
    from codepilot.costs import is_billing_error

    anthropic = ("Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': "
                 "'Your credit balance is too low to access the Anthropic API.'}}")
    try:
        try:
            raise ValueError(anthropic)
        except ValueError as inner:
            raise RuntimeError("subagent failed") from inner
    except RuntimeError as wrapped:
        assert is_billing_error(wrapped)
    assert is_billing_error(RuntimeError("Error code: 429 - {'error': {'code': 'insufficient_quota'}}"))
    assert not is_billing_error(RuntimeError("Error code: 429 - rate limit exceeded"))
    assert not is_billing_error(None)
