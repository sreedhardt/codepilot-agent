"""Compare models as the Coder agent on seeded bookshelf-api issues.

Each run gets a fresh temporary copy of the target repo, a single DeepAgents
Coder behind the command guardrail, and the real issue text from GitHub. When
the agent stops, the hidden acceptance test for the issue is copied in and the
whole suite is run. Nothing is pushed: no branches, commits or PRs.

    uv run python evals/model_compare.py \
        --models groq:openai/gpt-oss-120b anthropic:claude-haiku-4-5 anthropic:claude-sonnet-5 \
        --issues 1 2
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from deepagents import create_deep_agent  # noqa: E402
from langchain.chat_models import init_chat_model  # noqa: E402
from langchain_community.utilities.github import GitHubAPIWrapper  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langgraph.errors import GraphRecursionError  # noqa: E402

from codepilot.sandbox import GuardedShellBackend  # noqa: E402

TARGET_REPO = ROOT.parent / "bookshelf-api"
TARGET_PYTHON_BIN = TARGET_REPO / ".venv" / "bin"
ACCEPTANCE = ROOT / "evals" / "acceptance"
RESULTS = ROOT / "evals" / "results"
BASELINE_TESTS = 20

# USD per million tokens: (input, output). Anthropic cache reads bill at 0.1x input,
# 5-minute cache writes at 1.25x and 1-hour writes at 2x.
PRICES = {
    "anthropic:claude-sonnet-5": (2.00, 10.00),
    "anthropic:claude-haiku-4-5": (1.00, 5.00),
}

CODER_PROMPT = """You are the Coder agent in CodePilot. You fix one GitHub issue in a Python repository.
Your file tools see the repository root as `/`. Shell commands already run in the repository root:
use relative paths and never `cd`.

Workflow:
1. Explore: use ls, glob, grep and read_file to find the relevant code. Read before editing.
2. Reproduce: add a test to the existing test files that fails because of the issue. Run it and confirm it fails.
3. Fix: make the smallest change that fixes the root cause. Use edit_file for surgical edits; never rewrite whole files.
4. Verify: run the full suite with `python -m pytest -q`. All tests must pass.
5. Finish: reply with a short summary: root cause, files changed, tests added, final test result.

Rules:
- Shell commands pass through a guardrail: one plain command per call, no pipes, chaining or redirection.
  Allowed: python -m pytest, ls, cat, grep, find, head, tail, wc, and git status/diff/log/show.
- Do not install packages or modify files unrelated to the issue.
- If tests still fail after 3 fix attempts, stop and report what is wrong."""


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = {"PATH": f"{TARGET_PYTHON_BIN}:/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=300)


def make_sandbox() -> Path:
    sandbox = Path(tempfile.mkdtemp(prefix="codepilot-sandbox-"))
    shutil.copytree(
        TARGET_REPO, sandbox, dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".venv", ".pytest_cache", "__pycache__"),
    )
    return sandbox


def build_model(spec: str):
    provider, _, model = spec.partition(":")
    kwargs = {"max_tokens": 16000} if provider == "anthropic" else {}
    return init_chat_model(model, model_provider=provider, max_retries=3, **kwargs)


def usage_and_cost(spec: str, messages: list) -> dict:
    totals = Counter()
    for m in messages:
        usage = getattr(m, "usage_metadata", None) or {}
        details = usage.get("input_token_details") or {}
        totals["input"] += usage.get("input_tokens", 0)
        totals["output"] += usage.get("output_tokens", 0)
        totals["cache_read"] += details.get("cache_read", 0) or 0
        # langchain-anthropic reports writes under TTL-specific keys and zeroes the generic one.
        totals["cache_write"] += (details.get("cache_creation") or 0) + (details.get("ephemeral_5m_input_tokens") or 0)
        totals["cache_write_1h"] += details.get("ephemeral_1h_input_tokens") or 0
    price_in, price_out = PRICES.get(spec, (0.0, 0.0))
    uncached = totals["input"] - totals["cache_read"] - totals["cache_write"] - totals["cache_write_1h"]
    cost = (
        uncached * price_in
        + totals["cache_read"] * price_in * 0.1
        + totals["cache_write"] * price_in * 1.25
        + totals["cache_write_1h"] * price_in * 2.0
        + totals["output"] * price_out
    ) / 1_000_000
    return {**totals, "cost_usd": round(cost, 4)}


def evaluate(sandbox: Path, issue: int) -> dict:
    collected = run(["python", "-m", "pytest", "--collect-only", "-q"], sandbox).stdout
    own_tests = sum(1 for line in collected.splitlines() if "::" in line)
    suite = run(["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"], sandbox)

    hidden = sandbox / "tests" / f"_acceptance_issue_{issue}.py"
    shutil.copy(ACCEPTANCE / f"issue_{issue}.py", hidden)
    accept = run(["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", str(hidden.relative_to(sandbox))], sandbox)
    hidden.unlink()

    changed = run(["git", "status", "--porcelain"], sandbox).stdout.split("\n")
    return {
        "suite_passed": suite.returncode == 0,
        "acceptance_passed": accept.returncode == 0,
        "tests_added": own_tests - BASELINE_TESTS,
        "files_changed": sorted(line[3:] for line in changed if line.strip()),
        "acceptance_tail": accept.stdout.strip().splitlines()[-1:] if accept.stdout else [],
    }


def run_one(spec: str, issue: int, recursion_limit: int) -> dict:
    gh_issue = GitHubAPIWrapper().github_repo_instance.get_issue(issue)
    task = f"Fix GitHub issue #{issue}: {gh_issue.title}\n\n{gh_issue.body}"

    sandbox = make_sandbox()
    backend = GuardedShellBackend(sandbox, env={"PATH": f"{TARGET_PYTHON_BIN}:/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})
    agent = create_deep_agent(model=build_model(spec), system_prompt=CODER_PROMPT, backend=backend)

    state, stop, started = {"messages": []}, "completed", time.monotonic()
    try:
        for state in agent.stream(
            {"messages": [{"role": "user", "content": task}]},
            config={"recursion_limit": recursion_limit},
            stream_mode="values",
        ):
            pass
    except GraphRecursionError:
        stop = "recursion_limit"
    except Exception as e:  # noqa: BLE001 - record provider errors (rate limits etc.) as a result
        stop = f"error: {type(e).__name__}: {str(e)[:200]}"
    elapsed = time.monotonic() - started

    messages = state.get("messages", [])
    ai = [m for m in messages if isinstance(m, AIMessage)]
    tool_calls = Counter(tc["name"] for m in ai for tc in m.tool_calls)
    tool_errors = sum(
        1 for m in messages
        if isinstance(m, ToolMessage) and (m.status == "error" or str(m.content).startswith("Error"))
    )

    result = {
        "model": spec,
        "issue": issue,
        "stop": stop,
        "seconds": round(elapsed, 1),
        "turns": len(ai),
        "tool_calls": dict(tool_calls),
        "tool_errors": tool_errors,
        "blocked_commands": backend.blocked,
        **usage_and_cost(spec, messages),
        **evaluate(sandbox, issue),
        "final_message": (ai[-1].text if ai and isinstance(ai[-1].text, str) else "")[:600],
        "sandbox": str(sandbox),
    }
    result["fixed"] = result["acceptance_passed"] and result["suite_passed"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--issues", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--recursion-limit", type=int, default=80)
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"model_compare_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    print(f"Writing {out.relative_to(ROOT)}\n")
    print("| Model | Issue | Fixed | Suite | Accept | Tests+ | Turns | Tool errs | Blocked | Tokens in/out | Cost | Time | Stop |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for spec in args.models:
        for issue in args.issues:
            r = run_one(spec, issue, args.recursion_limit)
            with out.open("a") as f:
                f.write(json.dumps(r) + "\n")
            print(
                f"| {spec} | #{issue} | {'✅' if r['fixed'] else '❌'} | {r['suite_passed']} | {r['acceptance_passed']} "
                f"| {r['tests_added']} | {r['turns']} | {r['tool_errors']} | {len(r['blocked_commands'])} "
                f"| {r['input']:,}/{r['output']:,} | ${r['cost_usd']:.3f} | {r['seconds']}s | {r['stop']} |",
                flush=True,
            )


if __name__ == "__main__":
    main()
