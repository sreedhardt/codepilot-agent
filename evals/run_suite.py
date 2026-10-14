"""Run CodePilot on the seeded bookshelf-api issues and score every result.

    uv run python evals/run_suite.py                 # all six issues
    uv run python evals/run_suite.py --issues 3 6    # a subset

Runs are dry runs (nothing is written to GitHub) in an isolated results folder,
with their own task records and no memory, so earlier lessons cannot help.
Each finished sandbox is scored on a copy: the hidden acceptance test for the
issue (never shown to the agents), the full suite, and for issue 5 (docs) an
automated checklist. Writes results.jsonl and results.md.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from langchain_community.utilities.github import GitHubAPIWrapper  # noqa: E402

from codepilot import github_ops  # noqa: E402
from codepilot.runner import Callbacks, RunOptions, run_issue  # noqa: E402
from codepilot.tasks import TaskRegistry  # noqa: E402
from codepilot.workspace import target_python_bin  # noqa: E402

ACCEPTANCE = ROOT / "evals" / "acceptance"
EXPECTED_TYPES = {1: "bug_fix", 2: "bug_fix", 3: "feature_addition", 4: "dependency_update",
                  5: "documentation", 6: "config_change"}
DOC_ENDPOINTS = ["/health", "/books", "/books/{book_id}", "/loans", "/loans/{loan_id}/return",
                 "/loans/{loan_id}", "/members/{member_id}/loans"]


def pytest(cwd: Path, *args: str) -> tuple[bool, str]:
    result = subprocess.run(["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", *args], cwd=cwd,
                            env={"PATH": f"{target_python_bin()}:/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
                            capture_output=True, text=True, timeout=600)
    lines = (result.stdout + result.stderr).strip().splitlines()
    return result.returncode == 0, lines[-1] if lines else ""


def score(issue: int, sandbox: Path, scoring_dir: Path) -> dict:
    """Score a finished sandbox on a copy, so the agents' workspace is never modified."""
    copy = scoring_dir / f"issue-{issue}"
    shutil.copytree(sandbox, copy)
    suite_ok, suite_line = pytest(copy)
    changed = [c.path for c in github_ops.sandbox_changes(copy)]

    if issue == 5:
        readme = (copy / "README.md").read_text()
        missing = [e for e in DOC_ENDPOINTS if e not in readme]
        checks = {
            "only_docs_changed": changed == ["README.md"],
            "all_endpoints_documented": not missing,
            "no_invented_endpoints": "/renew" not in readme,
            "has_curl_examples": "curl" in readme,
            "documents_errors": all(code in readme for code in ("404", "409", "422")),
        }
        return {"fixed": all(checks.values()), "acceptance": f"{sum(checks.values())}/{len(checks)} checklist",
                "checks": checks, "missing_endpoints": missing, "suite_passed": suite_ok, "suite": suite_line,
                "files_changed": changed}

    shutil.copy(ACCEPTANCE / f"issue_{issue}.py", copy / "tests" / "test_hidden_acceptance.py")
    accepted, accept_line = pytest(copy, "tests/test_hidden_acceptance.py")
    result = {"acceptance": accept_line, "suite_passed": suite_ok, "suite": suite_line, "files_changed": changed}
    if issue == 4:
        strict_ok, strict_line = pytest(copy, "-W", "error::pydantic.warnings.PydanticDeprecatedSince20",
                                        "--ignore=tests/test_hidden_acceptance.py")
        result["strict_pydantic"] = strict_line
        accepted = accepted and strict_ok
    result["fixed"] = accepted and suite_ok
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issues", nargs="+", type=int, default=sorted(EXPECTED_TYPES))
    args = parser.parse_args()

    out = ROOT / "evals" / "results" / f"suite_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True)
    registry, repo = TaskRegistry(out / "tasks.json"), GitHubAPIWrapper().github_repo_instance
    options = RunOptions(open_pr=False, runs_dir=out / "runs", watchdog_seconds=None)
    log = (out / "agent_log.txt").open("a")
    callbacks = Callbacks(log=lambda line: (log.write(line + "\n"), log.flush()),
                          approve=lambda request, ws: {"approved": False, "note": "evaluation run"})
    print(f"Results: {out.relative_to(ROOT)}")

    rows = []
    for number in args.issues:
        print(f"#{number} running...", flush=True)
        result = run_issue(repo.get_issue(number), options, callbacks, registry)  # no memory: no head start
        r = result.record
        row = {
            "issue": number, "title": r.title, "expected_type": EXPECTED_TYPES.get(number), "classified": r.task_type,
            "state": r.state.value, "outcome": r.outcome[:200], "seconds": round(result.seconds),
            "cost_usd": result.usage.get("total_cost_usd", 0.0), "usage": result.usage,
            "fix_rounds": r.retries_after_fail, "model_retries": len(result.retries),
            "blocked": [f"[{b.agent}] {b.action}" for b in result.blocked],
            **score(number, result.workspace.sandbox, out / "scoring"),
        }
        rows.append(row)
        (out / "results.jsonl").open("a").write(json.dumps(row) + "\n")
        print(f"#{number} {'FIXED' if row['fixed'] else 'NOT FIXED'} | {row['acceptance']} | "
              f"${row['cost_usd']:.3f} | {row['seconds']}s", flush=True)

    write_markdown(rows, out / "results.md")
    print((out / "results.md").read_text())


def write_markdown(rows: list[dict], path: Path) -> None:
    lines = ["| # | Issue | Type (expected / got) | Fixed | Hidden check | Files | Fix rounds | Blocked | Time | Cost |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        type_ok = "✅" if r["classified"] == r["expected_type"] else "❌"
        lines.append(f"| {r['issue']} | {r['title'][:40]} | {r['expected_type']} / {r['classified']} {type_ok} "
                     f"| {'✅' if r['fixed'] else '❌'} | {r['acceptance']} | {len(r['files_changed'])} "
                     f"| {r['fix_rounds']} | {len(r['blocked'])} | {r['seconds']}s | ${r['cost_usd']:.3f} |")
    fixed = sum(r["fixed"] for r in rows)
    typed = sum(r["classified"] == r["expected_type"] for r in rows)
    total_cost, total_time = sum(r["cost_usd"] for r in rows), sum(r["seconds"] for r in rows)
    lines += ["", f"**Fixed {fixed}/{len(rows)}, classified {typed}/{len(rows)}, total ${total_cost:.2f}, "
                  f"{total_time / 60:.1f} min (mean ${total_cost / max(len(rows), 1):.3f} and "
                  f"{total_time / max(len(rows), 1) / 60:.1f} min per issue).**"]
    by_agent: dict[str, list] = {}
    for r in rows:
        for name, u in (r["usage"].get("agents") or {}).items():
            agg = by_agent.setdefault(name, [u["model"], 0, 0, 0, 0.0])
            agg[1] += u["calls"]
            agg[2] += u["input_tokens"]
            agg[3] += u["output_tokens"]
            agg[4] += u["cost_usd"]
    lines += ["", "| Agent | Model | Calls | Input tokens | Output tokens | Cost |", "|---|---|---|---|---|---|"]
    for name, (model, calls, inp, outp, cost) in sorted(by_agent.items(), key=lambda kv: -kv[1][4]):
        lines.append(f"| {name} | {model} | {calls} | {inp:,} | {outp:,} | ${cost:.3f} |")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
