"""Evaluate the Reviewer gate on frozen changes: real fixes it should approve, flawed ones it should block.

    uv run python evals/review_eval.py [--repeats 2]

Cases live in evals/review_cases/: fixes from evaluation runs that passed the hidden acceptance tests, one real
run that missed part of its issue, and copies of good fixes with a flaw inserted (debug output, a weakened or
skipped test, an unrelated destructive script). Reports blocks caught, good fixes wrongly blocked, and cost.
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from codepilot.costs import UsageTracker  # noqa: E402
from codepilot.models import Role, build_model, model_spec  # noqa: E402
from codepilot.review import ReviewUnavailable, review_change  # noqa: E402

TESTS = "All tests pass."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1, help="Review each case N times to measure consistency")
    args = parser.parse_args()
    cases = [json.loads(p.read_text()) for p in sorted((ROOT / "evals" / "review_cases").glob("*.json"))]
    usage = UsageTracker(None, None)
    model = build_model(Role.REVIEWER, max_tokens=4000, callbacks=[usage])
    print(f"Reviewer: {model_spec(Role.REVIEWER)} | {len(cases)} cases x {args.repeats}\n")

    rows = []
    for case in cases:
        for _ in range(args.repeats):
            try:
                review = review_change(model, case["issue"], case["diff"], TESTS, case["repo_map"])
                verdict, blocking = review.verdict, review.gaps()
            except ReviewUnavailable:
                verdict, blocking = "unavailable", []
            ok = verdict == case["expect"]
            rows.append((case, ok))
            print(f"{case['id']:15} expect {case['expect']:16} got {verdict:16}{'' if ok else '  <- MISS'}")
            for line in blocking:
                print(f"{'':17}blocking: {line[:150]}")

    bad = [(c, ok) for c, ok in rows if c["expect"] == "request_changes"]
    good = [(c, ok) for c, ok in rows if c["expect"] == "approve"]
    summary = usage.summary()
    print(f"\nFlawed changes blocked  {sum(ok for _, ok in bad)}/{len(bad)}")
    print(f"Good fixes approved     {sum(ok for _, ok in good)}/{len(good)}")
    print(f"Cost                    ${summary['total_cost_usd']:.4f} total, "
          f"${summary['total_cost_usd'] / len(rows):.4f} per review")


if __name__ == "__main__":
    main()
