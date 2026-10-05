"""Evaluate the issue classifier on the seeded issues plus extra labelled cases.

    uv run python evals/classifier_eval.py [--repeats 3]

Reports accuracy, invalid-output rate, latency, and whether self-reported
confidence separates right answers from wrong ones.
"""

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from codepilot.classifier import build_classifier  # noqa: E402
from codepilot.models import Role, model_spec  # noqa: E402

# Expected types for the seeded bookshelf-api issues (see evals/seeded_issues.md).
SEEDED_LABELS = {
    "1": ["bug_fix"],
    "2": ["bug_fix"],
    "3": ["feature_addition"],
    "4": ["dependency_update"],
    "5": ["documentation"],
    "6": ["config_change"],
}


def seeded_cases() -> list[dict]:
    text = (ROOT / "evals" / "seeded_issues.md").read_text()
    sections = re.split(r"^## (\d+)\. (.+)$", text, flags=re.M)[1:]
    cases = []
    for num, title, body in zip(sections[0::3], sections[1::3], sections[2::3]):
        issue = body.split("**Issue**", 1)[1].split("**Ground truth**", 1)[0]
        lines = [line[2:] if line.startswith("> ") else "" for line in issue.strip().splitlines()]
        cases.append({"id": f"seed{num}", "title": title.strip(), "body": "\n".join(lines).strip(),
                      "accept": SEEDED_LABELS[num]})
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1, help="Run each case N times to measure consistency")
    parser.add_argument("--holdout", action="store_true",
                        help="Score only evals/classifier_holdout.jsonl: cases never used to tune the prompt")
    args = parser.parse_args()

    if args.holdout:
        cases = [json.loads(line) for line in (ROOT / "evals" / "classifier_holdout.jsonl").read_text().splitlines() if line]
    else:
        extra = [json.loads(line) for line in (ROOT / "evals" / "classifier_cases.jsonl").read_text().splitlines() if line]
        cases = seeded_cases() + extra
    classifier = build_classifier()
    print(f"Classifier: {model_spec(Role.CLASSIFIER)} | {len(cases)} cases x {args.repeats}\n")

    rows, latencies = [], []
    for case in cases:
        for _ in range(args.repeats):
            started = time.monotonic()
            try:
                result = classifier.classify(case["title"], case["body"])
                label, confidence, error = result.task_type.value, result.confidence, None
            except Exception as e:  # noqa: BLE001 - invalid output counts as a failure, not a crash
                label, confidence, error = None, None, f"{type(e).__name__}: {str(e)[:120]}"
            latencies.append(time.monotonic() - started)
            rows.append({**case, "label": label, "confidence": confidence, "error": error,
                         "correct": label in case["accept"]})

    for r in rows:
        if not r["correct"]:
            got = r["label"] or r["error"]
            print(f"  MISS {r['id']:7} expected {'/'.join(r['accept']):30} got {got} (conf {r['confidence']})")

    correct = [r for r in rows if r["correct"]]
    wrong = [r for r in rows if not r["correct"] and r["label"]]
    invalid = [r for r in rows if r["label"] is None]
    lat = sorted(latencies)
    print(f"\nAccuracy        {len(correct)}/{len(rows)} = {len(correct) / len(rows):.0%}")
    print(f"Invalid output  {len(invalid)}")
    print(f"Latency         p50 {statistics.median(lat):.2f}s  p95 {lat[int(0.95 * (len(lat) - 1))]:.2f}s")
    if correct:
        print(f"Confidence      right: mean {statistics.mean(r['confidence'] for r in correct):.2f}"
              + (f" | wrong: mean {statistics.mean(r['confidence'] for r in wrong):.2f}" if wrong else " | wrong: n/a"))
    if args.repeats > 1:
        by_case = {}
        for r in rows:
            by_case.setdefault(r["id"], set()).add(r["label"])
        unstable = [cid for cid, labels in by_case.items() if len(labels) > 1]
        print(f"Unstable cases  {len(unstable)} {unstable}")


if __name__ == "__main__":
    main()
