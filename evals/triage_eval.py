"""Evaluate triage: well-formed issues must proceed with the right task type; others must stop.

    uv run python evals/triage_eval.py [--holdout]

Tuning set: the seeded issues and classifier cases (all should proceed) plus evals/triage_cases.jsonl.
--holdout scores evals/classifier_holdout.jsonl and evals/triage_holdout.jsonl, never used for tuning.
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from classifier_eval import seeded_cases  # noqa: E402

from codepilot.classifier import build_classifier  # noqa: E402
from codepilot.models import Role, model_spec  # noqa: E402
from codepilot.triage import triage  # noqa: E402


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / "evals" / name).read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", action="store_true", help="Score only the held-out cases")
    args = parser.parse_args()
    if args.holdout:
        cases = load("classifier_holdout.jsonl") + load("triage_holdout.jsonl")
    else:
        cases = seeded_cases() + load("classifier_cases.jsonl") + load("triage_cases.jsonl")
    classifier = build_classifier()
    print(f"Classifier: {model_spec(Role.CLASSIFIER)} | {len(cases)} cases\n")

    rows = []
    for case in cases:
        allowed = case.get("decisions", ["accept"])
        try:
            c = classifier.classify(case["title"], case["body"])
            t = triage(c)
            row = {"proceed": t.proceed, "decision": c.decision, "complexity": c.complexity, "state": t.state.value,
                   "type_ok": c.task_type.value in case["accept"] if "accept" in case else None}
        except Exception as e:  # noqa: BLE001 - invalid output counts as a miss, not a crash
            row = {"proceed": None, "decision": f"{type(e).__name__}", "complexity": None, "state": None,
                   "type_ok": False}
        row["proceed_ok"] = (row["proceed"] is True and "accept" in allowed) or (
            row["proceed"] is False and any(d != "accept" for d in allowed))
        row["decision_ok"] = row["decision"] in allowed
        rows.append({**case, **row})
        flag = "" if row["proceed_ok"] and row["decision_ok"] and row["type_ok"] is not False else "  <- MISS"
        print(f"{case['id']:7} {row['decision']:8} cx {row['complexity']} -> {row['state']}{flag}")

    proceed_ok = sum(r["proceed_ok"] for r in rows)
    decision_ok = sum(r["decision_ok"] for r in rows)
    typed = [r for r in rows if r["type_ok"] is not None]
    leaked = [r["id"] for r in rows if r["proceed"] and "accept" not in r.get("decisions", ["accept"])]
    blocked = [r["id"] for r in rows if r["proceed"] is False and r.get("decisions", ["accept"]) == ["accept"]]
    print(f"\nProceed/stop correct  {proceed_ok}/{len(rows)}")
    print(f"Decision correct      {decision_ok}/{len(rows)}")
    print(f"Task type correct     {sum(r['type_ok'] for r in typed)}/{len(typed)}")
    print(f"Should stop, ran      {leaked}")
    print(f"Should run, stopped   {blocked}")


if __name__ == "__main__":
    main()
