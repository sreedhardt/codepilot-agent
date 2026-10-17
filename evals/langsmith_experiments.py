"""The labelled evaluation cases as LangSmith datasets, and runs of them as LangSmith experiments.

    uv run python evals/langsmith_experiments.py upload               # create or refresh the three datasets
    uv run python evals/langsmith_experiments.py run rail              # one experiment per call
    uv run python evals/langsmith_experiments.py run review
    uv run python evals/langsmith_experiments.py run triage --split holdout

Datasets: codepilot-triage (tuning and holdout splits), codepilot-review, codepilot-rail. Each experiment is
named after the suite and the model under test and records the git commit, so runs of different prompts or
models can be compared side by side in LangSmith. The scoring rules are the same as the local scripts
(evals/triage_eval.py, review_eval.py, rail_eval.py), which remain the way to evaluate without LangSmith.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))
load_dotenv(ROOT / ".env")

from langsmith import Client  # noqa: E402

from classifier_eval import seeded_cases  # noqa: E402
from codepilot.models import Role, build_model, model_spec  # noqa: E402

PREFIX = "codepilot"


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / "evals" / name).read_text().splitlines() if line]


# ----- datasets: (inputs, reference outputs, split) per case -----------------------------------------------------
def triage_examples() -> list[dict]:
    def example(case: dict, split: str) -> dict:
        return {"inputs": {"title": case["title"], "body": case["body"]},
                "outputs": {"decisions": case.get("decisions", ["accept"]), "task_types": case.get("accept")},
                "metadata": {"case": case["id"]}, "split": split}

    tuning = seeded_cases() + load("classifier_cases.jsonl") + load("triage_cases.jsonl")
    holdout = load("classifier_holdout.jsonl") + load("triage_holdout.jsonl")
    return [example(c, "tuning") for c in tuning] + [example(c, "holdout") for c in holdout]


def review_examples() -> list[dict]:
    cases = [json.loads(p.read_text()) for p in sorted((ROOT / "evals" / "review_cases").glob("*.json"))]
    return [{"inputs": {"issue": c["issue"], "diff": c["diff"], "repo_map": c["repo_map"]},
             "outputs": {"verdict": c["expect"]}, "metadata": {"case": c["id"], "note": c["note"]}} for c in cases]


def rail_examples() -> list[dict]:
    from rail_eval import INJECTED_ISSUE, MUST_BLOCK

    cases = seeded_cases() + load("classifier_cases.jsonl") + load("classifier_holdout.jsonl")
    cases += load("triage_cases.jsonl") + load("triage_holdout.jsonl")
    cases.append({"id": "attack", "title": "Fix typo in app.py", "body": INJECTED_ISSUE})
    return [{"inputs": {"title": c["title"], "body": c["body"]}, "outputs": {"blocked": c["id"] in MUST_BLOCK},
             "metadata": {"case": c["id"]}} for c in cases]


# ----- targets: what is under test --------------------------------------------------------------------------------
def triage_target():
    from codepilot.classifier import build_classifier
    from codepilot.triage import triage

    classifier = build_classifier()

    def target(inputs: dict) -> dict:
        c = classifier.classify(inputs["title"], inputs["body"])
        return {"task_type": c.task_type.value, "decision": c.decision, "complexity": c.complexity,
                "proceed": triage(c).proceed}
    return target, model_spec(Role.CLASSIFIER)


def review_target():
    from codepilot.review import review_change

    model = build_model(Role.REVIEWER, max_tokens=4000)

    def target(inputs: dict) -> dict:
        review = review_change(model, inputs["issue"], inputs["diff"], "All tests pass.", inputs["repo_map"])
        return {"verdict": review.verdict, "gaps": review.gaps()}
    return target, model_spec(Role.REVIEWER)


def rail_target():
    from codepilot.input_rail import NemoInputRail

    rail = NemoInputRail()
    return (lambda inputs: {"blocked": rail.screen(inputs["title"], inputs["body"]).blocked}), model_spec(Role.INPUT_RAIL)


# ----- evaluators: the local scripts' scoring rules ---------------------------------------------------------------
def proceed_correct(outputs: dict, reference_outputs: dict) -> dict:
    allowed = reference_outputs["decisions"]
    ok = (outputs["proceed"] and "accept" in allowed) or (not outputs["proceed"] and any(d != "accept" for d in allowed))
    return {"key": "proceed_correct", "score": int(ok)}


def decision_correct(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "decision_correct", "score": int(outputs["decision"] in reference_outputs["decisions"])}


def type_correct(outputs: dict, reference_outputs: dict) -> dict:
    types = reference_outputs.get("task_types")
    if not types:  # triage-only cases carry no task-type label
        return {"key": "type_correct", "score": None, "comment": "no task-type label"}
    return {"key": "type_correct", "score": int(outputs["task_type"] in types)}


def verdict_correct(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "verdict_correct", "score": int(outputs["verdict"] == reference_outputs["verdict"])}


def blocked_correct(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "blocked_correct", "score": int(outputs["blocked"] == reference_outputs["blocked"])}


SUITES = {
    "triage": (triage_examples, triage_target, [proceed_correct, decision_correct, type_correct]),
    "review": (review_examples, review_target, [verdict_correct]),
    "rail": (rail_examples, rail_target, [blocked_correct]),
}


def upload(client: Client) -> None:
    for suite, (examples, _, _) in SUITES.items():
        name = f"{PREFIX}-{suite}"
        if client.has_dataset(dataset_name=name):
            dataset = client.read_dataset(dataset_name=name)
            old = [e.id for e in client.list_examples(dataset_id=dataset.id)]
            if old:
                client.delete_examples(old)
        else:
            dataset = client.create_dataset(name, description=f"CodePilot {suite} evaluation cases (evals/)")
        rows = examples()
        client.create_examples(dataset_id=dataset.id, examples=rows)
        print(f"{name}: {len(rows)} examples")


def run(client: Client, suite: str, split: str | None) -> None:
    _, make_target, evaluators = SUITES[suite]
    target, model = make_target()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    data = client.list_examples(dataset_name=f"{PREFIX}-{suite}", splits=[split]) if split else f"{PREFIX}-{suite}"
    results = client.evaluate(target, data=data, evaluators=evaluators, max_concurrency=1,
                              experiment_prefix=f"{suite}-{model.partition(':')[2].replace('/', '-')}",
                              metadata={"model": model, "commit": commit, "split": split or "all"})
    scores: dict[str, list[int]] = {}
    for row in results:
        for result in row["evaluation_results"]["results"]:
            if result.score is not None:
                scores.setdefault(result.key, []).append(result.score)
    for key, values in scores.items():
        print(f"{key:18} {sum(values)}/{len(values)}")
    print(f"experiment: {results.experiment_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["upload", "run"])
    parser.add_argument("suite", nargs="?", choices=sorted(SUITES))
    parser.add_argument("--split", choices=["tuning", "holdout"], help="Triage only: run one split")
    args = parser.parse_args()
    client = Client()
    if args.command == "upload":
        upload(client)
    else:
        run(client, args.suite, args.split)


if __name__ == "__main__":
    main()
