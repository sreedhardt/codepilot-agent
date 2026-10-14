"""Hidden acceptance test for bookshelf-api#6 (configurable lending policy). Never shown to the agent.

Each case starts a fresh interpreter with the environment set, as a deployment would, so the test is fair to
any design (settings read at import or at call time) while still catching values captured too early.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

SCENARIO = """
import json
from datetime import date, timedelta
from fastapi.testclient import TestClient
from app.deps import get_today
from app.main import app
from app.store import store

clock = {"today": date(2026, 3, 2)}
app.dependency_overrides[get_today] = lambda: clock["today"]
with TestClient(app) as client:
    store.seed()
    loan = client.post("/loans", json={"book_id": 1, "member_id": 1}).json()
    second = client.post("/loans", json={"book_id": 2, "member_id": 1}).status_code
    clock["today"] = date.fromisoformat(loan["due"]) + timedelta(days=4)
    returned = client.post(f"/loans/{loan['id']}/return").json()
    print(json.dumps({"loan_days": (date.fromisoformat(loan["due"]) - date(2026, 3, 2)).days,
                      "second_checkout": second, "fee_4_days_late": returned["fee_cents"]}))
"""


def run(env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", SCENARIO], cwd=REPO, env={"PATH": "/usr/bin:/bin", **env},
                          capture_output=True, text=True, timeout=60)


def outcome(env: dict[str, str]) -> dict:
    result = run(env)
    assert result.returncode == 0, result.stderr[-800:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_defaults_are_unchanged():
    assert outcome({}) == {"loan_days": 14, "second_checkout": 201, "fee_4_days_late": 100}


def test_each_variable_overrides_its_policy():
    assert outcome({"BOOKSHELF_LOAN_PERIOD_DAYS": "7"})["loan_days"] == 7
    assert outcome({"BOOKSHELF_MAX_ACTIVE_LOANS": "1"})["second_checkout"] == 409
    assert outcome({"BOOKSHELF_DAILY_LATE_FEE_CENTS": "50"})["fee_4_days_late"] == 200
    assert outcome({"BOOKSHELF_MAX_LATE_FEE_CENTS": "60"})["fee_4_days_late"] == 60


@pytest.mark.parametrize("variable", ["BOOKSHELF_LOAN_PERIOD_DAYS", "BOOKSHELF_DAILY_LATE_FEE_CENTS",
                                      "BOOKSHELF_MAX_LATE_FEE_CENTS", "BOOKSHELF_MAX_ACTIVE_LOANS"])
@pytest.mark.parametrize("value", ["abc", "0", "-3"])
def test_invalid_values_fail_fast_with_a_clear_error(variable, value):
    result = run({variable: value})
    assert result.returncode != 0
    assert variable in result.stderr  # the error names the offending variable
