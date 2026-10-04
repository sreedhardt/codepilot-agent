"""Verify the configured credentials work before running CodePilot.

    uv run python scripts/smoke_check.py

Checks the GitHub App can see the target repository's issues, and that every
model provider referenced in .env answers a one-word prompt.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from langchain.chat_models import init_chat_model  # noqa: E402
from langchain_community.utilities.github import GitHubAPIWrapper  # noqa: E402

from codepilot.models import Role, model_spec  # noqa: E402

failures = 0


def report(ok: bool, label: str, detail: str = "") -> None:
    global failures
    failures += not ok
    print(f"  {'OK  ' if ok else 'FAIL'} {label}{f'  ({detail})' if detail else ''}")


print("GitHub App")
try:
    github = GitHubAPIWrapper()
    # Count by iterating: PaginatedList.totalCount is an estimate and undercounts small lists.
    issues = [i for i in github.github_repo_instance.get_issues(state="open") if i.pull_request is None]
    report(True, os.environ["GITHUB_REPOSITORY"], f"{len(issues)} open issues")
except Exception as e:  # noqa: BLE001 - surface any setup problem
    report(False, os.getenv("GITHUB_REPOSITORY", "GITHUB_REPOSITORY unset"), f"{type(e).__name__}: {e}"[:200])

print("Models")
specs = {model_spec(role) for role in Role}
extra = [s.strip() for s in os.getenv("SMOKE_EXTRA_MODELS", "").split(",") if s.strip()]
for spec in sorted(specs | set(extra)):
    provider, _, model = spec.partition(":")
    try:
        reply = init_chat_model(model, model_provider=provider, max_tokens=50).invoke("Reply with the single word: ready")
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        report(True, spec, text.strip()[:40])
    except Exception as e:  # noqa: BLE001
        report(False, spec, f"{type(e).__name__}: {e}"[:200])

sys.exit(1 if failures else 0)
