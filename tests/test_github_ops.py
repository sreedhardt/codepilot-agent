import subprocess
from types import SimpleNamespace

import pytest

from codepilot import github_ops
from codepilot.classifier import TaskType
from codepilot.github_ops import Change, MergeConflictError


def test_branch_name_is_kebab_case_and_bounded():
    name = github_ops.branch_name(2, "Book search is case-sensitive!")
    assert name == "codepilot/issue-2-book-search-is-case-sensitive"
    long = github_ops.branch_name(9, "A " * 60)
    assert long.startswith("codepilot/issue-9-") and len(long) <= len("codepilot/issue-9-") + 40


def test_commit_message_follows_brief_format():
    msg = github_ops.commit_message(TaskType.BUG_FIX, 2, "Make book search case-insensitive.",
                                    ["Casefold query, title and author in search_books"],
                                    ["Users type lowercase queries"])
    assert msg == (
        "fix(#2): Make book search case-insensitive\n\n"
        "- Casefold query, title and author in search_books\n"
        "- Users type lowercase queries\n"
        "- Closes #2"
    )


@pytest.mark.parametrize("task_type, prefix", [(TaskType.FEATURE_ADDITION, "feat"), (TaskType.DOCUMENTATION, "docs")])
def test_commit_type_follows_task_type(task_type, prefix):
    assert github_ops.commit_message(task_type, 3, "x", [], []).startswith(f"{prefix}(#3): X")


def test_pr_body_has_required_sections():
    body = github_ops.pr_body(2, "Search is case-sensitive", "https://github.com/o/r/issues/2", "Summary", 'Approach."',
                              [Change("app/catalog.py"), Change("old.py", deleted=True)], "25 passed")
    for section in ("## Issue", "## Approach", "## Files changed", "## Test results", "Closes #2"):
        assert section in body
    assert "`old.py` (deleted)" in body
    assert "**#2 Search is case-sensitive**" in body
    assert 'Approach."' not in body and "Approach." in body


def test_sandbox_changes_reports_modified_added_and_deleted(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    (tmp_path / "a.py").write_text("a")
    (tmp_path / "b.py").write_text("b")
    git("init", "-q")
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    (tmp_path / "a.py").write_text("changed")
    (tmp_path / "b.py").unlink()
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_new.py").write_text("x")

    changes = {c.path: c.deleted for c in github_ops.sandbox_changes(tmp_path)}
    assert changes == {"a.py": False, "b.py": True, "tests/test_new.py": False}


def fake_repo(head: str, upstream_files: list[str]):
    return SimpleNamespace(
        default_branch="main",
        get_branch=lambda name: SimpleNamespace(commit=SimpleNamespace(sha=head)),
        compare=lambda base, head: SimpleNamespace(files=[SimpleNamespace(filename=f) for f in upstream_files]),
    )


def test_no_conflict_when_base_unchanged():
    github_ops.check_conflicts(fake_repo("abc", ["app/catalog.py"]), "abc", [Change("app/catalog.py")])


def test_no_conflict_when_upstream_touched_other_files():
    github_ops.check_conflicts(fake_repo("def", ["README.md"]), "abc", [Change("app/catalog.py")])


def test_conflict_when_upstream_touched_same_file():
    with pytest.raises(MergeConflictError, match="app/catalog.py"):
        github_ops.check_conflicts(fake_repo("def", ["app/catalog.py"]), "abc", [Change("app/catalog.py")])


@pytest.mark.parametrize("raw, expected", [
    ('"Wrapped in quotes"', "Wrapped in quotes"),
    ("matches 'Dune' and 'Le Guin'", "matches 'Dune' and 'Le Guin'"),
    ('Ends with a stray quote."', "Ends with a stray quote."),
    ('Says "hi" politely', 'Says "hi" politely'),
])
def test_clean_only_strips_stray_quotes(raw, expected):
    assert github_ops.clean(raw) == expected


@pytest.mark.parametrize("summary", ["Fix: book search is case-insensitive", "fix(search): book search is case-insensitive",
                                     "FIX:  book search is case-insensitive"])
def test_commit_message_drops_a_type_prefix_the_model_added(summary):
    msg = github_ops.commit_message(TaskType.BUG_FIX, 2, summary, [], [])
    assert msg.splitlines()[0] == "fix(#2): Book search is case-insensitive"
