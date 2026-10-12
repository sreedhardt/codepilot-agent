import subprocess
from pathlib import Path

import pytest

from codepilot.repo_map import build_repo_map, count_tokens, describe_python


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "loans.py").write_text(
        '"""Lending rules: checkout and late fees."""\n'
        "MAX_LOANS = 3\n"
        "def checkout(book_id): ...\n"
        "def _private(): ...\n"
        "class LoanError(Exception):\n    def describe(self): ...\n"
    )
    (tmp_path / "app" / "catalog.py").write_text("def search_books(query): ...\n")
    (tmp_path / "README.md").write_text("# Bookshelf API\n\nDocs.\n")
    (tmp_path / ".gitignore").write_text("ignored.txt\n")
    (tmp_path / "ignored.txt").write_text("secret")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "-A")
    git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    return tmp_path


def test_describe_python_lists_public_symbols_and_docstring():
    symbols, doc = describe_python('"""Do things.\n\nMore."""\nX = 1\ndef f(): ...\ndef _g(): ...\nclass C:\n    def m(self): ...\n')
    assert symbols == ["X", "f()", "C[m]"]
    assert doc == "Do things."


def test_map_respects_gitignore_and_describes_files(repo):
    m = build_repo_map(repo)
    assert "ignored.txt" not in m.files
    loans = m.files["app/loans.py"]
    assert loans.description == "Lending rules: checkout and late fees."
    assert loans.symbols == ["MAX_LOANS", "checkout()", "LoanError[describe]"]
    assert m.files["README.md"].description == "Bookshelf API"


def test_summarizer_only_called_for_undescribed_files(repo):
    calls = []
    build_repo_map(repo, summarize=lambda path, text: calls.append(path) or "summary")
    assert sorted(calls) == [".gitignore", "app/catalog.py"]


def test_render_fits_budget(repo):
    m = build_repo_map(repo)
    for budget in (4000, 60, 25):
        assert count_tokens(m.render(budget)) <= budget


def test_keyword_search_ranks_by_paths_and_symbols(repo):
    m = build_repo_map(repo)
    assert m.keyword_search("search for books is case sensitive")[0] == "app/catalog.py"
    assert m.keyword_search("late fee on checkout")[0] == "app/loans.py"


def test_cache_rebuilds_only_changed_files(repo, tmp_path_factory):
    cache = tmp_path_factory.mktemp("cache")
    calls = []

    def summarize(path, text):
        calls.append(path)
        return "s"

    build_repo_map(repo, cache_dir=cache, summarize=summarize)
    calls.clear()
    build_repo_map(repo, cache_dir=cache, summarize=summarize)
    assert calls == []  # nothing changed: everything from cache

    (repo / "app" / "catalog.py").write_text("def search_books(query, limit): ...\n")
    build_repo_map(repo, cache_dir=cache, summarize=summarize)
    assert calls == ["app/catalog.py"]
