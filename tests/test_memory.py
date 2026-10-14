import subprocess
from types import SimpleNamespace

import pytest
from deepagents import CompiledSubAgent, create_deep_agent
from langchain_core.messages import AIMessage

from codepilot.memory.episodic import EpisodicMemory, TaskEpisode, open_store
from codepilot.memory.semantic import Lesson, SemanticMemory, render_lessons
from codepilot.memory.working import WorkingMemory, WorkingMemoryMiddleware
from codepilot.workspace import TaskWorkspace
from tests.fakes import ScriptedModel, tool_call


@pytest.fixture
def ws(tmp_path):
    repo = tmp_path / "repo"
    (repo / "app").mkdir(parents=True)
    (repo / "app" / "catalog.py").write_text("def search(): ...\n")
    for args in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "i"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    return TaskWorkspace(2, runs_dir=tmp_path / "runs", source=repo)


# --- working memory -------------------------------------------------------------------------------------------

def test_working_memory_reads_progress_from_workspace(ws):
    ws.write("relevant_files.md", "1. `/app/catalog.py` - search\n2. app/missing.py - not a real file\n")
    ws.write("test_report.md", "# Report\n\nResult: PASS - 25 passed, 0 failed\n")
    ws.write("proposed_diff.txt", "diff ...")
    wm = WorkingMemory(2, "Search", "u", "bug_fix", "bug_fix_skill")

    wm.refresh(ws, "TESTING", 1)

    assert wm.relevant_files == ["app/catalog.py"]
    assert wm.test_result == "Result: PASS - 25 passed, 0 failed"
    assert wm.diff == "/working/proposed_diff.txt" and wm.state == "TESTING" and wm.retry_count == 1
    assert '"state": "TESTING"' in wm.snapshot()


def test_working_memory_is_archived_when_task_ends(ws):
    wm = WorkingMemory(2, "Search", "u", "bug_fix", "bug_fix_skill")
    wm.save(ws)
    wm.archive(ws)
    assert not (ws.working / "working_memory.json").exists()
    assert (ws.working / "working_memory.final.json").exists()


def test_every_delegation_carries_the_working_memory_snapshot():
    child_model = ScriptedModel(script=[AIMessage(content="done")], received=[])
    child = create_deep_agent(model=child_model, name="child")
    parent = create_deep_agent(
        model=ScriptedModel(script=[tool_call("task", {"description": "Fix search.", "subagent_type": "child"}, "t1"),
                                    AIMessage(content="ok")]),
        subagents=[CompiledSubAgent(name="child", description="child", runnable=child)],
        middleware=[WorkingMemoryMiddleware(lambda: "Working memory: {\"state\": \"IMPLEMENTING\"}")],
    )
    parent.invoke({"messages": [{"role": "user", "content": "go"}]})

    first_human = next(m for m in child_model.received[0] if m.type == "human")
    assert first_human.content.startswith("Fix search.")
    assert '"state": "IMPLEMENTING"' in first_human.content


# --- episodic memory ------------------------------------------------------------------------------------------

def episode(n, state="DONE", outcome="PR_OPENED https://x/pull/7", pr=None):
    return TaskEpisode(n, f"Issue {n}", "bug_fix", state, outcome, ["app/catalog.py"], 120.0, pr)


def test_sessions_persist_and_recent_excludes_the_current_one(tmp_path):
    path = tmp_path / "episodes.sqlite"
    first = EpisodicMemory(open_store(path), "o/r")
    first.start_session()
    first.record(episode(2, pr="https://x/pull/7"))
    first.record(episode(1, state="FAILED", outcome="FAILED: tests do not pass"))
    first.end_session()

    second = EpisodicMemory(open_store(path), "o/r")  # a later process
    second.start_session()
    recent = second.recent()
    assert len(recent) == 1 and len(recent[0]["tasks"]) == 2
    assert "1 done, 1 failed" in recent[0]["summary"]
    assert [a["state"] for a in second.previous_attempts(1)] == ["FAILED"]
    assert "#1 Issue 1 [bug_fix] FAILED" in second.render_recent()


def test_sessions_are_separated_by_repository(tmp_path):
    store = open_store(tmp_path / "episodes.sqlite")
    a = EpisodicMemory(store, "o/a")
    a.start_session()
    a.record(episode(1))
    a.end_session()
    b = EpisodicMemory(store, "o/b")
    b.start_session()
    assert b.recent() == []


# --- semantic memory ------------------------------------------------------------------------------------------

LESSON = Lesson(problem="Search compared raw strings, so lowercase queries missed title-case books.",
                files_changed=["app/catalog.py"], approach="Casefold query, title and author before comparing.",
                pitfalls="Use casefold rather than lower for Unicode.")


@pytest.fixture
def semantic(tmp_path):
    memory = SemanticMemory(tmp_path / "lessons", "o/r")
    memory.add_draft(2, "Book search is case-sensitive", "bug_fix", LESSON, pr_number=7)
    return memory


def test_drafts_are_excluded_unless_requested(semantic):
    query = "Author filter ignores capitalisation of names"
    assert semantic.search(query) == []
    found = semantic.search(query, include_drafts=True)
    assert [l["id"] for l in found] == ["o/r#2"] and found[0]["status"] == "draft"
    assert "UNMERGED" in render_lessons(found)


def test_unrelated_issues_retrieve_nothing(semantic):
    assert semantic.search("Add CSV export for the monthly finance report", include_drafts=True) == []


@pytest.mark.parametrize("pr_state, status", [("merged", "merged"), ("closed", "rejected"), ("open", "draft")])
def test_lessons_follow_their_pull_request(semantic, pr_state, status):
    changed = semantic.promote(lambda number: pr_state)
    meta = semantic.collection.get(ids=["o/r#2"], include=["metadatas"])["metadatas"][0]
    assert meta["status"] == status
    assert changed == ({} if pr_state == "open" else {"o/r#2": status})


def test_merged_lessons_are_retrieved_by_default(semantic):
    semantic.promote(lambda number: "merged")
    assert [l["id"] for l in semantic.search("search should ignore letter case")] == ["o/r#2"]
