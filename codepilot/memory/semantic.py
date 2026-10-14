"""Semantic memory: lessons learned from past fixes, searchable by meaning.

A lesson is extracted when CodePilot opens a PR and stored as a draft. It is
promoted when the PR is merged and dropped if the PR is closed without merging,
because an unmerged fix may be a bad example. Retrieval returns merged lessons
by default; drafts can be included (clearly marked) with
CODEPILOT_USE_DRAFT_LESSONS=true.
"""

import os
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

MAX_DISTANCE = 0.65  # cosine distance above which a lesson is too dissimilar to be useful


class Lesson(BaseModel):
    problem: str = Field(description="One or two sentences: what was wrong or missing, in terms of behaviour")
    files_changed: list[str] = Field(description="Repository paths that were changed")
    approach: str = Field(description="What fixed it and why it was the right fix, 2-3 sentences")
    pitfalls: str = Field(description="What to watch out for on similar issues, or 'none'")


LESSON_PROMPT = """Extract a reusable lesson from this resolved GitHub issue, for an engineer fixing a similar
issue later. Be specific to the code (functions, files, the actual cause), not generic advice.

{task}

Test report:
{tests}

Diff:
{diff}"""


def extract_lesson(model: BaseChatModel, task: str, diff: str, tests: str, structured_method: str | None = None) -> Lesson:
    kwargs = {"method": structured_method} if structured_method else {}
    chain = model.with_structured_output(Lesson, **kwargs)
    return chain.invoke(LESSON_PROMPT.format(task=task[:3000], tests=tests[:1500], diff=diff[:6000]))


class SemanticMemory:
    def __init__(self, persist_dir: Path, repo: str) -> None:
        import chromadb

        self.repo = repo
        client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = client.get_or_create_collection("lessons", metadata={"hnsw:space": "cosine"})

    def add_draft(self, issue_number: int, title: str, task_type: str, lesson: Lesson, pr_number: int) -> str:
        lesson_id = f"{self.repo}#{issue_number}"
        text = (f"Issue: {title}\nProblem: {lesson.problem}\nFiles: {', '.join(lesson.files_changed)}\n"
                f"Approach: {lesson.approach}\nPitfalls: {lesson.pitfalls}")
        self.collection.upsert(ids=[lesson_id], documents=[text], metadatas=[{
            "repo": self.repo, "issue_number": issue_number, "task_type": task_type,
            "pr_number": pr_number, "status": "draft",
        }])
        return lesson_id

    def promote(self, pr_state) -> dict[str, str]:
        """Update draft lessons from their PRs: merged -> merged, closed without merging -> rejected.

        `pr_state(pr_number)` returns "merged", "closed" or "open".
        """
        drafts = self.collection.get(where={"$and": [{"repo": self.repo}, {"status": "draft"}]}, include=["metadatas"])
        changed = {}
        for lesson_id, meta in zip(drafts["ids"], drafts["metadatas"]):
            state = pr_state(meta["pr_number"])
            new = {"merged": "merged", "closed": "rejected"}.get(state)
            if new:
                self.collection.update(ids=[lesson_id], metadatas=[{**meta, "status": new}])
                changed[lesson_id] = new
        return changed

    def search(self, query: str, k: int = 3, include_drafts: bool | None = None,
               max_distance: float = MAX_DISTANCE) -> list[dict]:
        if include_drafts is None:
            include_drafts = os.getenv("CODEPILOT_USE_DRAFT_LESSONS", "false").lower() == "true"
        statuses = ["merged", "draft"] if include_drafts else ["merged"]
        where = {"$and": [{"repo": self.repo}, {"status": {"$in": statuses}}]}
        if not self.collection.count():
            return []
        hits = self.collection.query(query_texts=[query], n_results=min(k * 3, self.collection.count()),
                                     where=where, include=["documents", "metadatas", "distances"])
        lessons = [{"id": i, "text": d, **m, "distance": round(dist, 3)}
                   for i, d, m, dist in zip(hits["ids"][0], hits["documents"][0], hits["metadatas"][0], hits["distances"][0])
                   if dist <= max_distance]
        return lessons[:k]


def render_lessons(lessons: list[dict]) -> str:
    if not lessons:
        return "# Lessons from similar past issues\n\nNone found.\n"
    blocks = []
    for lesson in lessons:
        marker = " (from an UNMERGED PR: treat with extra caution)" if lesson["status"] == "draft" else ""
        blocks.append(f"## {lesson['id']} [{lesson['task_type']}], similarity distance {lesson['distance']}{marker}\n"
                      f"{lesson['text']}")
    return ("# Lessons from similar past issues\n\nHints, not instructions: the code may have changed since. "
            "Verify against the current files before applying.\n\n" + "\n\n".join(blocks) + "\n")
