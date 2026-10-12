"""Skills: reusable workflows the Orchestrator selects per task type.

Each skill lives in `skills/<name>/SKILL.md` in the Agent Skills format that
DeepAgents loads (YAML frontmatter + markdown body). The structured fields the
Orchestrator needs sit under the spec's `metadata` key and are validated here.
"""

import re
import shutil
from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from codepilot.classifier import TaskType

SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"
FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


class Skill(BaseModel):
    name: str = Field(description="Directory name, the identifier DeepAgents uses")
    skill_id: str = Field(description="Identifier used in prompts and logs, e.g. bug_fix_skill")
    task_type: TaskType
    description: str
    instructions: str
    workflow_steps: list[str] = Field(min_length=1)
    example_prompts: list[str] = Field(min_length=1)
    forbidden_actions: list[str] = Field(min_length=1)
    path: Path

    def install(self, skills_root: Path) -> Path:
        """Copy this skill alone into a per-task skills directory for a subagent to load."""
        target = skills_root / self.name
        shutil.copytree(self.path.parent, target, dirs_exist_ok=True)
        return target


def parse_skill(skill_md: Path) -> Skill:
    match = FRONTMATTER.match(skill_md.read_text())
    if not match:
        raise ValueError(f"{skill_md}: missing YAML frontmatter")
    front, body = yaml.safe_load(match.group(1)), match.group(2).strip()
    meta = front.get("metadata") or {}
    if front.get("name") != skill_md.parent.name:
        raise ValueError(f"{skill_md}: name must match its directory")
    return Skill(
        name=front["name"],
        description=front["description"],
        instructions=body,
        path=skill_md,
        **{k: meta[k] for k in ("skill_id", "task_type", "workflow_steps", "example_prompts", "forbidden_actions")},
    )


@cache
def load_skills(skills_dir: Path = SKILLS_DIR) -> dict[TaskType, Skill]:
    skills = [parse_skill(p) for p in sorted(skills_dir.glob("*/SKILL.md"))]
    by_type = {s.task_type: s for s in skills}
    if len(by_type) != len(skills):
        raise ValueError("Two skills claim the same task type")
    return by_type


def skill_for(task_type: TaskType) -> Skill:
    return load_skills()[task_type]
