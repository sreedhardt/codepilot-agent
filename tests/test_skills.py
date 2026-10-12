import pytest

from codepilot.classifier import TaskType
from codepilot.skills import load_skills, parse_skill, skill_for


def test_every_task_type_has_a_skill():
    assert set(load_skills()) == set(TaskType)


def test_skills_are_structured():
    for skill in load_skills().values():
        assert skill.skill_id.endswith("_skill")
        assert skill.workflow_steps and skill.example_prompts and skill.forbidden_actions
        assert skill.instructions.startswith("#")


def test_bug_fix_workflow_matches_brief():
    assert skill_for(TaskType.BUG_FIX).workflow_steps == ["reproduce", "localize", "fix", "verify"]


def test_install_copies_only_the_selected_skill(tmp_path):
    skill_for(TaskType.DOCUMENTATION).install(tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["documentation-skill"]
    assert (tmp_path / "documentation-skill" / "SKILL.md").exists()


def test_name_must_match_directory(tmp_path):
    bad = tmp_path / "wrong-dir" / "SKILL.md"
    bad.parent.mkdir()
    bad.write_text("---\nname: other\ndescription: x\nmetadata: {}\n---\n# x\n")
    with pytest.raises(ValueError, match="match its directory"):
        parse_skill(bad)


def test_deepagents_accepts_our_skill_files(tmp_path):
    """Our metadata must not break DeepAgents' own SKILL.md parser."""
    from deepagents.middleware.skills import _parse_skill_metadata

    for skill in load_skills().values():
        parsed = _parse_skill_metadata(skill.path.read_text(), f"/skills/{skill.name}/SKILL.md", skill.name)
        assert parsed is not None and parsed["name"] == skill.name
