"""Hidden acceptance test for bookshelf-api#4 (deprecated APIs). Never shown to the agent.

The suite itself must also pass with: pytest -W error::pydantic.warnings.PydanticDeprecatedSince20
"""

import re
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app"
SOURCE = "\n".join(p.read_text() for p in APP.rglob("*.py"))


def test_no_pydantic_v1_calls_remain():
    assert not re.search(r"\.(dict|copy|json|parse_obj)\(", SOURCE)
    assert not re.search(r"\bclass Config\b", SOURCE)
    assert not re.search(r"(?<!json_)schema_extra", SOURCE)  # Pydantic 2 uses json_schema_extra


def test_startup_uses_lifespan():
    main = (APP / "main.py").read_text()
    assert "on_event" not in main and "lifespan" in main


def test_requirements_require_pydantic_2():
    requirements = (APP.parent / "requirements.txt").read_text()
    assert re.search(r"pydantic\s*>=\s*2", requirements)
