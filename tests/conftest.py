import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def target_python(monkeypatch):
    """Run sandboxed test suites with this interpreter, not the default target repo's venv.

    The default (../bookshelf-api/.venv) exists only on a machine with that repo checked out beside this one.
    """
    monkeypatch.setenv("CODEPILOT_TARGET_PYTHON_BIN", str(Path(sys.executable).parent))


@pytest.fixture(autouse=True)
def no_live_input_rail(monkeypatch):
    """The NeMo input rail calls a model; tests that need it pass a fake rail to run_issue."""
    monkeypatch.setenv("CODEPILOT_INPUT_RAIL", "off")
