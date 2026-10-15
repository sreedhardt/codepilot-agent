import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def target_python(monkeypatch):
    """Run sandboxed test suites with this interpreter, not the default target repo's venv.

    The default (../bookshelf-api/.venv) exists only on a machine with that repo checked out beside this one.
    """
    monkeypatch.setenv("CODEPILOT_TARGET_PYTHON_BIN", str(Path(sys.executable).parent))
