"""The Docker test sandbox: built from the target's dependencies, run with no network and no host access."""

import pytest

from codepilot import test_sandbox
from codepilot.sandbox import GuardedShellBackend
from codepilot.test_sandbox import DockerTestRunner, requirements

docker = pytest.mark.skipif(not test_sandbox.docker_available(), reason="Docker is not available")


def test_requirements_come_from_requirements_files_and_pyproject(tmp_path):
    (tmp_path / "requirements.txt").write_text("fastapi>=0.115\n# a comment\n-e .\nhttpx\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\ndependencies = ["pydantic>=2"]\n'
                                             '[project.optional-dependencies]\ntest = ["pytest-cov"]\n')
    assert requirements(tmp_path).split() == ["fastapi>=0.115", "httpx", "pydantic>=2", "pytest-cov"]


def test_the_image_follows_the_dependencies(tmp_path):
    (tmp_path / "requirements.txt").write_text("fastapi\n")
    runner = DockerTestRunner(tmp_path)
    before = runner.image()
    (tmp_path / "requirements.txt").write_text("fastapi\nhttpx\n")  # e.g. a dependency update task
    assert runner.image() != before


def test_runs_are_isolated(tmp_path):
    command = DockerTestRunner(tmp_path).command(["python", "-m", "pytest"], "img")
    joined = " ".join(command)
    for flag in ("--network none", "--read-only", "--cap-drop ALL", "--security-opt no-new-privileges"):
        assert flag in joined
    assert f"{tmp_path.resolve()}:/work" in joined  # only the sandbox copy is mounted


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    root = tmp_path_factory.mktemp("sandbox")
    (root / "tests").mkdir()
    (root / "requirements.txt").write_text("")
    return root


@docker
def test_a_normal_suite_passes(sandbox):
    (sandbox / "tests" / "test_ok.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n")
    result = DockerTestRunner(sandbox).run(["python", "-m", "pytest", "-q", "tests/test_ok.py"])
    assert result.exit_code == 0 and "1 passed" in result.output


@docker
def test_agent_written_tests_cannot_read_host_files(sandbox, tmp_path):
    secret = tmp_path / "host_secret.txt"
    secret.write_text("host-only")
    (sandbox / "tests" / "test_steal.py").write_text(
        f"def test_steal():\n    open('stolen.txt', 'w').write(open({str(secret)!r}).read())\n")
    result = DockerTestRunner(sandbox).run(["python", "-m", "pytest", "-q", "tests/test_steal.py"])
    stolen = sandbox / "stolen.txt"
    assert "FileNotFoundError" in result.output
    assert not stolen.exists() or "host-only" not in stolen.read_text()


@docker
def test_agent_written_tests_have_no_network(sandbox):
    (sandbox / "tests" / "test_net.py").write_text(
        "import socket\n\ndef test_net():\n    socket.create_connection(('1.1.1.1', 53), timeout=3)\n")
    result = DockerTestRunner(sandbox).run(["python", "-m", "pytest", "-q", "tests/test_net.py"])
    assert result.exit_code != 0 and "OSError" in result.output


@docker
def test_the_agent_shell_sends_tests_to_the_container(sandbox):
    runner = DockerTestRunner(sandbox)
    shell = GuardedShellBackend(sandbox, test_runner=runner.run)  # no PATH: a host pytest could not even start
    result = shell.execute("python -m pytest -q tests/test_ok.py")
    assert result.exit_code == 0 and "1 passed" in result.output
