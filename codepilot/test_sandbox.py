"""Where agent-written tests run: a network-less Docker container, or the host.

Running a test suite executes code an agent wrote. On the host that code could read any file the user can and
reach the network; in the container it sees only the disposable sandbox copy, has no network, runs as the
host's (non-root) user on a read-only root filesystem with no capabilities, and is limited in processes, memory and CPU.

The image is built from the target repo's own dependency files and tagged by their hash, so a change to them
(a dependency update task) builds a new one. Builds have network access to install packages; test runs never
do.

CODEPILOT_TEST_SANDBOX: "docker" (required), "host", or "auto" (default: Docker when available).
"""

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_REPO = "codepilot-tests"
DEFAULT_PYTHON = "3.12"
RUN_TIMEOUT = 600
BUILD_TIMEOUT = 900
RUN_FLAGS = [
    "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=256m", "--cap-drop", "ALL",
    "--security-opt", "no-new-privileges", "--pids-limit", "256", "--memory", "1g", "--cpus", "2",
    "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "HOME=/tmp",
]
DOCKERFILE = """FROM python:{python}-slim
RUN useradd --create-home --uid 1000 runner
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt pytest
USER runner
WORKDIR /work
"""


@dataclass
class Completed:
    exit_code: int
    output: str


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def mode() -> str:
    """The test sandbox in effect: "docker" or "host"."""
    setting = os.getenv("CODEPILOT_TEST_SANDBOX", "auto").lower()
    if setting == "host":
        return "host"
    if docker_available():
        return "docker"
    if setting == "docker":
        raise RuntimeError("CODEPILOT_TEST_SANDBOX=docker but Docker is not available (is Docker Desktop running?)")
    logger.warning("Docker is not available: agent-written tests will run on the host, without isolation")
    return "host"


def requirements(sandbox: Path) -> str:
    """The packages the target's tests need, from requirements*.txt or pyproject.toml."""
    lines = []
    for path in sorted(sandbox.glob("requirements*.txt")):
        lines += [line.strip() for line in path.read_text().splitlines()
                  if line.strip() and not line.strip().startswith(("#", "-e", "."))]
    pyproject = sandbox / "pyproject.toml"
    if pyproject.exists():
        project = tomllib.loads(pyproject.read_text()).get("project", {})
        lines += project.get("dependencies", [])
        for group in ("test", "tests", "dev"):
            lines += project.get("optional-dependencies", {}).get(group, [])
    return "\n".join(sorted(set(lines))) + "\n"


def python_version(sandbox: Path) -> str:
    pinned = sandbox / ".python-version"
    if pinned.exists():
        return ".".join(pinned.read_text().strip().split(".")[:2])
    return DEFAULT_PYTHON


def run_as() -> str:
    """The container's uid:gid: the host user's, so the mounted sandbox is readable and writable on Linux, where
    Docker does not remap file ownership as Docker Desktop does; never root, even if CodePilot runs as root."""
    uid, gid = os.getuid(), os.getgid()
    return "1000:1000" if uid == 0 else f"{uid}:{gid}"


class DockerTestRunner:
    """Runs test commands for one sandbox in a container built for its dependencies."""

    def __init__(self, sandbox: Path) -> None:
        self.sandbox = sandbox

    def image(self) -> str:
        spec = python_version(self.sandbox) + "\n" + requirements(self.sandbox)
        return f"{IMAGE_REPO}:{hashlib.sha256(spec.encode()).hexdigest()[:12]}"

    def ensure_image(self) -> str:
        """Build the image for the sandbox's current dependencies unless it already exists."""
        tag = self.image()
        if subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0:
            return tag
        with tempfile.TemporaryDirectory() as context:
            Path(context, "requirements.txt").write_text(requirements(self.sandbox))
            Path(context, "Dockerfile").write_text(DOCKERFILE.format(python=python_version(self.sandbox)))
            logger.info("building test image %s", tag)
            built = subprocess.run(["docker", "build", "-q", "-t", tag, context], capture_output=True, text=True,
                                   timeout=BUILD_TIMEOUT)
        if built.returncode != 0:
            raise RuntimeError(f"could not build the test image {tag}: {built.stderr.strip()[-500:]}")
        return tag

    def command(self, argv: list[str], image: str) -> list[str]:
        return ["docker", "run", "--rm", *RUN_FLAGS, "--user", run_as(), "-v", f"{self.sandbox.resolve()}:/work",
                "-w", "/work", image, *argv]

    def run(self, argv: list[str], timeout: int = RUN_TIMEOUT) -> Completed:
        try:
            image = self.ensure_image()
            result = subprocess.run(self.command(argv, image), capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return Completed(124, f"test run timed out after {timeout}s")
        except (OSError, RuntimeError) as e:
            return Completed(125, f"could not run the tests in Docker: {e}")
        return Completed(result.returncode, result.stdout + result.stderr)
