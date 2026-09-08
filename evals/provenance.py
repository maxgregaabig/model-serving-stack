"""Provenance capture for benchmark runs.

`results/README.md` requires every committed result to record the conditions
that produced it -- the commit under test, the exact command, and the versions
of every backend involved. `results/baseline_bench_2026-06.json` records none of
those, which is why it cannot be reproduced.

Everything here is *observed*, never estimated. A version that cannot be read
from the running system is recorded as null rather than guessed, so a reader can
tell "not installed" from "not recorded".
"""

from __future__ import annotations

import os
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any

# Packages whose versions matter to a serving benchmark. Anything not installed
# is reported as null.
_TRACKED_DISTRIBUTIONS = {
    "vllm": "vllm",
    "tensorrt_llm": "tensorrt-llm",
    "tritonclient": "tritonclient",
    "ray": "ray",
    "torch": "torch",
    "transformers": "transformers",
    "httpx": "httpx",
}


def git_sha(repo_root: str | None = None) -> str | None:
    """Full commit SHA of the repository state under test.

    Returns None outside a git checkout. A dirty tree is reported with a
    "-dirty" suffix, because the committed SHA alone would not describe what
    actually ran.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None

    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return sha

    return f"{sha}-dirty" if dirty else sha


def invoked_command(argv: list[str] | None = None) -> str:
    """The exact invocation, quoted so it can be pasted back into a shell."""
    argv = sys.argv if argv is None else argv
    return shlex.join([sys.executable, "-m", "evals.bench_report", *argv[1:]])


def package_versions() -> dict[str, str | None]:
    """Installed versions of the packages a serving benchmark depends on."""
    versions: dict[str, str | None] = {}
    for key, distribution in _TRACKED_DISTRIBUTIONS.items():
        try:
            versions[key] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[key] = None
    versions["python"] = platform.python_version()
    return versions


def cuda_driver_version() -> str | None:
    """Driver version reported by nvidia-smi, or None where there is no GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    return out.splitlines()[0].strip() if out else None


def detected_hardware() -> str | None:
    """GPU model and memory plus host CPU/RAM, read from the running machine.

    None when there is no GPU, so a CPU-only run cannot silently inherit the
    hardware string of a previous GPU run.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    if not out:
        return None

    gpus = [line.strip() for line in out.splitlines() if line.strip()]
    gpu_part = ", ".join(gpus)
    cpu_count = os.cpu_count() or "unknown"

    ram_part = ""
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    ram_gib = int(line.split()[1]) / (1024 * 1024)
                    ram_part = f", {ram_gib:.0f}GB RAM"
                    break
    except (OSError, ValueError, IndexError):
        pass

    return f"{gpu_part}, {cpu_count} vCPU{ram_part}"


@dataclass
class Provenance:
    """The conditions a benchmark run happened under."""

    git_sha: str | None
    command: str
    software_versions: dict[str, str | None]
    hardware: str | None
    triton_server_image: str | None = None
    notes: list[str] = field(default_factory=list)

    @classmethod
    def capture(
        cls,
        *,
        argv: list[str] | None = None,
        repo_root: str | None = None,
        hardware_override: str | None = None,
        triton_server_image: str | None = None,
    ) -> Provenance:
        """Read provenance off the machine this is running on."""
        versions = package_versions()
        versions["cuda_driver"] = cuda_driver_version()

        hardware = hardware_override or detected_hardware()

        notes: list[str] = []
        if hardware is None:
            notes.append(
                "No GPU detected and no --hardware given: this run did not "
                "measure GPU serving."
            )
        if versions.get("cuda_driver") is None:
            notes.append("nvidia-smi unavailable; cuda_driver not recorded.")

        return cls(
            git_sha=git_sha(repo_root),
            command=invoked_command(argv),
            software_versions=versions,
            hardware=hardware,
            triton_server_image=triton_server_image,
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        versions = dict(self.software_versions)
        if self.triton_server_image is not None:
            versions["triton_server_image"] = self.triton_server_image
        payload: dict[str, Any] = {
            "git_sha": self.git_sha,
            "command": self.command,
            "software_versions": versions,
            "hardware": self.hardware,
        }
        if self.notes:
            payload["provenance_notes"] = self.notes
        return payload

    def missing_fields(self) -> list[str]:
        """Required provenance fields this run could not observe."""
        missing = []
        if not self.git_sha:
            missing.append("git_sha")
        if not self.command:
            missing.append("command")
        if not self.hardware:
            missing.append("hardware")
        if all(v is None for k, v in self.software_versions.items() if k != "python"):
            missing.append("software_versions")
        return missing
