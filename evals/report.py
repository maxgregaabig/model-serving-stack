"""Build and validate benchmark reports in the schema `results/README.md` defines.

Two things here that the previous tooling did not do:

* Emit the committed schema. `evals/load_test.py` produces a `LoadTestReport`
  with a different shape entirely, which is why no committed tool could
  reproduce `results/baseline_bench_2026-06.json`.
* Check the arithmetic in `key_findings`. Rule 3 in `results/README.md` says a
  percentage must follow from the numbers in the same file. The README's own
  history shows why: a "~25% lower TTFT" claim sat next to numbers that gave
  31%. Findings are now derived from the measurements rather than written by
  hand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from evals.provenance import Provenance

# Every field a committed result must carry, per results/README.md.
REQUIRED_TOP_LEVEL = (
    "benchmark_date",
    "model",
    "hardware",
    "git_sha",
    "command",
    "software_versions",
    "workload",
    "raw_output_path",
    "backends",
    "key_findings",
)

REQUIRED_WORKLOAD = (
    "concurrency",
    "request_count",
    "prompt_tokens_avg",
    "output_tokens_avg",
)

REQUIRED_BACKEND = (
    "ttft_p50_ms",
    "ttft_p99_ms",
    "throughput_tokens_per_sec",
    "requests_per_sec",
    "error_rate_pct",
    "gpu_util_avg_pct",
)


class ReportValidationError(ValueError):
    """A report does not satisfy the committed schema."""


@dataclass
class BackendMeasurement:
    """One backend's measured numbers."""

    key: str
    ttft_p50_ms: float
    ttft_p99_ms: float
    throughput_tokens_per_sec: float
    requests_per_sec: float
    error_rate_pct: float
    gpu_util_avg_pct: float | None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ttft_p50_ms": round(self.ttft_p50_ms, 1),
            "ttft_p99_ms": round(self.ttft_p99_ms, 1),
            "throughput_tokens_per_sec": round(self.throughput_tokens_per_sec, 1),
            "requests_per_sec": round(self.requests_per_sec, 2),
            "error_rate_pct": round(self.error_rate_pct, 2),
            "gpu_util_avg_pct": self.gpu_util_avg_pct,
            "notes": self.notes,
        }


@dataclass
class Workload:
    """What was sent."""

    concurrency: int
    request_count: int
    prompt_tokens_avg: int
    output_tokens_avg: int

    def to_dict(self) -> dict[str, int]:
        return {
            "concurrency": self.concurrency,
            "request_count": self.request_count,
            "prompt_tokens_avg": self.prompt_tokens_avg,
            "output_tokens_avg": self.output_tokens_avg,
        }


@dataclass
class BenchmarkReport:
    """A full run, ready to serialise into results/."""

    model: str
    workload: Workload
    provenance: Provenance
    backends: list[BackendMeasurement]
    raw_output_path: str
    benchmark_date: str = field(default_factory=lambda: date.today().isoformat())

    def key_findings(self) -> list[str]:
        """Findings derived from the measurements, so they cannot contradict them."""
        findings: list[str] = []
        if len(self.backends) < 2:
            return findings

        ranked = sorted(self.backends, key=lambda b: b.ttft_p50_ms)
        fastest, slowest = ranked[0], ranked[-1]
        if slowest.ttft_p50_ms > 0 and fastest.key != slowest.key:
            delta = slowest.ttft_p50_ms - fastest.ttft_p50_ms
            pct = delta / slowest.ttft_p50_ms * 100
            findings.append(
                f"{fastest.key} delivers {pct:.0f}% lower TTFT p50 than "
                f"{slowest.key} at concurrency {self.workload.concurrency} "
                f"({slowest.ttft_p50_ms:.0f} ms -> {fastest.ttft_p50_ms:.0f} ms)."
            )

        by_throughput = sorted(
            self.backends, key=lambda b: b.throughput_tokens_per_sec, reverse=True
        )
        best, worst = by_throughput[0], by_throughput[-1]
        if worst.throughput_tokens_per_sec > 0 and best.key != worst.key:
            pct = (
                (best.throughput_tokens_per_sec - worst.throughput_tokens_per_sec)
                / worst.throughput_tokens_per_sec
                * 100
            )
            findings.append(
                f"{best.key} sustains {pct:.0f}% higher token throughput than "
                f"{worst.key} ({worst.throughput_tokens_per_sec:.0f} -> "
                f"{best.throughput_tokens_per_sec:.0f} tok/s)."
            )

        errored = [b for b in self.backends if b.error_rate_pct > 0]
        if errored:
            detail = ", ".join(f"{b.key} {b.error_rate_pct:.2f}%" for b in errored)
            findings.append(f"Non-zero error rate observed: {detail}.")
        else:
            findings.append("All backends completed every request without error.")

        return findings

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "benchmark_date": self.benchmark_date,
            "model": self.model,
        }
        payload.update(self.provenance.to_dict())
        payload["workload"] = self.workload.to_dict()
        payload["raw_output_path"] = self.raw_output_path
        payload["backends"] = {b.key: b.to_dict() for b in self.backends}
        payload["key_findings"] = self.key_findings()
        return payload


def validate_report(payload: dict[str, Any]) -> list[str]:
    """Return the reasons `payload` fails the committed schema, empty if it passes.

    Used by the harness before writing, and by tests against the files in
    `results/`, so a result that cannot be reproduced cannot be committed
    unnoticed.
    """
    problems: list[str] = []

    for key in REQUIRED_TOP_LEVEL:
        if key not in payload:
            problems.append(f"missing required field '{key}'")
        elif payload[key] in (None, "", [], {}):
            problems.append(f"required field '{key}' is empty")

    workload = payload.get("workload")
    if isinstance(workload, dict):
        for key in REQUIRED_WORKLOAD:
            if key not in workload:
                problems.append(f"workload missing '{key}'")
    elif "workload" in payload:
        problems.append("'workload' must be an object")

    versions = payload.get("software_versions")
    if isinstance(versions, dict):
        recorded = [k for k, v in versions.items() if v and k != "python"]
        if not recorded:
            problems.append(
                "software_versions records no backend version; the run cannot be "
                "reproduced from it"
            )
    elif "software_versions" in payload:
        problems.append("'software_versions' must be an object")

    backends = payload.get("backends")
    if isinstance(backends, dict):
        if not backends:
            problems.append("'backends' is empty")
        for name, backend in backends.items():
            if not isinstance(backend, dict):
                problems.append(f"backend '{name}' must be an object")
                continue
            for key in REQUIRED_BACKEND:
                if key not in backend:
                    problems.append(f"backend '{name}' missing '{key}'")
    elif "backends" in payload:
        problems.append("'backends' must be an object")

    problems.extend(_check_finding_arithmetic(payload))
    return problems


def _check_finding_arithmetic(payload: dict[str, Any]) -> list[str]:
    """Check percentage claims in key_findings against the numbers beside them.

    Only findings written in the harness's own "A -> B" form are checkable; a
    hand-written prose finding is left alone.
    """
    import re

    problems: list[str] = []
    findings = payload.get("key_findings")
    if not isinstance(findings, list):
        return problems

    pattern = re.compile(
        r"(\d+(?:\.\d+)?)%\s+(lower|higher).*?\(\s*(\d+(?:\.\d+)?)\s*"
        r"(?:ms|tok/s)?\s*->\s*(\d+(?:\.\d+)?)"
    )
    for finding in findings:
        if not isinstance(finding, str):
            continue
        match = pattern.search(finding)
        if not match:
            continue
        claimed, _direction, before, after = match.groups()
        before_f, after_f = float(before), float(after)
        # Both forms read "X is N% <lower|higher> than Y (Y_value -> X_value)",
        # so the baseline is always the first number -- the value being
        # compared against.
        if before_f == 0:
            continue
        actual = abs(before_f - after_f) / before_f * 100
        if abs(actual - float(claimed)) > 1.0:
            problems.append(
                f"key_finding claims {claimed}% but {before} -> {after} is "
                f"{actual:.0f}%: {finding!r}"
            )
    return problems


def write_report(report: BenchmarkReport, path: Path) -> Path:
    """Validate and write a report. Refuses to write one that is not reproducible."""
    payload = report.to_dict()
    problems = validate_report(payload)
    if problems:
        raise ReportValidationError(
            "refusing to write a report that does not satisfy "
            "results/README.md:\n  - " + "\n  - ".join(problems)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_raw(records: Iterable[dict[str, Any]], directory: Path) -> Path:
    """Write per-request records so aggregates can be recomputed (rule 2)."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "requests.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path
