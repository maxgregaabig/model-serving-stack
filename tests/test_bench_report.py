"""Tests for the provenance-complete benchmark harness.

These cover the two failures that made `results/baseline_bench_2026-06.json`
unusable: a report with no provenance, and a key_finding whose percentage does
not follow from the numbers beside it.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from evals.bench_report import build_parser, percentile
from evals.provenance import Provenance, invoked_command, package_versions
from evals.report import (
    BackendMeasurement,
    BenchmarkReport,
    ReportValidationError,
    Workload,
    validate_report,
    write_raw,
    write_report,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _provenance() -> Provenance:
    return Provenance(
        git_sha="0123456789abcdef0123456789abcdef01234567",
        command="python -m evals.bench_report --requests 500",
        software_versions={"vllm": "0.4.2", "python": "3.11.9", "cuda_driver": "535.104.05"},
        hardware="NVIDIA A10G (24GB), 8 vCPU, 32GB RAM",
    )


def _report(backends: list[BackendMeasurement]) -> BenchmarkReport:
    return BenchmarkReport(
        model="meta/llama3-8b-instruct",
        workload=Workload(
            concurrency=32, request_count=500, prompt_tokens_avg=128, output_tokens_avg=256
        ),
        provenance=_provenance(),
        backends=backends,
        raw_output_path="results/raw/example",
        benchmark_date="2026-09-08",
    )


def _backend(key: str, ttft_p50: float, throughput: float = 1000.0) -> BackendMeasurement:
    return BackendMeasurement(
        key=key,
        ttft_p50_ms=ttft_p50,
        ttft_p99_ms=ttft_p50 * 2.5,
        throughput_tokens_per_sec=throughput,
        requests_per_sec=28.4,
        error_rate_pct=0.0,
        gpu_util_avg_pct=87,
    )


class TestPercentile:
    """The p99 the old harness computed was the maximum."""

    def test_p50_of_a_known_series(self):
        assert percentile([1, 2, 3, 4, 5], 50) == 3

    def test_p99_is_not_simply_the_maximum(self):
        values = list(range(1, 101))  # 1..100
        # sorted[int(0.99 * 100)] == sorted[99] == 100, the maximum.
        assert percentile(values, 99) < 100
        assert percentile(values, 99) == pytest.approx(99.01, abs=0.05)

    def test_empty_and_single(self):
        assert percentile([], 99) == 0.0
        assert percentile([42.0], 99) == 42.0


class TestProvenance:
    def test_command_is_shell_quoted(self):
        command = invoked_command(["prog", "--model", "a b"])
        assert "evals.bench_report" in command
        assert "'a b'" in command

    def test_package_versions_records_python_and_nulls_absent_packages(self):
        versions = package_versions()
        assert versions["python"]
        # tensorrt_llm is not installed in this environment; it must read as a
        # recorded null rather than be omitted or guessed.
        assert "tensorrt_llm" in versions

    def test_missing_fields_reports_what_was_not_observed(self):
        bare = Provenance(
            git_sha=None,
            command="",
            software_versions={"python": "3.11.9"},
            hardware=None,
        )
        missing = bare.missing_fields()
        assert set(missing) == {"git_sha", "command", "hardware", "software_versions"}

    def test_capture_notes_absence_of_a_gpu(self):
        captured = Provenance.capture(argv=["prog"], repo_root=str(REPO_ROOT))
        if captured.hardware is None:
            assert any("did not measure GPU serving" in n for n in captured.notes)

    def test_capture_reads_the_repo_sha(self):
        captured = Provenance.capture(argv=["prog"], repo_root=str(REPO_ROOT))
        # This repository is a git checkout, so a SHA must be observable.
        assert captured.git_sha
        assert len(captured.git_sha.removesuffix("-dirty")) == 40


class TestFindingsAreDerived:
    """Rule 3: a percentage must follow from the numbers in the same file."""

    def test_ttft_finding_matches_the_measurements(self):
        report = _report([_backend("vllm", 142), _backend("triton", 98)])
        findings = report.key_findings()
        # 142 -> 98 is a 31% reduction. The README once claimed 25%.
        assert any("31% lower TTFT p50" in f for f in findings)
        assert any("142 ms -> 98 ms" in f for f in findings)

    def test_no_comparison_finding_for_a_single_backend(self):
        report = _report([_backend("vllm", 142)])
        assert not any("lower TTFT" in f for f in report.key_findings())

    def test_error_rates_are_reported_when_non_zero(self):
        bad = _backend("vllm", 142)
        bad.error_rate_pct = 1.5
        findings = _report([bad, _backend("triton", 98)]).key_findings()
        assert any("Non-zero error rate" in f for f in findings)

    def test_clean_run_says_so(self):
        findings = _report([_backend("vllm", 142), _backend("triton", 98)]).key_findings()
        assert any("without error" in f for f in findings)


class TestValidation:
    def test_a_complete_report_validates(self):
        payload = _report([_backend("vllm", 142), _backend("triton", 98)]).to_dict()
        assert validate_report(payload) == []

    @pytest.mark.parametrize(
        "field", ["git_sha", "command", "software_versions", "raw_output_path", "workload"]
    )
    def test_missing_provenance_is_rejected(self, field):
        payload = _report([_backend("vllm", 142)]).to_dict()
        del payload[field]
        problems = validate_report(payload)
        assert any(field in p for p in problems)

    def test_software_versions_without_a_backend_version_is_rejected(self):
        payload = _report([_backend("vllm", 142)]).to_dict()
        payload["software_versions"] = {"python": "3.11.9"}
        problems = validate_report(payload)
        assert any("cannot be reproduced" in p for p in problems)

    def test_backend_missing_a_metric_is_rejected(self):
        payload = _report([_backend("vllm", 142)]).to_dict()
        del payload["backends"]["vllm"]["ttft_p50_ms"]
        assert any("ttft_p50_ms" in p for p in validate_report(payload))

    def test_a_finding_that_contradicts_its_numbers_is_rejected(self):
        """The exact defect the audit caught: 25% claimed where 31% was measured."""
        payload = _report([_backend("vllm", 142), _backend("triton", 98)]).to_dict()
        payload["key_findings"] = [
            "triton delivers 25% lower TTFT p50 than vllm at concurrency 32 "
            "(142 ms -> 98 ms)."
        ]
        problems = validate_report(payload)
        assert any("claims 25%" in p and "31%" in p for p in problems)

    def test_higher_comparisons_use_the_first_number_as_the_baseline(self):
        """"X is N% higher than Y (Y -> X)" is measured against Y, not X.

        The generator and validator once disagreed here, so a report the
        harness produced failed its own validation.
        """
        payload = _report(
            [_backend("vllm", 142, throughput=2000), _backend("triton", 98, throughput=2600)]
        ).to_dict()
        # 2000 -> 2600 is +30% of 2000, not +23% of 2600.
        assert any("30% higher token throughput" in f for f in payload["key_findings"])
        assert validate_report(payload) == []

    def test_a_finding_that_agrees_with_its_numbers_passes(self):
        payload = _report([_backend("vllm", 142), _backend("triton", 98)]).to_dict()
        payload["key_findings"] = [
            "triton delivers 31% lower TTFT p50 than vllm at concurrency 32 "
            "(142 ms -> 98 ms)."
        ]
        assert validate_report(payload) == []


class TestWriting:
    def test_write_report_round_trips(self, tmp_path):
        path = tmp_path / "bench.json"
        write_report(_report([_backend("vllm", 142), _backend("triton", 98)]), path)
        payload = json.loads(path.read_text())
        assert payload["git_sha"]
        assert payload["command"]
        assert set(payload["backends"]) == {"vllm", "triton"}
        assert validate_report(payload) == []

    def test_write_report_refuses_an_unreproducible_report(self, tmp_path):
        report = _report([_backend("vllm", 142)])
        report.provenance.git_sha = None
        with pytest.raises(ReportValidationError, match="git_sha"):
            write_report(report, tmp_path / "bench.json")

    def test_raw_records_are_written_one_per_line(self, tmp_path):
        path = write_raw(
            [{"backend": "vllm", "index": 0}, {"backend": "vllm", "index": 1}],
            tmp_path / "raw",
        )
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[1])["index"] == 1


class TestCLI:
    def test_backend_requires_name_equals_url(self):
        parser = build_parser()
        args = parser.parse_args(
            ["--backend", "vllm=http://localhost:8080", "--model", "m", "--output", "o.json"]
        )
        assert args.backend == [("vllm", "http://localhost:8080")]

    def test_output_and_model_are_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--backend", "vllm=http://x"])


class TestCommittedResults:
    """Every JSON committed under results/ must satisfy the schema it documents."""

    def test_committed_results_are_reproducible(self):
        results_dir = REPO_ROOT / "results"
        offenders = {}
        for path in sorted(results_dir.glob("*.json")):
            payload = json.loads(path.read_text())
            problems = validate_report(payload)
            if problems:
                offenders[path.name] = problems
        assert not offenders, (
            "results/ holds files that cannot be reproduced from the repository. "
            "Per results/README.md rule 1 they do not belong there; move them to "
            f"results/unverified/. Offenders: {json.dumps(offenders, indent=2)}"
        )


class TestStreamingMeasurement:
    """The async runner, driven through httpx's MockTransport -- no network."""

    @staticmethod
    def _sse_transport(chunks: int = 3, status: int = 200) -> httpx.MockTransport:
        body = "".join(
            f'data: {{"choices":[{{"delta":{{"content":"tok{i}"}}}}]}}\n\n'
            for i in range(chunks)
        ) + "data: [DONE]\n\n"

        def handler(request: httpx.Request) -> httpx.Response:
            if status != 200:
                return httpx.Response(status, text="upstream error")
            return httpx.Response(200, text=body)

        return httpx.MockTransport(handler)

    async def test_measures_ttft_and_throughput(self, monkeypatch):
        from evals import bench_report

        transport = self._sse_transport(chunks=4)
        original = httpx.AsyncClient

        def patched(*args, **kwargs):
            kwargs["transport"] = transport
            return original(*args, **kwargs)

        monkeypatch.setattr(bench_report.httpx, "AsyncClient", patched)

        measurement, records = await bench_report.run_backend(
            backend="vllm",
            endpoint="http://localhost:8080/v1/chat/completions",
            num_requests=5,
            concurrency=2,
            prompt="hi",
            max_tokens=16,
            timeout=10.0,
        )

        assert measurement.key == "vllm"
        assert measurement.error_rate_pct == 0.0
        assert measurement.ttft_p50_ms > 0
        assert measurement.requests_per_sec > 0
        assert len(records) == 5
        assert all(r.ok for r in records)
        # Four content chunks per request, [DONE] excluded.
        assert all(r.output_tokens == 4 for r in records)
        assert all(r.ttft_ms is not None for r in records)

    async def test_failures_are_counted_not_raised(self, monkeypatch):
        from evals import bench_report

        transport = self._sse_transport(status=503)
        original = httpx.AsyncClient

        def patched(*args, **kwargs):
            kwargs["transport"] = transport
            return original(*args, **kwargs)

        monkeypatch.setattr(bench_report.httpx, "AsyncClient", patched)

        measurement, records = await bench_report.run_backend(
            backend="vllm",
            endpoint="http://localhost:8080/v1/chat/completions",
            num_requests=4,
            concurrency=2,
            prompt="hi",
            max_tokens=16,
            timeout=10.0,
        )

        assert measurement.error_rate_pct == 100.0
        assert measurement.ttft_p50_ms == 0.0
        assert all(not r.ok for r in records)
        assert all(r.error for r in records)

    def test_request_record_serialises(self):
        from evals.bench_report import RequestRecord

        record = RequestRecord("vllm", 0, True, 12.3456, 99.9999, 4)
        payload = record.to_dict()
        assert payload["ttft_ms"] == 12.35
        assert payload["total_ms"] == 100.0
        assert payload["error"] is None


class TestCLIRun:
    """End-to-end through main(), still with no network."""

    @staticmethod
    def _patch_transport(monkeypatch):
        from evals import bench_report

        body = 'data: {"choices":[{"delta":{"content":"a"}}]}\n\ndata: [DONE]\n\n'
        transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
        original = httpx.AsyncClient

        def patched(*args, **kwargs):
            kwargs["transport"] = transport
            return original(*args, **kwargs)

        monkeypatch.setattr(bench_report.httpx, "AsyncClient", patched)

    def test_writes_a_valid_report(self, tmp_path, monkeypatch):
        from evals.bench_report import main

        self._patch_transport(monkeypatch)
        out = tmp_path / "bench.json"
        raw = tmp_path / "raw"

        code = main([
            "--backend", "vllm=http://localhost:8080/v1/chat/completions",
            "--backend", "triton=http://localhost:8081/v1/chat/completions",
            "--model", "meta/llama3-8b-instruct",
            "--requests", "4",
            "--concurrency", "2",
            "--hardware", "NVIDIA A10G (24GB), 8 vCPU, 32GB RAM",
            "--output", str(out),
            "--raw-dir", str(raw),
        ])

        assert code == 0
        payload = json.loads(out.read_text())
        assert validate_report(payload) == []
        assert set(payload["backends"]) == {"vllm", "triton"}
        assert payload["workload"]["request_count"] == 4
        assert payload["git_sha"]
        assert "evals.bench_report" in payload["command"]
        assert (raw / "requests.jsonl").exists()
        assert len((raw / "requests.jsonl").read_text().strip().splitlines()) == 8

    def test_refuses_to_write_without_observable_hardware(self, tmp_path, monkeypatch):
        """No GPU and no --hardware means the run did not measure GPU serving."""
        from evals import bench_report
        from evals.bench_report import main

        self._patch_transport(monkeypatch)
        monkeypatch.setattr(bench_report.Provenance, "capture", staticmethod(
            lambda **kwargs: Provenance(
                git_sha="abc", command="cmd", software_versions={"python": "3.11"},
                hardware=None,
            )
        ))
        out = tmp_path / "bench.json"

        code = main([
            "--backend", "vllm=http://localhost:8080/v1/chat/completions",
            "--model", "m", "--requests", "2", "--concurrency", "1",
            "--output", str(out), "--raw-dir", str(tmp_path / "raw"),
        ])

        assert code == 2
        assert not out.exists()

    def test_rejects_a_malformed_backend_spec(self, tmp_path, monkeypatch):
        from evals.bench_report import main

        self._patch_transport(monkeypatch)
        with pytest.raises(SystemExit):
            main([
                "--backend", "no-equals-sign",
                "--model", "m",
                "--output", str(tmp_path / "b.json"),
            ])
