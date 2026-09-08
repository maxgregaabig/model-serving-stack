"""Benchmark harness that emits the schema `results/README.md` requires.

`evals/benchmark.py` measures end-to-end request latency against one endpoint
and reports avg/p99 plus requests/sec. The committed baseline reports per-backend
TTFT percentiles and token throughput, so no committed tool could produce it.
This module closes that gap:

* Measures **time to first token** from a streamed response, rather than total
  request duration.
* Runs **one or more named backends** in a single report, so a comparison
  between them comes from one run under one workload.
* Captures **provenance** -- commit, command, package and driver versions,
  detected hardware -- and refuses to write a report missing any of it.
* Writes **raw per-request records** alongside the summary so the aggregates can
  be recomputed.

Usage:

    python -m evals.bench_report \\
        --backend vllm_paged_attention=http://localhost:8080/v1/chat/completions \\
        --backend triton_trt_llm=http://localhost:8081/v1/chat/completions \\
        --model meta/llama3-8b-instruct \\
        --requests 500 --concurrency 32 --output results/baseline_bench_2026-09.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from evals.provenance import Provenance
from evals.report import (
    BackendMeasurement,
    BenchmarkReport,
    ReportValidationError,
    Workload,
    write_raw,
    write_report,
)

logger = logging.getLogger(__name__)


@dataclass
class RequestRecord:
    """One request's observed timings."""

    backend: str
    index: int
    ok: bool
    ttft_ms: float | None
    total_ms: float
    output_tokens: int
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "index": self.index,
            "ok": self.ok,
            "ttft_ms": None if self.ttft_ms is None else round(self.ttft_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "output_tokens": self.output_tokens,
            "error": self.error,
        }


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile.

    `evals/benchmark.py` indexes `sorted[int(0.99 * n)]`, which for n=100 returns
    the maximum rather than the 99th percentile. This does not.
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


async def _stream_one(
    client: httpx.AsyncClient,
    backend: str,
    endpoint: str,
    index: int,
    payload: dict[str, Any],
    timeout: float,
) -> RequestRecord:
    """Issue one streamed completion and time the first token."""
    start = time.perf_counter()
    ttft: float | None = None
    tokens = 0
    try:
        async with client.stream(
            "POST", endpoint, json=payload, timeout=timeout
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                if ttft is None:
                    # First streamed chunk carrying content is the first token.
                    ttft = (time.perf_counter() - start) * 1000
                tokens += 1
        total = (time.perf_counter() - start) * 1000
        return RequestRecord(backend, index, True, ttft, total, tokens)
    except Exception as exc:  # noqa: BLE001 - any failure is a failed request
        total = (time.perf_counter() - start) * 1000
        return RequestRecord(
            backend, index, False, None, total, 0, f"{type(exc).__name__}: {exc}"
        )


async def run_backend(
    *,
    backend: str,
    endpoint: str,
    num_requests: int,
    concurrency: int,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> tuple[BackendMeasurement, list[RequestRecord]]:
    """Drive one backend and summarise it."""
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(client: httpx.AsyncClient, index: int) -> RequestRecord:
        async with semaphore:
            return await _stream_one(client, backend, endpoint, index, payload, timeout)

    started = time.perf_counter()
    async with httpx.AsyncClient() as client:
        records = await asyncio.gather(
            *[bounded(client, i) for i in range(num_requests)]
        )
    elapsed = time.perf_counter() - started

    successful = [r for r in records if r.ok]
    ttfts = [r.ttft_ms for r in successful if r.ttft_ms is not None]
    total_tokens = sum(r.output_tokens for r in successful)

    measurement = BackendMeasurement(
        key=backend,
        ttft_p50_ms=percentile(ttfts, 50),
        ttft_p99_ms=percentile(ttfts, 99),
        throughput_tokens_per_sec=total_tokens / elapsed if elapsed > 0 else 0.0,
        requests_per_sec=len(successful) / elapsed if elapsed > 0 else 0.0,
        error_rate_pct=(len(records) - len(successful)) / len(records) * 100
        if records
        else 0.0,
        # Read from DCGM/nvidia-smi during the run, or left unrecorded rather
        # than invented. See gpu_util_avg_pct in results/README.md.
        gpu_util_avg_pct=None,
        notes=f"endpoint={endpoint}, max_tokens={max_tokens}",
    )
    return measurement, list(records)


def _parse_backend(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--backend expects name=url, got {spec!r}"
        )
    name, url = spec.split("=", 1)
    if not name or not url:
        raise argparse.ArgumentTypeError(f"--backend expects name=url, got {spec!r}")
    return name, url


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals.bench_report",
        description=(
            "Benchmark one or more serving backends and emit a provenance-complete "
            "report in the schema results/README.md defines."
        ),
    )
    parser.add_argument(
        "--backend",
        action="append",
        required=True,
        # Parsed here so a malformed spec is an argparse usage error rather than
        # a traceback partway through a benchmark run.
        type=_parse_backend,
        metavar="NAME=URL",
        help="Backend to measure, repeatable (e.g. vllm_paged_attention=http://...).",
    )
    parser.add_argument("--model", required=True, help="Model identifier under test.")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--prompt", default="Explain tensor parallelism briefly.")
    parser.add_argument(
        "--prompt-tokens-avg",
        type=int,
        default=0,
        help="Average prompt tokens, for the workload block. Measure it; do not guess.",
    )
    parser.add_argument(
        "--hardware",
        default=None,
        help="Override the detected hardware string. Detection is preferred.",
    )
    parser.add_argument(
        "--triton-server-image",
        default=None,
        help="Exact Triton container image tag, when Triton is one of the backends.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Where to write the report, e.g. results/baseline_bench_2026-09.json.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Directory for raw per-request records. Defaults to results/raw/<stem>/.",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    backends: list[tuple[str, str]] = args.backend

    measurements: list[BackendMeasurement] = []
    all_records: list[RequestRecord] = []
    for name, url in backends:
        logger.info("Benchmarking %s at %s", name, url)
        measurement, records = await run_backend(
            backend=name,
            endpoint=url,
            num_requests=args.requests,
            concurrency=args.concurrency,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        measurements.append(measurement)
        all_records.extend(records)
        logger.info(
            "%s: ttft_p50=%.1fms rps=%.2f errors=%.2f%%",
            name,
            measurement.ttft_p50_ms,
            measurement.requests_per_sec,
            measurement.error_rate_pct,
        )

    raw_dir = args.raw_dir or Path("results/raw") / args.output.stem
    write_raw((r.to_dict() for r in all_records), raw_dir)

    provenance = Provenance.capture(
        hardware_override=args.hardware,
        triton_server_image=args.triton_server_image,
    )
    missing = provenance.missing_fields()
    if missing:
        logger.error(
            "Refusing to write a report: could not observe %s. "
            "A result without these is not reproducible (results/README.md rule 1). "
            "Pass --hardware if detection is unavailable on this host.",
            ", ".join(missing),
        )
        return 2

    observed_output_tokens = [r.output_tokens for r in all_records if r.ok]
    report = BenchmarkReport(
        model=args.model,
        workload=Workload(
            concurrency=args.concurrency,
            request_count=args.requests,
            prompt_tokens_avg=args.prompt_tokens_avg,
            output_tokens_avg=(
                int(statistics.mean(observed_output_tokens))
                if observed_output_tokens
                else 0
            ),
        ),
        provenance=provenance,
        backends=measurements,
        raw_output_path=str(raw_dir),
    )

    try:
        path = write_report(report, args.output)
    except ReportValidationError as exc:
        logger.error("%s", exc)
        return 2

    logger.info("Wrote %s", path)
    print(json.dumps(report.to_dict()["key_findings"], indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
