# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- **`results/baseline_bench_2026-06.json` is retired.** It recorded no
  `git_sha`, no `command` and no `software_versions`, no committed tool emitted
  its schema, and the raw output was never kept — so nothing in it could be
  reproduced or checked. Under this repository's own rule
  ("A result without `git_sha`, `command`, and `software_versions` is not
  reproducible and should not be committed", `results/README.md`) it did not
  belong in `results/`. It has moved to `results/unverified/`, which states
  plainly that nothing there is evidence.
- Every performance number that rested on it: the benchmark table and the
  "~31% lower TTFT" takeaway in `README.md`, the per-backend table in
  `docs/benchmarks.md`, and the Triton-versus-vLLM comparison in
  `docs/architecture.md`. **This project now publishes no performance numbers.**
  It has a harness; it does not yet have a result.

### Added

- `evals/bench_report.py` — a benchmark harness that emits the schema
  `results/README.md` defines. It measures **time to first token** from a
  streamed response (the old harness measured total request latency), drives
  **several named backends** in one run so a comparison comes from one workload,
  and writes raw per-request records so aggregates can be recomputed.
- `evals/provenance.py` — captures commit SHA (suffixed `-dirty` when the tree
  is), the exact shell-quoted invocation, installed package versions, CUDA
  driver and detected GPU. Values that cannot be observed are recorded as
  `null` rather than guessed, and the harness **refuses to write a report**
  whose provenance is incomplete.
- `evals/report.py` — schema validation, plus an arithmetic check on
  `key_findings`. `key_findings` are now derived from the measurements, so a
  percentage cannot contradict the numbers beside it. This is the defect that
  put "~25% lower TTFT" next to numbers giving 31%.
- `tests/test_bench_report.py` — 35 tests, including one asserting that every
  `*.json` in `results/` satisfies the schema, so an unreproducible result
  cannot land there unnoticed.
- `results/unverified/README.md` — what is in there, why, and the command that
  would produce a real replacement.

### Fixed

- `percentile()` replaces `sorted[int(0.99 * n)]`, which returned the maximum
  rather than the 99th percentile for n=100.
- `--backend` without an `=` is now an argparse usage error rather than a
  traceback raised partway through a run.


- `LICENSE` — MIT, the license the README badge and link already claimed.
- `docs/README.md` — documentation index.
- `docs/benchmarks.md` — committed baseline figures, exact rerun commands for
  `evals/benchmark.py`, `evals/load_test.py` and `triton/perf_analyzer.sh`, and an
  explicit note on the provenance metadata missing from the committed results.
- `docs/configuration.md` — environment variables actually read by code, which
  `.env.template` entries nothing in the repo consumes, YAML file ownership, and
  worked single-model / multi-model / multi-backend examples.
- `docs/autoscaling.md` — canonical DCGM-driven HPA path, Prometheus Adapter
  dependency, apply order, and tuning guidance.
- `docs/testing.md` — test-to-module coverage map, uncovered areas, and the `gpu`
  marker convention.
- `results/README.md` — required metadata schema for future benchmark artifacts.
- `SECURITY.md` and `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1).
- `CHANGELOG.md` (this file).
- `[project.optional-dependencies] dev` in `pyproject.toml`, which
  `pip install -e ".[dev]"` had always required but never had.
- Release process section in `CONTRIBUTING.md`.

### Changed

- README rewritten for accuracy: real project-structure tree, corrected
  environment variable table, Quick Start matching the actual Compose files and
  gateway port, Kubernetes commands referencing manifests that exist, and a
  development section matching CI exactly.
- README reframed from "production LLM serving infrastructure" to a reference
  stack, and the portfolio framing reworded to remove any implied employer
  affiliation.
- README mermaid diagram now uses `<br/>` line breaks, which GitHub renders,
  instead of literal `\n`.
- `docs/architecture.md` corrected to reference files that exist and to describe
  the gateway's actual single-backend behaviour.
- `pyproject.toml` `build-backend` fixed to `setuptools.build_meta`; the previous
  value was not a real setuptools entry point.
- `deploy/docker-compose.yml` Prometheus mount now points at
  `monitoring/prometheus/prometheus.yml`.
- `.gitignore` narrowed from `results/` to `results/load_tests/` so committed
  benchmark artifacts and documentation are tracked.

### Fixed

- Benchmark takeaway in the README said "~25% lower TTFT" while the committed
  `results/baseline_bench_2026-06.json` records ~31% (142 ms -> 98 ms). Both now
  say 31%.

### Removed

- `autoscaling/hpa.yaml`, a duplicate of `kubernetes/hpa-dcgm.yaml` that used a
  metric name the DCGM Exporter does not publish. `kubernetes/hpa-dcgm.yaml` is
  the single canonical HPA manifest.

## [0.1.0] - 2026-06-09

### Added

- FastAPI gateway (`api/`) with OpenAI-compatible `POST /v1/chat/completions`,
  `GET /v1/models`, `GET /health`, and `StructuredLoggingMiddleware` adding
  `X-Request-ID` and `X-Backend` headers.
- vLLM integration: launcher (`vllm/server.py`), in-process OpenAI-compatible
  server (`vllm/openai_server.py`), client (`vllm/client.py`), and batching
  configuration reference.
- Triton integration: Python-backend model repository for LLaMA 3, client
  wrapper, and a `perf_analyzer` benchmark script.
- Ray Serve deployment and multi-model traffic-splitting router.
- BentoML packaging service and build script.
- Kubernetes manifests: vLLM and Triton Deployments plus Services, NGINX
  Ingress, DCGM Exporter DaemonSet, and a DCGM-driven HPA.
- Prometheus scrape configuration and a Grafana GPU serving dashboard.
- Benchmark harnesses (`evals/benchmark.py`, `evals/load_test.py`) and the
  committed baseline `results/baseline_bench_2026-06.json`.
- Docker Compose stacks at the repository root and under `deploy/`.
- Test suite (`tests/`) and a GitHub Actions CI workflow running `ruff` plus
  `pytest` with `--cov-fail-under=80`.
