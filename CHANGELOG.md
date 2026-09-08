# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
