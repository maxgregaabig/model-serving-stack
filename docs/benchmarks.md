# Benchmarks

The one committed benchmark artifact is
[`results/baseline_bench_2026-06.json`](../results/baseline_bench_2026-06.json).
Everything below either comes from that file or describes how to run the
harnesses in `evals/`.

## Committed baseline (2026-06-09)

From the JSON's top-level fields:

| Field | Value |
|-------|-------|
| `benchmark_date` | `2026-06-09` |
| `model` | `meta/llama3-8b-instruct` |
| `hardware` | NVIDIA A10G (24GB), 8 vCPU, 32GB RAM |
| `concurrency` | 32 |
| `request_count` | 500 |
| `prompt_tokens_avg` | 128 |
| `output_tokens_avg` | 256 |

Per-backend results:

| Backend (JSON key) | TTFT p50 (ms) | TTFT p99 (ms) | Throughput (tok/s) | Req/s | Error rate | Avg GPU util |
|---|---|---|---|---|---|---|
| `vllm_paged_attention` | 142 | 387 | 1840 | 28.4 | 0.0% | 87% |
| `triton_trt_llm` | 98 | 241 | 2310 | 35.7 | 0.0% | 91% |
| `ray_serve_vllm_backend` | 156 | 412 | 1780 | 27.5 | 0.0% | 85% |

Recorded `key_findings`:

- TRT-LLM via Triton delivers ~31% lower TTFT p50 than vLLM at concurrency 32
  (142 ms -> 98 ms).
- vLLM's PagedAttention advantage becomes dominant at concurrency >= 64 due to
  memory efficiency.
- Ray Serve adds ~10% overhead versus native vLLM from actor scheduling.
- The DCGM-driven HPA triggered scale-out at 70% GPU utilization and stayed
  stable under sustained load.

## Reproducibility gap

**These numbers cannot currently be reproduced from the repository alone.** The
committed JSON records results but not the conditions that produced them, and no
script in this repo emits that schema. Specifically:

- **No `git_sha`** — the code state that produced the run is unknown.
- **No `command`** — the exact invocation, including flags, is unknown.
- **No `software_versions`** — vLLM, TensorRT-LLM, Triton, CUDA driver and
  container image tags are all unrecorded. `requirements.txt` pins only lower
  bounds (`vllm>=0.4.0`, `tritonclient[all]>=2.44.0`, `ray[serve]>=2.10.0`), so it
  does not identify the versions used.
- **No raw output path** — only the aggregated summary is committed.
- **Schema mismatch with the harnesses** — the JSON reports per-backend TTFT
  percentiles and token throughput. `evals/benchmark.py` measures end-to-end
  request latency (avg, p99) and requests/sec, not TTFT or tokens/sec, and
  `triton/perf_analyzer.sh` emits `perf_analyzer` output in its own format. No
  committed tool produces this file.

Treat the table above as a recorded historical measurement, not a verified
claim. Any future run should be written with the metadata schema in
[`results/README.md`](../results/README.md) so it is reproducible.

## Rerunning the harnesses

Both harnesses drive the FastAPI gateway (`api/main.py`), which listens on port
`8080` and proxies to vLLM via `vllm/client.py`. Start the gateway and a backend
first — see [configuration.md](configuration.md).

### `evals/benchmark.py`

No CLI flags. Defaults are hardcoded in `run_benchmark()`: endpoint
`http://localhost:8080/v1/chat/completions`, `num_requests=100`,
`concurrency=10`, prompt `"What is machine learning?"`.

```bash
python -m evals.benchmark
```

Prints throughput (req/s), average latency, p99 latency, and failure count to
stdout. **Writes no file.**

### `evals/load_test.py`

Wraps `run_benchmark()` and persists a JSON report. Real flags, from its
`argparse` definition:

| Flag | Default |
|------|---------|
| `--endpoint` | `http://localhost:8080/v1/chat/completions` |
| `--requests` | `20` |
| `--concurrency` | `4` |
| `--prompt` | `Explain tensor parallelism briefly.` |
| `--output-dir` | `results/load_tests` |

```bash
python -m evals.load_test \
  --endpoint http://localhost:8080/v1/chat/completions \
  --requests 500 \
  --concurrency 32 \
  --output-dir results/load_tests
```

Run it from the repository root (it imports `evals.benchmark`). Output lands at
`<output-dir>/load_test_<UTC timestamp>.json` containing `endpoint`,
`num_requests`, `concurrency`, `successful`, `failed`, `avg_latency_ms`,
`p99_latency_ms`, `throughput_rps`, and `timestamp`. Files under
`results/load_tests/` are git-ignored.

### `triton/perf_analyzer.sh`

Runs NVIDIA `perf_analyzer` against a deployed Triton model. Configured by
environment variables, not flags:

```bash
MODEL_NAME=llama3 TRITON_URL=localhost:8000 CONCURRENCY=32 REQUEST_COUNT=500 \
  bash triton/perf_analyzer.sh
```

Defaults: `MODEL_NAME=llama3`, `TRITON_URL=localhost:8000`, `CONCURRENCY=4`,
`REQUEST_COUNT=100`. The script reports the 95th percentile and requires
`perf_analyzer` on `PATH` (it ships in the Triton SDK container).

### `vllm/client_test.py`

A standalone script that hits `http://localhost:8000/v1/chat/completions`
directly (bypassing the gateway) with hardcoded `CONCURRENCY = 10` and
`NUM_REQUESTS = 100`, printing average/p95 latency. Excluded from coverage in
`pyproject.toml`.
